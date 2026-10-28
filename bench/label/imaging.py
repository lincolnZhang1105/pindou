"""标注工具的图像计算（服务端）：取色、按颜色自动填充初稿、网格线吸附。

坐标一律是「方向已校正」的原图像素（JPEG 按 EXIF 旋转；HEIC 用 collect convert 转出的 PNG），
与浏览器显示、core 的约定一致。图片按原分辨率读取，不缩放。
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from bench.common.color import delta_e_2000, srgb8_to_lab

_CACHE: OrderedDict = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 4


def load_rgb(path: Path) -> np.ndarray:
    """(H, W, 3) uint8，按 EXIF 方向校正，不缩放。最近用过的几张缓存在内存里。"""
    path = Path(path)
    key = (str(path), path.stat().st_mtime_ns)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    with Image.open(path) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            # 透明区域按白底合成（和浏览器里的观感接近）
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, rgba)
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    with _CACHE_LOCK:
        _CACHE[key] = arr
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return arr


def sample(img: np.ndarray, x: float, y: float, r: int = 4, tol: float = 20.0) -> list[int]:
    """取色：以 (x, y) 为中心、边长 2r+1 的方块里的「主色」。

    不用简单中位数：点在格子中心时，方块里大半是色号文字（抗锯齿后是一片灰），中位数会偏成灰色。
    这里找颜色最密集的像素（RGB 距离 tol 以内的邻居最多），返回这些邻居的中位数——也就是底色。"""
    h, w = img.shape[:2]
    xi, yi = int(math.floor(x)), int(math.floor(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError(f"取色点 ({x}, {y}) 在图片外")
    patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
    d = np.linalg.norm(patch[:, None, :] - patch[None, :, :], axis=2)
    near = d < tol
    best = int(np.argmax(near.sum(axis=1)))
    return [int(v) for v in np.median(patch[near[best]], axis=0).round()]


def _span(a: float, b: float, inner: float, limit: int) -> tuple[int, int]:
    c = (a + b) / 2.0
    half = max(0.5, (b - a) * inner / 2.0)
    lo = max(0, int(math.floor(c - half)))
    hi = min(limit, int(math.ceil(c + half)))
    if hi <= lo:
        hi = min(limit, lo + 1)
    return lo, hi


def cell_stats(img: np.ndarray, x_lines, y_lines, inner: float = 0.5, ink_thr: float = 80.0):
    """每格中心区域（边长 = 格宽 × inner）的颜色中位数和「墨迹比例」（离中位数颜色很远的像素占比，
    用来判断格内有没有色号文字：白豆和空格颜色一样，只能靠文字区分）。
    返回 med (rows, cols, 3) float、ink (rows, cols) float。"""
    h, w = img.shape[:2]
    cols, rows = len(x_lines) - 1, len(y_lines) - 1
    med = np.zeros((rows, cols, 3), dtype=np.float64)
    ink = np.zeros((rows, cols), dtype=np.float64)
    xs = [_span(x_lines[c], x_lines[c + 1], inner, w) for c in range(cols)]
    for r in range(rows):
        y0, y1 = _span(y_lines[r], y_lines[r + 1], inner, h)
        band = img[y0:y1].astype(np.float64)
        for c, (x0, x1) in enumerate(xs):
            patch = band[:, x0:x1].reshape(-1, 3)
            if patch.size == 0:
                continue
            m = np.median(patch, axis=0)
            med[r, c] = m
            ink[r, c] = float(np.mean(np.linalg.norm(patch - m, axis=1) > ink_thr))
    return med, ink


def assign(med: np.ndarray, ink: np.ndarray, swatches: list[tuple[str | None, list[int]]], *,
           max_de: float = 12.0, ambig_de: float = 2.0, twin_de: float = 4.0, ink_frac: float = 0.04):
    """按最近的色块（CIEDE2000）给每格定色号。swatches = [(code 或 None（空格）, rgb)]。

    颜色几乎一样的「空格」色块和「色号」色块（白豆 vs 空格）按墨迹比例区分：有文字 → 色号，没有 → 空格。
    返回 (codes: rows×cols 的 list，flags: [(r, c, 原因)])。原因：far（离所有色块都远）、
    ambiguous（最近的两个色块难区分）、text?（有无文字拿不准）。"""
    rows, cols = med.shape[:2]
    if not swatches:
        raise ValueError("没有色块：先在图例面板里给色号取色")
    sw_lab = srgb8_to_lab(np.array([s[1] for s in swatches], dtype=np.float64))
    cell_lab = srgb8_to_lab(med)
    de = delta_e_2000(cell_lab[:, :, None, :], sw_lab[None, None, :, :])   # (rows, cols, k)
    k = len(swatches)
    codes = [s[0] for s in swatches]
    # 每个色块的「孪生」色块（颜色几乎一样、一个是空格一个是色号）
    sw_de = delta_e_2000(sw_lab[:, None, :], sw_lab[None, :, :])
    twin_code = [None] * k   # 空格色块 → 与之孪生的色号色块下标
    twin_empty = [None] * k  # 色号色块 → 与之孪生的空格色块下标
    for i in range(k):
        for j in range(k):
            if i != j and sw_de[i, j] < twin_de:
                if codes[i] is None and codes[j] is not None and twin_code[i] is None:
                    twin_code[i] = j
                if codes[i] is not None and codes[j] is None and twin_empty[i] is None:
                    twin_empty[i] = j
    order = np.argsort(de, axis=2)
    out: list[list[str | None]] = []
    flags: list[tuple[int, int, str]] = []
    for r in range(rows):
        row = []
        for c in range(cols):
            b, s = int(order[r, c, 0]), int(order[r, c, 1]) if k > 1 else int(order[r, c, 0])
            d0 = float(de[r, c, b])
            choice = b
            resolved_twin = False
            e_idx = b if codes[b] is None else twin_empty[b]
            c_idx = twin_code[b] if codes[b] is None else (b if twin_empty[b] is not None else None)
            if e_idx is not None and c_idx is not None:
                choice = c_idx if ink[r, c] > ink_frac else e_idx
                resolved_twin = True
                if ink_frac * 0.5 < ink[r, c] < ink_frac * 2.0:
                    flags.append((r, c, "text?"))
            if d0 > max_de:
                flags.append((r, c, "far"))
            elif (not resolved_twin and k > 1 and codes[s] != codes[b]
                  and float(de[r, c, s]) - d0 < ambig_de):
                flags.append((r, c, "ambiguous"))
            row.append(codes[choice])
        out.append(row)
    return out, flags


# --------------------------------------------------------------------------- 网格线吸附

def _profile(gray: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis 方向的边缘强度剖面：axis=1 → 每个 x 位置（竖线），axis=0 → 每个 y 位置（横线）。
    位置 i 的值 = 像素 i 两侧的平均梯度（i-1→i 与 i→i+1），对 1–2 px 宽的网格线峰值落在线中心。"""
    d = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
    p = np.zeros(gray.shape[axis], dtype=np.float64)
    p[1:] += d
    p[:-1] += d
    k = np.array([1.0, 2.0, 1.0]) / 4.0
    return np.convolve(p, k, mode="same")


def _snap_axis(prof: np.ndarray, lines: list[float], frac: float) -> tuple[list[float], int]:
    """从第一条线开始依次吸附。每条线的搜索中心 = 原位置 + 前一条线的修正量，
    这样「越往后越对不齐」的累积漂移也能一路跟上。"""
    n = len(lines) - 1
    out = list(lines)
    moved = 0
    # 先估计整体平移：在 ±半格内找让所有线位置上边缘强度之和最大的偏移（对坐标数字等局部干扰不敏感）
    arr = np.asarray(lines, dtype=np.float64)
    half = 0.5 * (arr[-1] - arr[0]) / max(1, n)
    idx_axis = np.arange(len(prof), dtype=np.float64)
    cands = np.arange(-half, half + 1e-9, 0.25)
    scores = [float(np.interp(arr + s - 0.5, idx_axis, prof).sum()) for s in cands]
    shift = float(cands[int(np.argmax(scores))]) if len(cands) else 0.0
    for i in range(n + 1):
        x = lines[i] + shift
        cw = (lines[min(i + 1, n)] - lines[max(i - 1, 0)]) / (2 if 0 < i < n else 1)
        lo = max(0, int(math.floor(x - frac * cw)))
        hi = min(len(prof) - 1, int(math.ceil(x + frac * cw)))
        if hi - lo < 2:
            continue
        win = prof[lo:hi + 1]
        j = int(np.argmax(win))
        peak = win[j]
        base = float(np.median(win))
        if peak <= base * 1.5 + 1.0 or j == 0 or j == len(win) - 1:
            continue  # 窗口里没有明显的线（或峰在窗口边上），保持原位
        # 亚像素：峰值附近 ±2 px 的加权质心；+0.5 把像素下标换成连续坐标（像素 i 占 [i, i+1)）
        a, b = max(0, j - 2), min(len(win), j + 3)
        wts = np.clip(win[a:b] - base, 0, None)
        idx = np.arange(a, b)
        pos = lo + (float((wts * idx).sum() / wts.sum()) if wts.sum() > 0 else float(j)) + 0.5
        if abs(pos - lines[i]) > 1e-6:
            moved += 1
        out[i] = round(pos, 2)
        shift = out[i] - lines[i]
    # 去掉离群：吸附后某格宽偏离中位数 ±35% 的，相关的线恢复原位
    for _ in range(3):
        widths = np.diff(out)
        med = float(np.median(widths))
        bad = [i for i, wv in enumerate(widths) if wv < 0.65 * med or wv > 1.35 * med]
        if not bad:
            break
        for i in bad:
            for j in (i, i + 1):
                if out[j] != lines[j]:
                    out[j] = lines[j]
                    moved -= 1
    return out, max(moved, 0)


def snap_lines(img: np.ndarray, x_lines: list[float], y_lines: list[float], frac: float = 0.3):
    """把每条网格线吸附到附近 ±frac 格范围内边缘最强的位置（处理「越往后越对不齐」的累积漂移）。
    只在图纸区域内统计剖面：竖线用整个图纸高度的平均，格内文字的位置各不相同，会被平均掉。"""
    gray = img.astype(np.float64).mean(axis=2)
    h, w = gray.shape
    y0, y1 = max(0, int(y_lines[0])), min(h, int(math.ceil(y_lines[-1])) + 1)
    x0, x1 = max(0, int(x_lines[0])), min(w, int(math.ceil(x_lines[-1])) + 1)
    # 剖面只取图纸内部（去掉外框附近一格，避免坐标数字干扰）
    cy = (y_lines[-1] - y_lines[0]) / max(1, len(y_lines) - 1)
    cx = (x_lines[-1] - x_lines[0]) / max(1, len(x_lines) - 1)
    ya, yb = int(y0 + cy * 0.5), int(y1 - cy * 0.5)
    xa, xb = int(x0 + cx * 0.5), int(x1 - cx * 0.5)
    if yb - ya < 2 or xb - xa < 2:
        return list(x_lines), list(y_lines), 0
    px = _profile(gray[ya:yb, :], axis=1)
    py = _profile(gray[:, xa:xb], axis=0)
    nx, mx = _snap_axis(px, list(x_lines), frac)
    ny, my = _snap_axis(py, list(y_lines), frac)
    return nx, ny, mx + my
