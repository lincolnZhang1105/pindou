"""OCR 训练样本生成：现场渲染色号、颗数、坐标数字，再做退化，最后用和线上一样的 ink_crop 预处理。

数据来源（登记在 docs/05 §7）：
- 字体：bench/synth/fonts/ 下的 OFL 字体（下载和锁定见 bench/synth/fonts.lock.json）；
- 文字内容：色卡里的色号字符串（只用色号文字，**不用 palettes_unverified 的色值**）+ 随机生成的色号和数字；
- 背景色：palettes/build/palettes.json 的正式色卡颜色，或随机颜色。
不使用任何真实图纸（data/samples 和 benchmark 图片不得用于训练，AGENTS 规则 8）。
"""
from __future__ import annotations

import io
import json
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from training.ocr.common import CHARSET, ink_crop, norm_label, rgb_to_gray

REPO = Path(__file__).resolve().parents[2]
FONT_DIR = REPO / "bench" / "synth" / "fonts"
CLEAN_PALETTES = REPO / "palettes" / "build" / "palettes.json"
BENCH_PALETTES = REPO / "bench" / "palettes_unverified" / "build" / "palettes_bench.json"


@lru_cache(maxsize=1)
def font_files() -> list[Path]:
    files = sorted(p for p in FONT_DIR.glob("*/*.ttf"))
    # 中文字体只给水印用，拉丁文字用其余字体
    latin = [p for p in files if "noto-sans-sc" not in str(p) and "zcool" not in str(p)]
    if not latin:
        raise SystemExit("没有找到字体：先运行 bench/synth 的字体下载脚本")
    return latin


@lru_cache(maxsize=1)
def cjk_font() -> Path:
    return next(FONT_DIR.glob("noto-sans-sc/*.ttf"))


@lru_cache(maxsize=1)
def code_pool() -> list[str]:
    codes: set[str] = set()
    for p in (CLEAN_PALETTES, BENCH_PALETTES):
        if p.exists():
            for pal in json.loads(p.read_text(encoding="utf-8"))["palettes"]:
                for c in pal["colors"]:  # 只取色号文字
                    code = c["code"].upper()
                    if code and all(ch in CHARSET for ch in code) and len(code) <= 6:
                        codes.add(code)
    return sorted(codes)


@lru_cache(maxsize=1)
def color_pool() -> np.ndarray:
    cols = []
    if CLEAN_PALETTES.exists():
        for pal in json.loads(CLEAN_PALETTES.read_text(encoding="utf-8"))["palettes"]:
            cols += [c["rgb"] for c in pal["colors"]]
    return np.array(cols or [[200, 200, 200]], dtype=np.int32)


@lru_cache(maxsize=256)
def load_font(path: str, size: int, weight: int | None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight])
        except Exception:
            pass
    return f


def random_text(rng: random.Random) -> tuple[str, str, str]:
    """返回 (kind, 渲染文字, 标注)。"""
    r = rng.random()
    if r < 0.5:
        if rng.random() < 0.7:
            t = rng.choice(code_pool())
        else:
            letters = rng.choice(["A", "B", "C", "D", "E", "F", "G", "H", "M", "P", "Q", "R", "T", "Y", "ZG", "S", "W"])
            t = letters + str(rng.randint(1, 140 if len(letters) == 1 else 30))
        return "code", t, t
    if r < 0.8:
        n = int(round(np.exp(rng.uniform(0, np.log(2500)))))
        style = rng.random()
        if style < 0.45:
            t = f"x{n}"
        elif style < 0.55:
            t = f"×{n}"
        elif style < 0.7:
            t = str(n)
        elif style < 0.85:
            t = f"{n}{rng.choice(['颗', '个', '粒'])}"
        else:
            t = f"({n})"
        return "count", t, norm_label(t)
    n = rng.randint(1, 300) if rng.random() < 0.8 else rng.randint(1, 60)
    return "coord", str(n), str(n)


def contrast_fg(bg: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    if rng.random() < 0.8:
        base = 20 if lum > 140 else 245
        j = rng.randint(-15, 10)
        v = int(np.clip(base + j, 0, 255))
        return (v, v, v)
    # 偶尔用彩色或灰色字，但保证亮度差 ≥ 90
    for _ in range(20):
        c = [rng.randint(0, 255) for _ in range(3)]
        if abs(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] - lum) >= 90:
            return tuple(c)
    return (0, 0, 0) if lum > 128 else (255, 255, 255)


def render(rng: random.Random) -> tuple[np.ndarray, str, str]:
    kind, text, label = random_text(rng)
    # 含中文量词时用中文字体（它也带拉丁字母和数字）
    fpath = str(cjk_font()) if any(ord(ch) >= 0x2E80 for ch in text) else str(rng.choice(font_files()))
    size = rng.randint(12, 56)
    weight = rng.choice([None, 400, 500, 600, 700])
    font = load_font(fpath, size, weight)
    # 背景：色号多半画在色块上；颗数和坐标多半是白底，坐标偶尔在青色坐标栏里
    if kind == "code" and rng.random() < 0.75:
        bg = color_pool()[rng.randrange(len(color_pool()))]
    elif kind == "coord" and rng.random() < 0.2:
        bg = np.array([rng.randint(150, 200), rng.randint(220, 245), rng.randint(215, 240)])
    else:
        v = rng.randint(236, 255)
        bg = np.array([v, v, v])
    fg = contrast_fg(bg, rng)
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    mx, my = rng.randint(2, max(3, size // 2)), rng.randint(2, max(3, size // 2))
    W, H = tw + 2 * mx, th + 2 * my
    im = Image.new("RGB", (W, H), tuple(int(v) for v in bg))
    d = ImageDraw.Draw(im)
    d.text((mx - l, my - t), text, font=font, fill=fg)
    # 表格式图例：颗数格四周有黑线，裁剪时外扩的 pad 会把线带进来
    if rng.random() < 0.15:
        lw = rng.randint(1, 3)
        d.rectangle([0, 0, W - 1, H - 1], outline=(rng.randint(0, 60),) * 3, width=lw)
    # 水印：半透明的斜线或浅色块
    if rng.random() < 0.15:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        g = rng.randint(120, 220)
        a = rng.randint(25, 70)
        x0 = rng.randint(-W, W)
        od.line([(x0, 0), (x0 + H, H)], fill=(g, g, g, a), width=rng.randint(2, max(3, H // 4)))
        im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    arr = np.asarray(im)
    # 缩小（模拟 1080 宽的图纸里只有 6–20 像素高的字）
    scale = rng.uniform(1.0, 2.6)
    if scale > 1.05:
        nw, nh = max(4, int(round(W / scale))), max(4, int(round(H / scale)))
        interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        arr = cv2.resize(arr, (nw, nh), interpolation=interp)
    if rng.random() < 0.25:
        arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.6:
        q = rng.randint(45, 92)
        fmt = rng.choice(["JPEG", "WEBP"])
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt, quality=q)
        arr = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    gray = rgb_to_gray(arr)
    # 文字框：和线上一样用「和背景亮度差大的像素」的外接框（去掉边框线所在的最外两圈）
    bg_gray = int(np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])))
    diff = np.abs(gray.astype(np.int32) - bg_gray)
    core = diff[2:-2, 2:-2] if diff.shape[0] > 6 and diff.shape[1] > 6 else diff
    thr = max(30, int(0.4 * core.max()))
    ys, xs = np.nonzero(core >= thr)
    if len(xs) == 0:
        return render(rng)
    off = 2 if core is not diff else 0
    bx, by = xs.min() + off, ys.min() + off
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    # 框的抖动：连通块外接框偶尔多 / 少一个像素
    bx += rng.randint(-1, 1)
    by += rng.randint(-1, 1)
    bw += rng.randint(-1, 1)
    bh += rng.randint(-1, 1)
    box = (max(0, bx), max(0, by), max(2, bw), max(3, bh))
    return ink_crop(gray, box), label, kind


def make_set(n: int, seed: int) -> list[tuple[np.ndarray, str, str]]:
    rng = random.Random(seed)
    return [render(rng) for _ in range(n)]
