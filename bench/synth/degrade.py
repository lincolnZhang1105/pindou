"""退化（04 §2.3）：水印、截图边框、缩放、模糊、有损压缩、拍照（透视、摩尔纹、光照、色偏、噪声）。

每个改变几何的操作都返回 3×3 矩阵 M（输入图坐标 → 输出图坐标，连续像素坐标，见 geom.py），
由调用方累乘，再用 `Ann.transformed` 同步更新真值和附加标注里的所有坐标。

水印只用程序生成的文字和简单图标（圆、星、爱心、圆角方块），**不模仿任何真实平台的 logo**。
截图边框也是通用的手机界面元素（状态栏、导航栏、底部按钮），不模仿具体 App。
"""
from __future__ import annotations

import io
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

from bench.synth.fonts import FontBook
from bench.synth.geom import cv2_matrix, mat_scale, mat_translate

# --------------------------------------------------------------------------- 文字素材（全部通用词，不含任何平台名）
_SYL = ["dou", "bean", "pix", "mi", "tu", "xiao", "ya", "le", "bo", "ka", "momo", "lulu", "nini", "pupu", "kiki",
        "tang", "guo", "mao", "zz", "qi"]
_CN_NAME = ["豆豆", "小豆", "像素", "糖糖", "团子", "布丁", "饭团", "芝士", "奶盖", "橘子", "小满", "木木"]
_CN_TAIL = ["的豆屋", "拼豆", "手作", "工作室", "的图纸", "日记", "小铺"]
_CN_NOTE = ["原创图纸 禁止商用", "禁止二改搬运", "仅供个人使用", "图纸原创", "盗图必究"]
_UI_TITLE = ["图纸预览", "详情", "图片", "我的图纸", "作品", "预览"]
_UI_BTN = ["保存", "分享", "收藏", "编辑", "导出", "评论"]


def _pick(rng, xs):
    return xs[int(rng.integers(len(xs)))]


def watermark_text(rng, cjk: bool) -> str:
    r = rng.random()
    if not cjk or r < 0.35:
        name = "".join(_pick(rng, _SYL) for _ in range(int(rng.integers(1, 3))))
        if rng.random() < 0.6:
            name += str(int(rng.integers(1, 9999)))
        return "@" + name
    if r < 0.7:
        return _pick(rng, _CN_NAME) + _pick(rng, _CN_TAIL)
    if r < 0.85:
        return _pick(rng, _CN_NOTE)
    return f"ID:{int(rng.integers(10_000_000, 99_999_999))}"


def _icon(d: ImageDraw.ImageDraw, rng, x: float, y: float, s: float, fill) -> None:
    kind = int(rng.integers(4))
    if kind == 0:
        d.ellipse([x, y, x + s, y + s], outline=fill, width=max(1, int(s / 8)))
    elif kind == 1:
        d.rounded_rectangle([x, y, x + s, y + s], radius=int(s / 4), outline=fill, width=max(1, int(s / 8)))
    elif kind == 2:
        pts = []
        for k in range(10):
            r = s / 2 if k % 2 == 0 else s / 4.5
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((x + s / 2 + r * math.cos(a), y + s / 2 + r * math.sin(a)))
        d.polygon(pts, outline=fill, width=max(1, int(s / 10)))
    else:
        t = np.linspace(0, 2 * np.pi, 40)
        hx = 16 * np.sin(t) ** 3
        hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        pts = [(x + s / 2 + s / 36 * a, y + s / 2 + s / 36 * b) for a, b in zip(hx, hy)]
        d.polygon(pts, outline=fill, width=max(1, int(s / 10)))


# --------------------------------------------------------------------------- 水印

def watermark(img: np.ndarray, rng: np.random.Generator, fonts: FontBook) -> tuple[np.ndarray, dict]:
    """平铺半透明水印（文字 + 简单图标，旋转，错行平铺），可选细斜线。几何不变。"""
    H, W = img.shape[:2]
    cands = fonts.candidates("watermark")
    face = cands[int(rng.integers(len(cands)))] if cands else None
    text = watermark_text(rng, face is not None and face.cjk)
    size = int(max(10, W * rng.uniform(0.018, 0.04)))
    font = fonts.get(face, size, "regular")
    tw = int(font.getlength(text)) + 4
    icon_s = int(size * rng.uniform(1.0, 1.8)) if rng.random() < 0.7 else 0
    tile_w = max(tw, icon_s) + int(size * rng.uniform(2, 6))
    tile_h = icon_s + int(size * 1.4) + int(size * rng.uniform(2, 6))
    tile = Image.new("L", (tile_w, tile_h), 0)
    d = ImageDraw.Draw(tile)
    y = 2
    if icon_s:
        _icon(d, rng, (tile_w - icon_s) / 2, y, icon_s, 255)
        y += icon_s + size * 0.2
    d.text((tile_w / 2, y), text, font=font, fill=255, anchor="mt")
    angle = float(rng.uniform(-35, 35)) if rng.random() < 0.7 else 0.0
    tile = tile.rotate(angle, resample=Image.BICUBIC, expand=True)
    ta = np.asarray(tile, dtype=np.float32) / 255.0
    th, tw2 = ta.shape
    mask = np.zeros((H, W), dtype=np.float32)
    stagger = rng.random() < 0.6
    ox, oy = int(rng.integers(0, tw2)), int(rng.integers(0, th))
    for j, y0 in enumerate(range(-oy, H, th)):
        shift = (tw2 // 2) if (stagger and j % 2) else 0
        for x0 in range(-ox - shift, W, tw2):
            xa, ya = max(0, x0), max(0, y0)
            xb, yb = min(W, x0 + tw2), min(H, y0 + th)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] = np.maximum(mask[ya:yb, xa:xb], ta[ya - y0:yb - y0, xa - x0:xb - x0])
    lines = rng.random() < 0.3
    if lines:  # 细斜线（3_0 那种）
        yy, xx = np.mgrid[0:H, 0:W]
        per = W * rng.uniform(0.15, 0.35)
        lw = max(1.0, W / 900)
        dd = np.abs(((xx + yy) % per) - per / 2)
        mask = np.maximum(mask, (dd < lw / 2).astype(np.float32) * 0.7)
    alpha = float(rng.uniform(0.08, 0.32))
    r = rng.random()
    color = (np.array([int(rng.integers(120, 200))] * 3) if r < 0.6 else
             np.array([255, 255, 255]) if r < 0.75 else np.array([int(rng.integers(30, 90))] * 3) if r < 0.9 else
             rng.integers(80, 220, 3))
    a = (mask * alpha)[..., None]
    out = img.astype(np.float32) * (1 - a) + color.astype(np.float32) * a
    if r >= 0.6 and r < 0.75:  # 白字水印通常带一点暗描边
        edge = cv2.dilate(mask, np.ones((3, 3), np.uint8)) - mask
        e = (np.clip(edge, 0, 1) * alpha * 0.6)[..., None]
        out = out * (1 - e) + 60.0 * e
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), {
        "text": text, "font": face.id if face else None, "alpha": round(alpha, 3), "angle": round(angle, 2),
        "icon": bool(icon_s), "diag_lines": lines}


# --------------------------------------------------------------------------- 缩放、模糊、压缩

def resize(img: np.ndarray, width: int, height: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    H, W = img.shape[:2]
    if height is None:
        height = max(1, int(round(H * width / W)))
    interp = cv2.INTER_AREA if width < W else cv2.INTER_CUBIC
    out = cv2.resize(img, (int(width), int(height)), interpolation=interp)
    return out, mat_scale(width / W, height / H)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def encode(img: np.ndarray, fmt: str, quality: int | None = None) -> bytes:
    """编码成文件字节（png / webp / jpg）。参数固定，保证同样输入得到逐字节相同的输出。"""
    im = Image.fromarray(img)
    buf = io.BytesIO()
    if fmt == "png":
        im.save(buf, format="PNG", compress_level=6)
    elif fmt == "webp":
        im.save(buf, format="WEBP", quality=int(quality), method=4)
    elif fmt == "jpg":
        im.save(buf, format="JPEG", quality=int(quality), subsampling=2 if quality < 90 else 0, optimize=False)
    else:
        raise ValueError(fmt)
    return buf.getvalue()


def decode(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


# --------------------------------------------------------------------------- 截图边框（T5）

def screenshot(img: np.ndarray, rng: np.random.Generator, fonts: FontBook) -> tuple[np.ndarray, np.ndarray, dict, list]:
    """把图纸放进一张通用的手机截图里：状态栏、导航栏、（可选）底部按钮栏。"""
    H0, W0 = img.shape[:2]
    Wp = int(_pick(rng, [1080, 1080, 1170, 1125, 1242, 1440]))
    Hp = int(round(Wp * rng.uniform(2.0, 2.22)))
    dark = rng.random() < 0.3
    bg = (0, 0, 0) if dark else ((255, 255, 255) if rng.random() < 0.6 else (242, 242, 246))
    fg = (255, 255, 255) if dark else (20, 20, 20)
    canvas = Image.new("RGB", (Wp, Hp), bg)
    d = ImageDraw.Draw(canvas)
    cands = fonts.candidates("ui")
    face = cands[int(rng.integers(len(cands)))] if cands else None
    cjk = face is not None and face.cjk
    sb = int(Wp * rng.uniform(0.045, 0.06))
    nav = int(Wp * rng.uniform(0.1, 0.13)) if rng.random() < 0.85 else 0
    bot = int(Wp * rng.uniform(0.11, 0.15)) if rng.random() < 0.6 else 0
    texts = []
    # 状态栏
    f_sb = fonts.get(face, int(sb * 0.5), "medium")
    t = f"{int(rng.integers(0, 24))}:{int(rng.integers(0, 60)):02d}"
    d.text((Wp * 0.06, sb / 2), t, font=f_sb, fill=fg, anchor="lm")
    texts.append({"kind": "ui", "text": t, "box": list(d.textbbox((Wp * 0.06, sb / 2), t, font=f_sb, anchor="lm"))})
    bx = Wp * 0.94
    d.rounded_rectangle([bx - sb * 0.9, sb * 0.32, bx, sb * 0.68], radius=int(sb * 0.08), outline=fg, width=2)
    d.rectangle([bx - sb * 0.85, sb * 0.37, bx - sb * (0.85 - 0.8 * rng.uniform(0.2, 1.0)), sb * 0.63], fill=fg)
    for k in range(4):
        hh = sb * (0.15 + 0.1 * k)
        x = bx - sb * 2.2 + k * sb * 0.18
        d.rectangle([x, sb * 0.7 - hh, x + sb * 0.11, sb * 0.7], fill=fg)
    # 导航栏
    if nav:
        y = sb + nav / 2
        d.line([(Wp * 0.06, y), (Wp * 0.06 + nav * 0.18, y - nav * 0.18)], fill=fg, width=max(2, Wp // 300))
        d.line([(Wp * 0.06, y), (Wp * 0.06 + nav * 0.18, y + nav * 0.18)], fill=fg, width=max(2, Wp // 300))
        title = _pick(rng, _UI_TITLE) if cjk else _pick(rng, ["Preview", "Pattern", "Photo", "Details"])
        f_t = fonts.get(face, int(nav * 0.32), "medium")
        d.text((Wp / 2, y), title, font=f_t, fill=fg, anchor="mm")
        texts.append({"kind": "ui", "text": title, "box": list(d.textbbox((Wp / 2, y), title, font=f_t, anchor="mm"))})
        for k in range(3):
            cx = Wp * 0.9 + (k - 1) * nav * 0.12
            d.ellipse([cx - 3, y - 3, cx + 3, y + 3], fill=fg)
    # 底部按钮
    if bot:
        y0 = Hp - bot
        d.line([(0, y0), (Wp, y0)], fill=(80, 80, 80) if dark else (225, 225, 225), width=1)
        n = int(rng.integers(3, 6))
        f_b = fonts.get(face, int(bot * 0.2), "regular")
        for k in range(n):
            cx = Wp * (k + 0.5) / n
            s = bot * 0.28
            _icon(d, rng, cx - s / 2, y0 + bot * 0.18, s, fg)
            lab = _pick(rng, _UI_BTN) if cjk else _pick(rng, ["Save", "Share", "Like", "Edit"])
            d.text((cx, y0 + bot * 0.72), lab, font=f_b, fill=fg, anchor="mm")
            texts.append({"kind": "ui", "text": lab, "box": list(d.textbbox((cx, y0 + bot * 0.72), lab, font=f_b, anchor="mm"))})
    # 内容区
    top, bottom = sb + nav, Hp - bot
    mx = int(Wp * rng.uniform(0, 0.05))
    avail_w, avail_h = Wp - 2 * mx, (bottom - top) - int(Wp * rng.uniform(0, 0.06))
    s = min(avail_w / W0, avail_h / H0)
    nw, nh = max(1, int(round(W0 * s))), max(1, int(round(H0 * s)))
    small, M_s = resize(img, nw, nh)
    ox = (Wp - nw) // 2
    oy = top + (bottom - top - nh) // 2 if rng.random() < 0.7 else top + int((bottom - top - nh) * rng.uniform(0, 0.3))
    arr = np.asarray(canvas, dtype=np.uint8).copy()
    arr[oy:oy + nh, ox:ox + nw] = small
    M = mat_translate(ox, oy) @ M_s
    return arr, M, {"phone": [Wp, Hp], "dark": dark, "content_box": [ox, oy, ox + nw, oy + nh]}, texts


# --------------------------------------------------------------------------- 拍照（T6）

def _smooth_noise(rng, h: int, w: int, scale: int) -> np.ndarray:
    small = rng.random((max(2, h // scale), max(2, w // scale))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def photo(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    """模拟拍照：打印件（纸边）或屏幕翻拍（黑边 + 摩尔纹），透视、光照不均、色偏、模糊、噪声。"""
    H0, W0 = img.shape[:2]
    screen = rng.random() < 0.4
    info: dict = {"mode": "screen" if screen else "print"}
    # 1. 纸边 / 屏幕黑边
    padf = rng.uniform(0.02, 0.08)
    pad = int(max(H0, W0) * padf)
    border = (np.array([10, 10, 12]) if screen else np.array([246, 244, 238]) + rng.integers(-8, 5, 3)).clip(0, 255)
    paper = np.empty((H0 + 2 * pad, W0 + 2 * pad, 3), dtype=np.uint8)
    paper[:] = border.astype(np.uint8)
    paper[pad:pad + H0, pad:pad + W0] = img
    M = mat_translate(pad, pad)
    # 2. 先缩到照片尺度
    out_long = int(rng.uniform(1400, 2200))
    ph, pw = paper.shape[:2]
    s0 = out_long * rng.uniform(0.7, 0.9) / max(ph, pw)
    paper, Ms = resize(paper, max(1, int(round(pw * s0))), max(1, int(round(ph * s0))))
    M = Ms @ M
    ph, pw = paper.shape[:2]
    if ph >= pw:
        Ho, Wo = out_long, int(out_long * rng.uniform(0.7, 0.8))
    else:
        Wo, Ho = out_long, int(out_long * rng.uniform(0.7, 0.8))
    # 3. 透视：纸的四角 → 抖动后的四边形
    ang = math.radians(rng.uniform(-10, 10))
    sc = min(Wo / pw, Ho / ph) * rng.uniform(0.82, 0.95)
    cx, cy = Wo / 2 + rng.uniform(-0.04, 0.04) * Wo, Ho / 2 + rng.uniform(-0.04, 0.04) * Ho
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float64)
    dst = []
    jit = rng.uniform(0.02, 0.08)
    for x, y in src:
        x1, y1 = (x - pw / 2) * sc, (y - ph / 2) * sc
        xr = x1 * math.cos(ang) - y1 * math.sin(ang)
        yr = x1 * math.sin(ang) + y1 * math.cos(ang)
        dst.append([cx + xr + rng.uniform(-jit, jit) * pw * sc, cy + yr + rng.uniform(-jit, jit) * ph * sc])
    dst = np.array(dst, dtype=np.float64)
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)).astype(np.float64)
    Mcv = cv2_matrix(Hm)
    # 4. 背景（桌面）
    base = rng.integers(60, 200, 3).astype(np.float32)
    tex = _smooth_noise(rng, Ho, Wo, int(rng.integers(20, 80)))[..., None]
    back = np.clip(base * (0.8 + 0.4 * tex) + rng.normal(0, 4, (Ho, Wo, 1)), 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(paper, Mcv, (Wo, Ho), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wmask = cv2.warpPerspective(np.full(paper.shape[:2], 255, np.uint8), Mcv, (Wo, Ho), flags=cv2.INTER_LINEAR)
    a = (wmask.astype(np.float32) / 255.0)[..., None]
    out = warped.astype(np.float32) * a + back.astype(np.float32) * (1 - a)
    M = Hm @ M
    yy, xx = np.mgrid[0:Ho, 0:Wo].astype(np.float32)
    # 5. 摩尔纹（屏幕翻拍）
    if screen and rng.random() < 0.85:
        f1 = rng.uniform(0.18, 0.35)
        t1 = rng.uniform(0, np.pi)
        f2 = f1 * rng.uniform(0.94, 1.06)
        t2 = t1 + rng.uniform(-0.08, 0.08)
        amp = rng.uniform(0.04, 0.12)
        moire = np.zeros((Ho, Wo, 3), np.float32)
        for ch in range(3):
            ph_ = ch * rng.uniform(0.3, 1.2)
            g = (np.cos(2 * np.pi * f1 * (xx * np.cos(t1) + yy * np.sin(t1)) + ph_) *
                 np.cos(2 * np.pi * f2 * (xx * np.cos(t2) + yy * np.sin(t2))))
            moire[..., ch] = g
        out = out * (1 + amp * moire * a)
        info["moire"] = round(float(amp), 3)
    # 6. 光照不均 + 暗角
    gx_, gy_ = rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)
    light = 1 + gx_ * (xx / Wo - 0.5) + gy_ * (yy / Ho - 0.5)
    r2 = ((xx / Wo - 0.5) ** 2 + (yy / Ho - 0.5) ** 2)
    light *= 1 - rng.uniform(0.0, 0.5) * r2
    light *= rng.uniform(0.85, 1.08)
    out = out * light[..., None]
    # 7. 色偏、gamma
    gains = rng.uniform(0.88, 1.12, 3).astype(np.float32)
    out = out * gains
    gamma = rng.uniform(0.85, 1.15)
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    info.update({"gains": [round(float(g), 3) for g in gains], "gamma": round(float(gamma), 3)})
    # 8. 模糊（失焦 / 轻微运动）
    sigma = rng.uniform(0.3, 1.8)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(sigma))
    if rng.random() < 0.2:
        k = int(rng.integers(3, 8))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), float(rng.uniform(0, 180)), 1.0)
        ker = cv2.warpAffine(ker, rot, (k, k))
        ker /= max(ker.sum(), 1e-6)
        out = cv2.filter2D(out, -1, ker)
        info["motion_blur"] = k
    info["blur_sigma"] = round(float(sigma), 3)
    # 9. 传感器噪声
    out = out + rng.normal(0, rng.uniform(1, 5), out.shape).astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), M, info
