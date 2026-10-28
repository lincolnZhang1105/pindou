"""缩略图拼贴：抽 N 张合成样本拼成一张大图，供人工检查「和真实图纸风格是否接近」（T0.6 DoD）。

`--overlay` 会在缩略图上用红线画出真值网格（每 5 条画一条，外框全画），用来目测几何真值是否对齐。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from bench.common.formats import load_json
from bench.synth.fonts import FontBook
from bench.synth.geom import apply_pts


def _overlay(d: ImageDraw.ImageDraw, page: dict, s: float, ox: float, oy: float) -> None:
    H = np.array(page["homography"]) if "homography" in page else None
    xs = page.get("flat_x_lines", page["x_lines"])
    ys = page.get("flat_y_lines", page["y_lines"])

    def tf(pts):
        pts = np.asarray(pts, dtype=np.float64)
        if H is not None:
            pts = apply_pts(H, pts)
        return [(ox + x * s, oy + y * s) for x, y in pts]

    nx, ny = len(xs) - 1, len(ys) - 1
    for i, x in enumerate(xs):
        if i in (0, nx) or (page["col_offset"] + i) % 5 == 0:
            d.line(tf([[x, ys[0]], [x, ys[-1]]]), fill=(255, 0, 0), width=1)
    for j, y in enumerate(ys):
        if j in (0, ny) or (page["row_offset"] + j) % 5 == 0:
            d.line(tf([[xs[0], y], [xs[-1], y]]), fill=(255, 0, 0), width=1)


def make_sheet(in_dir: Path, n: int = 50, out: Path | None = None, ncols: int = 10, thumb: int = 300,
               overlay: bool = False, offset: int = 0) -> Path:
    in_dir = Path(in_dir)
    labels = sorted((in_dir / "labels").glob("*.json"))[offset:offset + n]
    if not labels:
        raise SystemExit(f"{in_dir}/labels 下没有真值文件")
    fonts = FontBook(allow_fallback=True)
    font = fonts.get(fonts.pick(np.random.default_rng(0), "ui") if not fonts.fallback else None, 13, "regular")
    cap_h = 34
    th = int(thumb * 1.3)
    nrows = math.ceil(len(labels) / ncols)
    sheet = Image.new("RGB", (ncols * (thumb + 8) + 8, nrows * (th + cap_h + 8) + 8), (64, 64, 64))
    d = ImageDraw.Draw(sheet)
    for i, lp in enumerate(labels):
        lab = load_json(lp)
        ex_p = in_dir / "extra" / f"{lab['image_id']}.extra.json"
        style = load_json(ex_p)["style"]["name"] if ex_p.exists() else "?"
        img_path = in_dir / "images" / lab["images"][0]["file"]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            s = min(thumb / W, th / H)
            t = im.resize((max(1, int(W * s)), max(1, int(H * s))), Image.LANCZOS)
        r, c = divmod(i, ncols)
        x0, y0 = 8 + c * (thumb + 8), 8 + r * (th + cap_h + 8)
        ox, oy = x0 + (thumb - t.width) // 2, y0 + (th - t.height) // 2
        d.rectangle([x0, y0, x0 + thumb - 1, y0 + th - 1], fill=(250, 250, 250))
        sheet.paste(t, (ox, oy))
        if overlay:
            for p in lab["pages"]:
                if p["image_index"] == 0:
                    _overlay(d, p, s, ox, oy)
        cap = f"{lab['image_id'][-6:]} {lab['type']} {style} {lab['grid']['cols']}x{lab['grid']['rows']}"
        cap2 = f"{len(lab['images'])}img {W}x{H} {','.join(t[:6] for t in lab.get('tags', []))}"
        d.text((x0 + 2, y0 + th + 2), cap, font=font, fill=(255, 255, 255))
        d.text((x0 + 2, y0 + th + 18), cap2, font=font, fill=(200, 200, 200))
    out = Path(out) if out else in_dir / ("contact_sheet_overlay.png" if overlay else "contact_sheet.png")
    sheet.save(out)
    return out
