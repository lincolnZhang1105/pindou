"""生成测速用的标准图纸（T0.14）：104×104 格铺满 1080 宽（每格约 10 像素，04 §1.1 的典型大图），
带细网格线、每 5 格粗线、横排图例。内容是程序生成的，固定种子，每次结果相同。

    python tools/make_bench_image.py --cols 104 --rows 104 --out build/bench_104.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

COLORS = [((236, 197, 172), "G3"), ((0, 0, 0), "H7"), ((171, 35, 47), "F8"), ((249, 238, 114), "A4"),
          ((98, 114, 52), "B11"), ((15, 82, 189), "C8"), ((226, 224, 236), "D16"), ((137, 133, 141), "H4"),
          ((240, 154, 127), "F1"), ((82, 37, 34), "F11"), ((211, 111, 124), "F9"), ((221, 160, 61), "G6")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=104)
    ap.add_argument("--rows", type=int, default=104)
    ap.add_argument("--width", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    margin = 30
    pitch = (a.width - 2 * margin) / a.cols
    legend_h = 140
    H = int(margin * 2 + a.rows * pitch + legend_h)
    im = Image.new("RGB", (a.width, H), "white")
    d = ImageDraw.Draw(im)
    # 内容：几个椭圆色块叠在一起，外面留白
    cells = [[None] * a.cols for _ in range(a.rows)]
    for _ in range(9):
        cx, cy = rng.uniform(0.2, 0.8) * a.cols, rng.uniform(0.2, 0.8) * a.rows
        rx, ry = rng.uniform(0.1, 0.3) * a.cols, rng.uniform(0.1, 0.3) * a.rows
        k = rng.randrange(len(COLORS))
        for r in range(a.rows):
            for c in range(a.cols):
                if ((c - cx) / rx) ** 2 + ((r - cy) / ry) ** 2 <= 1:
                    cells[r][c] = k
    X = lambda i: margin + i * pitch  # noqa: E731
    Y = lambda j: margin + j * pitch  # noqa: E731
    counts = [0] * len(COLORS)
    for r in range(a.rows):
        for c in range(a.cols):
            k = cells[r][c]
            if k is None:
                continue
            counts[k] += 1
            d.rectangle([X(c), Y(r), X(c + 1), Y(r + 1)], fill=COLORS[k][0])
    for i in range(a.cols + 1):
        d.line([X(i), Y(0), X(i), Y(a.rows)], fill=(180, 160, 230) if i % 5 == 0 else (200, 200, 200), width=2 if i % 5 == 0 else 1)
    for j in range(a.rows + 1):
        d.line([X(0), Y(j), X(a.cols), Y(j)], fill=(180, 160, 230) if j % 5 == 0 else (200, 200, 200), width=2 if j % 5 == 0 else 1)
    font_path = next(Path(__file__).resolve().parents[1].glob("bench/synth/fonts/inter/*.ttf"), None)
    font = ImageFont.truetype(str(font_path), 18) if font_path else ImageFont.load_default()
    x = margin
    y0 = int(Y(a.rows)) + 30
    for k, (rgb, code) in enumerate(COLORS):
        if counts[k] == 0:
            continue
        d.rounded_rectangle([x, y0, x + 56, y0 + 56], radius=8, fill=rgb)
        tc = (255, 255, 255) if sum(rgb) < 300 else (0, 0, 0)
        d.text((x + 28, y0 + 28), code, font=font, fill=tc, anchor="mm")
        d.text((x + 28, y0 + 78), f"x{counts[k]}", font=font, fill=(0, 0, 0), anchor="mm")
        x += 80
    a.out.parent.mkdir(parents=True, exist_ok=True)
    im.save(a.out)
    print(f"写入 {a.out}：{a.cols}×{a.rows} 格，{im.size[0]}×{im.size[1]} 像素，每格 {pitch:.1f} 像素")


if __name__ == "__main__":
    main()
