"""用 colour-science 生成 sRGB→Lab 与 CIEDE2000 的对拍数据（T0.2）。

输出 core/tests/data/colour_golden.csv，供 core/tests/test_color.cpp 对拍。
固定随机种子，结果可复现。colour-science 只在离线环境使用，不打包进 App（05 §2）。

用法：python -m bench.tools.gen_colour_golden [--n 10000]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import colour
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "core" / "tests" / "data" / "colour_golden.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rgb1 = rng.integers(0, 256, size=(args.n, 3))
    rgb2 = rng.integers(0, 256, size=(args.n, 3))
    # 一半样本让第二个颜色靠近第一个：拼豆里真正要区分的是相近色
    near = rng.random(args.n) < 0.5
    jitter = rng.integers(-12, 13, size=(args.n, 3))
    rgb2[near] = np.clip(rgb1[near] + jitter[near], 0, 255)
    # 边界样本：完全相同、纯灰、纯黑白
    rgb1[:4] = [[0, 0, 0], [255, 255, 255], [128, 128, 128], [10, 200, 30]]
    rgb2[:4] = [[0, 0, 0], [254, 254, 254], [128, 128, 128], [10, 200, 30]]

    with colour.utilities.domain_range_scale("reference"):
        lab1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb1 / 255.0))
        lab2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb2 / 255.0))
        de = colour.delta_E(lab1, lab2, method="CIE 2000")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["r1", "g1", "b1", "r2", "g2", "b2", "L1", "a1", "b1_", "L2", "a2", "b2_", "de2000"])
        for i in range(args.n):
            w.writerow([*rgb1[i], *rgb2[i], *(f"{v:.10f}" for v in lab1[i]), *(f"{v:.10f}" for v in lab2[i]),
                        f"{de[i]:.10f}"])
    print(f"写入 {OUT}（{args.n} 组，colour-science {colour.__version__}）")


if __name__ == "__main__":
    main()
