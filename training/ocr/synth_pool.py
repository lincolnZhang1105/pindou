"""从 bench/synth 生成的整页合成图里切出文字裁剪（图例色号、颗数、坐标数字），作为 OCR 的第二类训练样本。

整页合成图经过了排版、缩放到 1080 宽、WebP/JPEG 压缩、水印，比单独渲染的文字更接近线上输入。
文字框按线上的方式重新求：在标注框附近取「和背景亮度差大的像素」的外接框，再走同一个 ink_crop。

    python -m training.ocr.synth_pool --src training/data/synth_ocr --out training/data/ocr_pool.npz

只用合成数据（不含任何真实图纸，AGENTS 规则 8）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.ocr.common import ink_crop, norm_label, rgb_to_gray


def tight_box(gray: np.ndarray, box) -> tuple[int, int, int, int] | None:
    x0, y0, x1, y1 = box
    H, W = gray.shape
    # 不向外扩：坐标数字挨得很近，外扩会把邻居的半个字形算进框里（标注却只有中间这个数）
    ex0, ey0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    ex1, ey1 = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
    if ex1 - ex0 < 2 or ey1 - ey0 < 2:
        return None
    region = gray[ey0:ey1, ex0:ex1].astype(np.int32)
    bg = int(np.sort(region.ravel())[region.size // 2])
    diff = np.abs(region - bg)
    thr = max(30, int(0.4 * diff.max()))
    ys, xs = np.nonzero(diff >= thr)
    if len(xs) == 0:
        return None
    return (ex0 + int(xs.min()), ey0 + int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def cell_items(ex: dict, label: dict, image_index: int, rng: np.random.Generator, per_image: int) -> list:
    """格内色号：按真值网格线和附加标注里的相对文字框（单位是格距）求出每格的文字框。"""
    rel = ex.get("cell_text_boxes") or {}
    has_text = ex.get("cells", {}).get("has_text")
    if not rel or not has_text:
        return []
    out = []
    for page in label["pages"]:
        if page["image_index"] != image_index or "homography" in page:
            continue
        xs, ys = page["x_lines"], page["y_lines"]
        cand = []
        for r in range(page["rows"]):
            for c in range(page["cols"]):
                gr, gc = r + page["row_offset"], c + page["col_offset"]
                code = label["cells"][gr][gc]
                if code is None or has_text[gr][gc] != "1":
                    continue
                code = code.split(":")[-1]
                if code not in rel:
                    continue
                pw, ph = xs[c + 1] - xs[c], ys[r + 1] - ys[r]
                if min(pw, ph) < 12:
                    continue
                b = rel[code]
                cand.append(("cell", (xs[c] + b[0] * pw, ys[r] + b[1] * ph, xs[c] + b[2] * pw, ys[r] + b[3] * ph), code))
        if cand:
            pick = rng.choice(len(cand), size=min(per_image, len(cand)), replace=False)
            out += [cand[i] for i in pick]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells-per-image", type=int, default=25, help="每张图抽多少个格内色号裁剪（0 = 不抽）")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    crops, labels, kinds = [], [], []
    extras = sorted((a.src / "extra").glob("*.json"))
    for i, ep in enumerate(extras):
        ex = json.loads(ep.read_text(encoding="utf-8"))
        for im in ex["images"]:
            path = a.src / "images" / im["file"]
            if not path.exists():
                continue
            gray = rgb_to_gray(np.asarray(Image.open(path).convert("RGB")))
            items = []
            for it in im.get("legend", []):
                if it.get("code_box") and it.get("code_text"):
                    items.append(("code", it["code_box"], it["code_text"]))
                if it.get("count_box") and it.get("count_text"):
                    items.append(("count", it["count_box"], it["count_text"]))
            for c in im.get("coords", []):
                items.append(("coord", c["box"], c["text"]))
            if a.cells_per_image > 0:
                lab_path = a.src / "labels" / f"{ex['image_id']}.json"
                if lab_path.exists():
                    label = json.loads(lab_path.read_text(encoding="utf-8"))
                    items += cell_items(ex, label, ex["images"].index(im), rng, a.cells_per_image)
            for kind, box, text in items:
                label = norm_label(text)
                tb = tight_box(gray, box)
                if label is None or tb is None or tb[3] < 3:
                    continue
                crop = ink_crop(gray, tb)
                crops.append(np.clip(np.rint(crop * 255), 0, 255).astype(np.uint8))
                labels.append(label)
                kinds.append(kind)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(extras)} 页，{len(crops)} 个裁剪", flush=True)
    widths = np.array([c.shape[1] for c in crops], np.int32)
    flat = np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, flat=flat, widths=widths, labels=np.array(labels), kinds=np.array(kinds))
    print(f"写入 {a.out}：{len(crops)} 个裁剪（{dict(zip(*np.unique(kinds, return_counts=True)))}）")


if __name__ == "__main__":
    main()
