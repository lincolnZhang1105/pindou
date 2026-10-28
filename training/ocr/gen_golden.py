"""生成 core 单元测试用的对拍数据：ink_crop 的 Python 结果（以及有模型时的模型输出）。

    python -m training.ocr.gen_golden [--model core/models/ocr_v1.onnx]

输出 core/tests/data/ocr_golden.json：一张确定性的小灰度图、若干文字框（覆盖缩小 / 放大、0.5 舍入边界），
每个框的 ink_crop 结果；给了模型时再附上模型输出的 log 概率（C++ 与 PyTorch / ORT 误差要 < 1e-3）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.ocr.common import ink_crop

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "core" / "tests" / "data" / "ocr_golden.json"


def make_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    img = np.full((120, 200), 240, np.uint8)
    img[10:40, 10:90] = 60          # 深色块
    img[50:60, 100:190] = 200       # 浅色条
    img[70:110, 20:60] = rng.integers(0, 255, size=(40, 40), dtype=np.uint8)  # 噪声块
    for k in range(0, 200, 7):      # 竖线
        img[:, k] = 30
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, help="ONNX 模型（用 onnxruntime 求参考输出）")
    ap.add_argument("--checkpoint", type=Path, help="PyTorch 检查点（用 PyTorch 求参考输出，T0.11 的对拍口径）")
    ap.add_argument("--onnx-name", default="", help="写进 golden 的模型名，C++ 测试按它找 core/models/<name>.onnx")
    a = ap.parse_args()
    img = make_image()
    # 框高 10 → pad 1.5（五成双取 2），框高 30 → 4.5（取 4）；宽高比造成的 .5 也覆盖到
    boxes = [(12, 12, 40, 10), (5, 5, 60, 30), (20, 60, 30, 45), (100, 48, 80, 14), (0, 0, 16, 20),
             (150, 90, 49, 29), (60, 15, 21, 8)]
    items = []
    sess = None
    torch_model = None
    if a.model:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(a.model), providers=["CPUExecutionProvider"])
    if a.checkpoint:
        import torch
        from training.ocr.model import CRNN
        torch_model = CRNN()
        torch_model.load_state_dict(torch.load(a.checkpoint, map_location="cpu")["model"])
        torch_model.eval()
    for b in boxes:
        crop = ink_crop(img, b)
        it = {"box": list(b), "w": int(crop.shape[1]), "crop": [round(float(v), 7) for v in crop.ravel()]}
        if sess is not None or torch_model is not None:
            if torch_model is not None:
                import torch
                with torch.no_grad():
                    out = torch_model(torch.from_numpy(crop[None, None].astype(np.float32))).numpy()
            else:
                out = sess.run(None, {"x": crop[None, None].astype(np.float32)})[0]
            it["logp_shape"] = list(out.shape)
            it["logp"] = [round(float(v), 6) for v in out.ravel()]
        items.append(it)
    OUT.write_text(json.dumps({"image_shape": list(img.shape), "image": img.ravel().tolist(), "model": a.onnx_name,
                               "items": items}) + "\n",
                   encoding="utf-8")
    print(f"写入 {OUT}（{len(items)} 个框{'，含模型输出' if (sess or torch_model) else ''}）")


if __name__ == "__main__":
    main()
