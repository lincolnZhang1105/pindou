"""OCR 模型的公共约定：字符集、裁剪预处理。**预处理必须和 core/src/nn/ocr.cpp 逐步一致**，
否则训练分布和线上不一致（benchmark 结果就不等于线上效果，03 §1）。
"""
from __future__ import annotations

import cv2
import numpy as np

# 色号（字母 + 数字）、颗数（x534 / ×534 / 534 / 534颗 / (534)）、坐标数字。
# 标注规范化（norm_label）：× 写成 x；颗、个、粒这类中文量词统一写成 #；去掉空白。
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZx-()#"
BLANK = 0  # CTC blank 的下标；字符下标 = CHARSET.index(c) + 1
NUM_CLASSES = len(CHARSET) + 1

INPUT_H = 32
MIN_W, MAX_W = 16, 320


def norm_label(text: str) -> str | None:
    """渲染文字 → 训练标注；含字符集以外的字符时返回 None。"""
    out = []
    for ch in text:
        if ch.isspace():
            continue
        if ch in "×✕╳":
            ch = "x"
        elif ord(ch) >= 0x2E80:  # CJK 量词
            ch = "#"
        elif ch in "（":
            ch = "("
        elif ch in "）":
            ch = ")"
        if ch not in CHARSET:
            return None
        out.append(ch)
    return "".join(out) or None


def encode(text: str) -> list[int]:
    return [CHARSET.index(c) + 1 for c in text]


def decode_greedy(idx) -> str:
    out, prev = [], BLANK
    for k in idx:
        k = int(k)
        if k != prev and k != BLANK:
            out.append(CHARSET[k - 1])
        prev = k
    return "".join(out)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """与 core 的 rgb_to_gray 相同：Rec.601 整数权重，四舍五入。"""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return ((299 * r + 587 * g + 114 * b + 500) // 1000).astype(np.uint8)


def ink_crop(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """文字框 → 模型输入（高 32 的浮点「墨迹图」，0 = 背景，1 = 最浓的笔画）。

    1. 框向四周各扩 pad = max(2, round(0.15 × 框高))，截到图内；
    2. 背景亮度 = 裁剪区全部像素的中位数（排序后取第 n//2 个）。不用最外一圈：表格式图例的格线常常正好落在那一圈；
    3. 墨迹 = |灰度 − 背景|，除以 max(最大墨迹, 32)；
    4. 缩放到高 32，宽按比例并截到 [16, 320]；缩小用 INTER_AREA，放大用 INTER_LINEAR。
    """
    x, y, w, h = box
    pad = max(2, int(round(0.15 * h)))
    H, W = gray.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    c = gray[y0:y1, x0:x1].astype(np.int32)
    ch, cw = c.shape
    flat = np.sort(c.ravel())
    bg = int(flat[len(flat) // 2])
    ink = np.abs(c - bg).astype(np.float32)
    ink /= max(float(ink.max()), 32.0)
    nw = int(round(cw * INPUT_H / ch))
    nw = min(MAX_W, max(MIN_W, nw))
    interp = cv2.INTER_AREA if ch > INPUT_H else cv2.INTER_LINEAR
    return cv2.resize(ink, (nw, INPUT_H), interpolation=interp)
