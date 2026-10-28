"""Pindo 输出 → 评测器的预测结果格式（RecognitionResult 的子集，见 bench/README.md「预测结果」）。

纯函数，不依赖 Node，单测在 bench/tests/test_pindo_convert.py。
"""
from __future__ import annotations

from bench.eval.codes import brand_of

# 我们的品牌（palette_id 第一个 `-` 之前的部分）→ Pindo 的品牌 id。
# 只收录色号写法和我们一致的品牌（A01 与 A1 由评测器统一）。PERLER 两边编号体系不同
#（Pindo 用 P01…，我们用 80-15xxx），按颜色名只能对上 61/70，暂不支持。
BRANDS = {
    "MARD": "mard",
    "COCO": "coco",
    "MANMAN": "manman",
    "PANPAN": "panpan",
    "MIXIAOWO": "mixiaowo",
    "HAMA": "hama",
}
# ARTKAL 系列 Pindo 只有 S 系列的色卡文件
FULL_IDS = {"ARTKAL-S": "artkal-s"}
DEFAULT_BRAND = "mard"   # Pindo 网页的默认品牌
DEFAULT_SIZE = (35, 35)  # Pindo 网页默认的宽高（只在自动检测不到网格、按均分处理时用到）


def pindo_brand(palette_hint: str | None) -> str | None:
    """palette_hint → Pindo 品牌；没有 hint 时用 Pindo 的默认品牌；不支持的品牌返回 None。"""
    if not palette_hint:
        return DEFAULT_BRAND
    hint = palette_hint.strip().upper()
    if hint in FULL_IDS:
        return FULL_IDS[hint]
    return BRANDS.get(brand_of(hint) or "")


def size_hint(pages: list[dict], source: str, label: dict | None = None) -> tuple[tuple[int, int], str]:
    """自动检测失败时 Pindo 需要用户填的宽高。source = manifest：用登记时人工预估的 grid（没有就退回默认）；
    label：用真值网格（颜色路径上限对照）；default：一律用 Pindo 网页默认值。返回 ((cols, rows), 实际来源)。"""
    g = (label or {}).get("grid") if source == "label" else pages[0].get("grid") if source == "manifest" else None
    cols, rows = (g or {}).get("cols"), (g or {}).get("rows")
    if isinstance(cols, int) and isinstance(rows, int) and cols > 0 and rows > 0:
        return (cols, rows), source
    return DEFAULT_SIZE, "default"


def error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def to_prediction(out: dict, palette_id: str | None, white_as_empty: bool = False) -> dict:
    """Node 端的一行输出 → 预测结果。palette_id 写进 codebook（评测器只看品牌部分）。

    white_as_empty：把 Pindo 归到「纯白对应色号」的格子当成空格。Pindo 的识别没有空格概念，
    图纸里的空白格会被认成白豆；这个变体给出另一种读法，真正的白豆会因此被算成空格。"""
    if not out.get("ok"):
        return error("internal", f"Pindo 运行出错：{out.get('error', '未知错误')}")
    cols, rows, codes = out["cols"], out["rows"], out["codes"]
    if len(codes) != cols * rows:
        return error("internal", f"Pindo 输出的格子数 {len(codes)} ≠ {cols}×{rows}")
    white = out.get("white_code") if white_as_empty else None
    book: list[str] = []
    index: dict[str, int] = {}
    cells: list[int] = []
    for c in codes:
        if c is None or c == white:
            cells.append(0)
            continue
        k = index.get(c)
        if k is None:
            book.append(c)
            k = index[c] = len(book)
        cells.append(k)
    pid = palette_id or DEFAULT_BRAND.upper()
    return {
        "grid": {"cols": cols, "rows": rows},
        "codebook": [{"palette_id": pid, "code": c} for c in book],
        "cells": cells,
        # Pindo 不读图例：和 beadcore v0 一样输出空图例（评测器把「图例解析完全正确」记为否）
        "legend": [],
        "legend_check": {"ok": False, "diff": [], "unavailable": True},
        "timing_ms": {"total": out.get("ms", 0)},
        "pindo": {"grid_source": out.get("grid_source"), "white_code": out.get("white_code"),
                  "palette_size": out.get("palette_size"), "white_as_empty": white_as_empty},
    }
