"""bench 共用的数据格式读写与校验：真值（04 §2.2）、预测结果（03 §3.2）、manifest。

各子模块（collect / label / synth / eval）都通过这里读写数据，不要各自再写一套解析。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "bench" / "datasets"
# 正式 benchmark 的 manifest 在仓库里。试用流程、给合成集做标注时用环境变量 PINDOU_MANIFEST 指到别处，
# 免得试用数据混进正式 manifest（collect、label 都按它读写）
MANIFEST = Path(os.environ["PINDOU_MANIFEST"]) if os.environ.get("PINDOU_MANIFEST") else DATASETS / "manifest.jsonl"

TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
SIZE_BUCKETS = ("S", "M", "L", "XL")
SOURCE_PATHS = ("app_save", "own_export", "photo")
KNOWN_TAGS = ("similar_colors", "single_pixel_detail", "gradient_skin", "white_vs_empty", "watermark", "blurry")


# --------------------------------------------------------------------------- 路径

def data_dir(required: bool = True) -> Path | None:
    """私有数据根目录（$PINDOU_DATA_DIR）。"""
    v = os.environ.get("PINDOU_DATA_DIR")
    if not v:
        if required:
            raise SystemExit("请设置环境变量 PINDOU_DATA_DIR，指向私有存储里的 benchmark 数据（见 bench/README.md）")
        return None
    return Path(v)


# --------------------------------------------------------------------------- 通用

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, obj: Any, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- 色号

_CODE_RE = re.compile(r"^([A-Z]+)0*(\d+)$")


def normalize_code(code: str) -> str:
    """与 core 的 bc::normalize_code 一致：去空白、转大写、去掉数字部分的前导 0（A01 → A1）。"""
    s = "".join(code.split()).upper()
    m = _CODE_RE.match(s)
    return f"{m.group(1)}{int(m.group(2))}" if m else s


def code_key(code: str | None, default_palette: str | None = None) -> str | None:
    """真值里的色号可能写成 `palette_id:code`（混用其他品牌）。比较时用规范化后的色号；
    显式带了品牌前缀的保留前缀（大写），以区分不同品牌的同名色号。"""
    if code is None:
        return None
    if ":" in code:
        pal, c = code.split(":", 1)
        return f"{pal.strip().upper()}:{normalize_code(c)}"
    return normalize_code(code)


def size_bucket_of(cols: int, rows: int, paged: bool = False) -> str:
    """04 §2.1：按边长（取长边）分档；分页图一律 XL。"""
    if paged:
        return "XL"
    n = max(cols, rows)
    if n <= 40:
        return "S"
    if n <= 80:
        return "M"
    if n <= 130:
        return "L"
    return "XL"


# --------------------------------------------------------------------------- 真值

def label_cells(label: dict) -> np.ndarray:
    """返回 (rows, cols) 的 object 数组，元素为规范化色号或 None。"""
    rows, cols = label["grid"]["rows"], label["grid"]["cols"]
    arr = np.empty((rows, cols), dtype=object)
    cells = label["cells"]
    for r in range(rows):
        for c in range(cols):
            arr[r, c] = code_key(cells[r][c])
    return arr


def count_codes(cells: np.ndarray) -> Counter:
    return Counter(v for v in cells.ravel() if v is not None)


def validate_label(label: dict) -> list[str]:
    """结构校验 + 图例校验和（04 §2.2 第 3 步）。返回错误列表，空列表表示通过。"""
    errs: list[str] = []
    for k in ("image_id", "type", "grid", "pages", "cells", "images"):
        if k not in label:
            errs.append(f"缺少字段 {k}")
    if errs:
        return errs
    if label["type"] not in TYPES:
        errs.append(f"type 不合法：{label['type']}")
    rows, cols = label["grid"].get("rows"), label["grid"].get("cols")
    if not (isinstance(rows, int) and isinstance(cols, int) and rows > 0 and cols > 0):
        return errs + ["grid.rows / grid.cols 必须是正整数"]
    cells = label["cells"]
    if len(cells) != rows or any(len(r) != cols for r in cells):
        errs.append(f"cells 尺寸应为 {rows}×{cols}")
        return errs
    for i, p in enumerate(label["pages"]):
        pc, pr = p.get("cols"), p.get("rows")
        if len(p.get("x_lines", [])) != (pc or 0) + 1:
            errs.append(f"pages[{i}].x_lines 应有 cols+1 = {(pc or 0) + 1} 个值")
        if len(p.get("y_lines", [])) != (pr or 0) + 1:
            errs.append(f"pages[{i}].y_lines 应有 rows+1 = {(pr or 0) + 1} 个值")
        if p.get("col_offset", 0) + (pc or 0) > cols or p.get("row_offset", 0) + (pr or 0) > rows:
            errs.append(f"pages[{i}] 超出整张图纸范围")
        if not (0 <= p.get("image_index", -1) < len(label["images"])):
            errs.append(f"pages[{i}].image_index 越界")
    legend = label.get("legend")
    if legend:
        counts = count_codes(label_cells(label))
        expected = Counter()
        for item in legend:
            expected[code_key(item["code"])] += int(item["count"])
        for code in sorted(set(counts) | set(expected), key=str):
            if counts.get(code, 0) != expected.get(code, 0):
                errs.append(f"图例校验不通过：{code} 图例 {expected.get(code, 0)} 颗，格子里 {counts.get(code, 0)} 颗")
    return errs


# --------------------------------------------------------------------------- 预测结果

def is_error(pred: dict) -> bool:
    return "error" in pred


def pred_cells(pred: dict) -> np.ndarray:
    """把 RecognitionResult 的一维 cells + codebook 还原成 (rows, cols) 的规范化色号数组。"""
    rows, cols = pred["grid"]["rows"], pred["grid"]["cols"]
    book = [normalize_code(e["code"]) for e in pred["codebook"]]
    flat = pred["cells"]
    if len(flat) != rows * cols:
        raise ValueError(f"cells 长度 {len(flat)} ≠ {rows}×{cols}")
    arr = np.empty(rows * cols, dtype=object)
    for i, k in enumerate(flat):
        arr[i] = None if k == 0 else book[k - 1]
    return arr.reshape(rows, cols)


def pred_confidence(pred: dict) -> np.ndarray | None:
    conf = pred.get("confidence")
    if conf is None:
        return None
    rows, cols = pred["grid"]["rows"], pred["grid"]["cols"]
    return np.asarray(conf, dtype=np.float64).reshape(rows, cols)


def pred_to_label_cells(pred: dict) -> list[list[str | None]]:
    """预测结果 → 真值格式的二维 cells（标注工具导入初稿时用）。"""
    arr = pred_cells(pred)
    return [[arr[r, c] for c in range(arr.shape[1])] for r in range(arr.shape[0])]


# --------------------------------------------------------------------------- manifest

def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    return {r["image_id"]: r for r in read_jsonl(path)}


def read_split(name: str) -> list[str]:
    """读取 bench/datasets/<name>.txt（每行一个 image_id；分页图写 group id 下的每一页）。"""
    p = DATASETS / f"{name}.txt"
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
