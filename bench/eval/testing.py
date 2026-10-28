"""手工构造真值 / 预测结果 / 私有数据目录的小工具。只给 bench/tests/test_eval*.py 用，不参与评测。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from bench.common import formats as F
from bench.eval.config import CONFIG_PATH


def label(image_id: str, cells: list[list[str | None]], *, type: str = "T1", tags=(), legend=None,
          palette: str | None = "MARD-221", cell_px: float = 10.0) -> dict:
    rows, cols = len(cells), len(cells[0])
    lab = {
        "image_id": image_id, "type": type, "palette_hint": palette,
        "images": [{"file": f"{image_id}.png", "sha256": ""}],
        "grid": {"cols": cols, "rows": rows},
        "pages": [{"image_index": 0, "bbox": [0, 0, cols * cell_px, rows * cell_px],
                   "x_lines": [i * cell_px for i in range(cols + 1)], "y_lines": [i * cell_px for i in range(rows + 1)],
                   "col_offset": 0, "row_offset": 0, "cols": cols, "rows": rows}],
        "cells": cells, "tags": list(tags),
    }
    if legend is not None:
        lab["legend"] = [{"code": c, "count": n} for c, n in legend]
    if palette is None:
        del lab["palette_hint"]
    return lab


def pred(cells: list[list], *, palette: str = "MARD-221", confidence=None, uncertain=None, legend=None,
         grid: tuple[int, int] | None = None, timing: dict | None = None) -> dict:
    """cells 的元素：None（空格）、"A1"（palette 品牌）或 (palette_id, code)。grid=(cols, rows) 可覆盖网格尺寸。"""
    book: list[tuple[str, str]] = []
    flat: list[int] = []
    for row in cells:
        for v in row:
            if v is None:
                flat.append(0)
                continue
            e = v if isinstance(v, tuple) else (palette, v)
            if e not in book:
                book.append(e)
            flat.append(book.index(e) + 1)
    cols, rows = grid if grid else (len(cells[0]), len(cells))
    out = {"engine_version": "test", "grid": {"cols": cols, "rows": rows},
           "codebook": [{"palette_id": p, "code": c} for p, c in book], "cells": flat}
    if confidence is not None:
        out["confidence"] = list(confidence)
    if uncertain is not None:
        out["uncertain"] = list(uncertain)
    if legend is not None:
        out["legend"] = [{"palette_id": palette, "code": c, "count": n} for c, n in legend]
    if timing is not None:
        out["timing_ms"] = timing
    return out


def grid_of(code: str | None, cols: int, rows: int) -> list[list[str | None]]:
    return [[code] * cols for _ in range(rows)]


def write_png(path: Path, w: int, h: int, sharp: bool = True) -> None:
    """sharp=True：棋盘格（拉普拉斯方差很大）；False：纯灰（方差为 0）。"""
    if sharp:
        yy, xx = np.mgrid[0:h, 0:w]
        arr = (((xx // 2 + yy // 2) % 2) * 255).astype(np.uint8)
    else:
        arr = np.full((h, w), 128, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "L").convert("RGB").save(path)


class Env:
    """一个临时的私有数据目录 + bench/datasets 目录。用法：env = Env(tmp_path, monkeypatch)。"""

    def __init__(self, tmp: Path, monkeypatch, **cfg_overrides):
        self.root = tmp / "data"
        self.datasets = tmp / "datasets"
        for d in ("images", "labels", "preds", "reports"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("PINDOU_DATA_DIR", str(self.root))
        monkeypatch.setattr(F, "DATASETS", self.datasets)
        self.config = tmp / "config.yaml"
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["gate"]["baseline"] = str(tmp / "baseline_dev.json")
        raw["bootstrap"]["n"] = cfg_overrides.pop("bootstrap_n", 2000)
        for k, v in cfg_overrides.items():
            sec, key = k.split("__")
            raw[sec][key] = v
        self.config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")

    def add(self, lab: dict, sharp: bool = True) -> None:
        """登记一张图：写真值和原图（尺寸按网格线）。"""
        p = lab["pages"][0]
        w, h = int(np.ceil(p["x_lines"][-1])) + 4, int(np.ceil(p["y_lines"][-1])) + 4
        write_png(self.root / "images" / lab["images"][0]["file"], w, h, sharp)
        F.save_json(self.root / "labels" / f"{lab['image_id']}.json", lab)

    def put_pred(self, run: str, image_id: str, obj: dict) -> None:
        F.save_json(self.root / "preds" / run / f"{image_id}.json", obj)

    def split(self, name: str, ids: list[str]) -> None:
        (self.datasets / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    def lock(self, ids: list[str]) -> None:
        entries = [{"image_id": i, "sha256": F.sha256_file(self.root / "images" / f"{i}.png")} for i in sorted(ids)]
        from bench.eval.discipline import ids_sha256
        F.save_json(self.datasets / "test.lock", {"created_at": "2026-10-28T00:00:00+08:00",
                                                  "ids_sha256": ids_sha256(ids), "entries": entries})

    def usage(self) -> list[dict]:
        return F.read_jsonl(self.datasets / "test_usage.jsonl")

    def args(self, *a: str) -> list[str]:
        return list(a) + ["--config", str(self.config)]


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
