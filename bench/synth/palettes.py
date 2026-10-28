"""合成数据用的色卡：干净色值 + 色号文字表。

**硬性约束**（bench/README.md「色卡」、palettes/SOURCES.md §2）：合成数据会用来训练要打包进 App 的模型，
所以**不得使用 `bench/palettes_unverified` 里的色值**。这里的做法：

- `load_clean_palettes()` 只读 `palettes/build/palettes.json`，并且拒绝 `verified` 不是 true 的条目；
  合成器的所有色值只能来自这里（或随机生成）。
- `load_code_tables()` 读 `palettes_bench.json` 时**只取 code 字段**，返回的 `CodeTable` 里没有任何颜色属性，
  色值在解析时就丢弃，不会进入后续任何数据结构。

测试见 `bench/tests/test_synth_palettes.py`。
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from bench.common.color import srgb8_to_lab
from bench.common.formats import REPO

CLEAN_PATH = REPO / "palettes" / "build" / "palettes.json"
BENCH_PATH = REPO / "bench" / "palettes_unverified" / "build" / "palettes_bench.json"


@dataclass(frozen=True)
class CleanPalette:
    """来源干净、可以进 App 的色卡（色号 + 色值）。"""
    id: str
    brand: str
    codes: tuple[str, ...]
    rgb: np.ndarray  # (n, 3) uint8
    lab: np.ndarray  # (n, 3) float64

    def source_of(self, i: int) -> str:
        return f"palette:{self.id}:{self.codes[i]}"


@dataclass(frozen=True)
class CodeTable:
    """只有色号文字的表，**刻意不带任何色值**。"""
    id: str
    brand: str
    codes: tuple[str, ...]
    clean: bool  # True：同 id 的干净色卡存在，可以用「色号 + 原色值」一致的 native 方案


class PaletteError(RuntimeError):
    pass


def _read(path: Path) -> dict:
    if not Path(path).exists():
        raise PaletteError(f"找不到色卡文件 {path}（先运行 python -m palettes.build_palettes）")
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_clean_palettes(path: Path = CLEAN_PATH) -> dict[str, CleanPalette]:
    out: dict[str, CleanPalette] = {}
    for p in _read(path)["palettes"]:
        if p.get("verified") is not True:
            raise PaletteError(f"{path} 里的 {p.get('id')} 不是 verified 色卡，合成器拒绝使用它的色值")
        codes = tuple(c["code"] for c in p["colors"])
        rgb = np.array([c["rgb"] for c in p["colors"]], dtype=np.uint8).reshape(-1, 3)
        out[p["id"]] = CleanPalette(id=p["id"], brand=p.get("brand", ""), codes=codes, rgb=rgb,
                                    lab=srgb8_to_lab(rgb))
    return out


@lru_cache(maxsize=4)
def load_code_tables(bench_path: Path = BENCH_PATH, clean_path: Path = CLEAN_PATH) -> dict[str, CodeTable]:
    """所有色号文字表：干净色卡的色号 + bench 内部色卡（来源存疑）的色号。只读 code 字段。"""
    clean = load_clean_palettes(clean_path)
    out: dict[str, CodeTable] = {pid: CodeTable(id=pid, brand=p.brand, codes=p.codes, clean=True)
                                 for pid, p in clean.items()}
    if not Path(bench_path).exists():
        warnings.warn(f"找不到 {bench_path}，只用干净色卡的色号文字（python -m bench.palettes_unverified.import_unverified）")
        return out
    for p in _read(bench_path)["palettes"]:
        pid = p["id"]
        if pid in out:
            continue  # 同 id 的干净色卡已经在了
        # 只取 code，色值在这里就丢掉
        codes = tuple(str(c["code"]) for c in p["colors"])
        out[pid] = CodeTable(id=pid, brand=str(p.get("brand", "")), codes=codes, clean=False)
    return out


@dataclass(frozen=True)
class ColorPool:
    """所有干净色卡合并后的色值池（按 rgb 去重），用于「色号文字来自别的表」时取色。"""
    rgb: np.ndarray       # (n, 3) uint8
    lab: np.ndarray       # (n, 3)
    sources: tuple[str, ...]


@lru_cache(maxsize=4)
def clean_color_pool(clean_path: Path = CLEAN_PATH) -> ColorPool:
    seen: dict[tuple[int, int, int], str] = {}
    for pid in sorted(load_clean_palettes(clean_path)):
        p = load_clean_palettes(clean_path)[pid]
        for i in range(len(p.codes)):
            key = tuple(int(v) for v in p.rgb[i])
            if key not in seen:
                seen[key] = p.source_of(i)
    keys = sorted(seen)
    rgb = np.array(keys, dtype=np.uint8).reshape(-1, 3)
    return ColorPool(rgb=rgb, lab=srgb8_to_lab(rgb), sources=tuple(seen[k] for k in keys))


# 色号方案的抽样权重。
#   text 方案：色号文字取自该表（多为国内品牌，来源存疑的只用文字），颜色取自干净色值池或随机生成；
#   native 方案：色号和色值都取自同一套干净色卡（无图例、无格内文字的图纸只能用这种，否则原则上不可识别）。
TEXT_SCHEMES = {"MARD-221": 0.32, "MARD-291": 0.16, "COCO-291": 0.08, "PANPAN-289": 0.05, "MIXIAOWO-290": 0.05,
                "ARTKAL-S": 0.06, "ARTKAL-M": 0.05, "ARTKAL-C": 0.04, "ARTKAL-A": 0.03, "HAMA-MIDI": 0.06,
                "YANT": 0.04, "NABBI": 0.02, "ARTKAL-R": 0.02, "HAMA-MINI": 0.02}
NATIVE_SCHEMES = {"ARTKAL-S": 0.22, "ARTKAL-M": 0.16, "ARTKAL-C": 0.16, "ARTKAL-A": 0.1, "HAMA-MIDI": 0.12,
                  "YANT": 0.1, "PERLER": 0.06, "NABBI": 0.03, "ARTKAL-R": 0.05}


def choose_scheme(rng, tables: dict[str, CodeTable], native_only: bool, short_codes: bool) -> tuple[str, bool]:
    """返回 (色号表 id, 是否 native)。short_codes：格内要写色号时排除 Perler 这种 8 位长色号。"""
    if native_only or rng.random() < 0.3:
        w = {k: v for k, v in NATIVE_SCHEMES.items() if k in tables and tables[k].clean}
        if short_codes:
            w.pop("PERLER", None)
        native = True
    else:
        w = {k: v for k, v in TEXT_SCHEMES.items() if k in tables}
        native = False
    keys = sorted(w)
    p = np.array([w[k] for k in keys], dtype=np.float64)
    pid = keys[int(rng.choice(len(keys), p=p / p.sum()))]
    return pid, native and tables[pid].clean
