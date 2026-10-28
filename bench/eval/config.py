"""读取并校验 bench/eval/config.yaml。参数含义见 config.yaml 里的注释和 bench/eval/README.md。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

from bench.common.formats import REPO, SIZE_BUCKETS, TYPES

CONFIG_PATH = Path(__file__).with_name("config.yaml")

# 进入参数指纹的配置节：它们决定确定性指标的取值（Hard/Clean 划分、标记上限等）
_FINGERPRINT_SECTIONS = ("blurry", "tau", "flag_cap", "scope", "hard")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    raw: dict
    path: Path
    # blurry
    min_cell_px: float
    theta: float
    theta_frozen: bool
    laplacian_ksize: int
    # τ
    tau: float
    # 标记上限
    flag_rate: Fraction
    flag_per_image_max: int
    # 口径
    main_excluded_types: tuple[str, ...]
    flag_legend_excluded_types: tuple[str, ...]
    # Hard
    hard_size_buckets: tuple[str, ...]
    hard_tags: tuple[str, ...]
    hard_types: tuple[str, ...]
    # bootstrap
    bootstrap_n: int
    seed: int
    confidence: float
    # go / no-go
    ratio_max: float
    clean_per_mille_max: float
    flag_recall_min: float
    grid_accuracy_min: float
    timing_ms_max: float
    legend_accuracy_min: float
    hard_images_target: int
    # gate
    gate_baseline: Path

    def fingerprint(self) -> str:
        """影响确定性指标的参数指纹（gate 基线和 test 报告都记录它）。"""
        part = {k: self.raw.get(k) for k in _FINGERPRINT_SECTIONS}
        return hashlib.sha256(json.dumps(part, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def flag_cap(self, n_beads: int) -> int:
        """单张图的不确定标记上限：min(floor(rate × 豆数), per_image_max)。用 Fraction 避免浮点取整误差。"""
        return min(int(self.flag_rate * n_beads), self.flag_per_image_max)  # int() 对非负 Fraction 即 floor

    def cap_rule_text(self) -> str:
        pct = float(self.flag_rate) * 100
        return (f"每张图上限 = min(floor({pct:g}% × 该图真值豆数), {self.flag_per_image_max})；"
                "超出时按 confidence 升序（相同时按格子编号升序；无 confidence 时只按格子编号）保留前「上限」个，其余视为未标记")


def _get(d: dict, path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise ConfigError(f"config 缺少 {path}")
        cur = cur[k]
    return cur


def _types(v: Any, name: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(v, list) or any(x not in allowed for x in v):
        raise ConfigError(f"{name} 只能取 {allowed} 中的值，实际为 {v!r}")
    return tuple(v)


def load_config(path: Path | None = None) -> Config:
    path = Path(path) if path else CONFIG_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} 不是 YAML 映射")
    rate = Fraction(str(_get(raw, "flag_cap.rate")))
    if not (0 < rate <= 1):
        raise ConfigError("flag_cap.rate 必须在 (0, 1] 内")
    per_image = int(_get(raw, "flag_cap.per_image_max"))
    if per_image < 0:
        raise ConfigError("flag_cap.per_image_max 不能为负")
    conf = float(_get(raw, "bootstrap.confidence"))
    if not (0 < conf < 1):
        raise ConfigError("bootstrap.confidence 必须在 (0, 1) 内")
    n_boot = int(_get(raw, "bootstrap.n"))
    if n_boot < 100:
        raise ConfigError("bootstrap.n 至少 100")
    baseline = Path(_get(raw, "gate.baseline"))
    if not baseline.is_absolute():
        baseline = REPO / baseline
    return Config(
        raw=raw, path=path,
        min_cell_px=float(_get(raw, "blurry.min_cell_px")),
        theta=float(_get(raw, "blurry.theta")),
        theta_frozen=bool(_get(raw, "blurry.theta_frozen")),
        laplacian_ksize=int(_get(raw, "blurry.laplacian_ksize")),
        tau=float(_get(raw, "tau")),
        flag_rate=rate,
        flag_per_image_max=per_image,
        main_excluded_types=_types(_get(raw, "scope.main_excluded_types"), "scope.main_excluded_types", TYPES),
        flag_legend_excluded_types=_types(_get(raw, "scope.flag_legend_excluded_types"),
                                          "scope.flag_legend_excluded_types", TYPES),
        hard_size_buckets=_types(_get(raw, "hard.size_buckets"), "hard.size_buckets", SIZE_BUCKETS),
        hard_tags=tuple(_get(raw, "hard.tags")),
        hard_types=_types(_get(raw, "hard.types"), "hard.types", TYPES),
        bootstrap_n=n_boot,
        seed=int(_get(raw, "bootstrap.seed")),
        confidence=conf,
        ratio_max=float(_get(raw, "gonogo.ratio_max")),
        clean_per_mille_max=float(_get(raw, "gonogo.clean_per_mille_max")),
        flag_recall_min=float(_get(raw, "gonogo.flag_recall_min")),
        grid_accuracy_min=float(_get(raw, "gonogo.grid_accuracy_min")),
        timing_ms_max=float(_get(raw, "gonogo.timing_ms_max")),
        legend_accuracy_min=float(_get(raw, "gonogo.legend_accuracy_min")),
        hard_images_target=int(_get(raw, "gonogo.hard_images_target")),
        gate_baseline=baseline,
    )
