"""汇总：把逐图结果按口径 / 分层聚合成指标（04 §3），并给出 bootstrap 置信区间。

每个指标都写成 scale × Σ分子 / Σ分母（分子、分母逐图给出），定义见 METRICS 和 bench/eval/README.md。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from bench.common.formats import KNOWN_TAGS, SIZE_BUCKETS, TYPES
from bench.eval.config import Config
from bench.eval.scoring import ImageResult
from bench.eval.stats import ratio_ci


@dataclass(frozen=True)
class MetricDef:
    key: str
    title: str
    scale: float
    per_image: Callable[[ImageResult], tuple[float, float]]  # (分子, 分母)
    higher_is_better: bool
    kind: str  # rate（显示为百分比）/ per_mille / ratio


def _ok(r: ImageResult) -> bool:
    return r.status == "ok"


METRICS: list[MetricDef] = [
    MetricDef("grid_accuracy", "网格正确率", 1.0, lambda r: (int(r.grid_ok), 1), True, "rate"),
    MetricDef("per_mille", "每千颗错误数（微平均）", 1000.0, lambda r: (r.errors, r.n_beads), False, "per_mille"),
    MetricDef("nonempty_acc_macro", "非空格准确率（宏平均）", 1.0,
              lambda r: ((r.nonempty_acc or 0.0), 1 if r.n_beads > 0 else 0), True, "rate"),
    MetricDef("zero_error_rate", "零错误率", 1.0, lambda r: (int(r.zero_error), 1), True, "rate"),
    MetricDef("failure_rate", "失败率", 1.0, lambda r: (int(r.failed), 1), False, "rate"),
    MetricDef("bom_error", "用量清单误差", 1.0, lambda r: (r.bom_abs, r.n_beads), False, "rate"),
    MetricDef("empty_to_bead_rate", "空格→豆子（/ 空格数）", 1.0,
              lambda r: ((r.empty_to_bead or 0), r.n_empty) if _ok(r) else (0, 0), False, "rate"),
    MetricDef("bead_to_empty_rate", "豆子→空格（/ 豆数）", 1.0,
              lambda r: ((r.bead_to_empty or 0), r.n_beads) if _ok(r) else (0, 0), False, "rate"),
    MetricDef("legend_accuracy", "图例解析完全正确率", 1.0,
              lambda r: (int(bool(r.legend_ok)), 1) if r.has_legend else (0, 0), True, "rate"),
    MetricDef("flag_recall", "不确定标记召回（上限内）", 1.0, lambda r: (r.flag_hits, r.errors), True, "rate"),
    MetricDef("flag_recall_ok_only", "不确定标记召回（上限内，只算成功图）", 1.0,
              lambda r: (r.flag_hits, r.errors) if _ok(r) else (0, 0), True, "rate"),
    MetricDef("flag_precision", "不确定标记精确率", 1.0,
              lambda r: (r.flag_hits, r.flag_kept) if _ok(r) else (0, 0), True, "rate"),
    MetricDef("flag_rate", "标记率（截断后）", 1.0,
              lambda r: (r.flag_kept, r.n_beads) if _ok(r) else (0, 0), False, "rate"),
    MetricDef("flag_rate_raw", "标记率（引擎原始）", 1.0,
              lambda r: (r.flag_raw, r.n_beads) if _ok(r) else (0, 0), False, "rate"),
]
METRIC_BY_KEY = {m.key: m for m in METRICS}


@dataclass
class Metric:
    key: str
    value: float | None
    num: float
    den: float
    ci_low: float | None
    ci_high: float | None
    n_images: int

    def to_dict(self) -> dict:
        return asdict(self)


def _num(x: float) -> float | int:
    return int(x) if float(x).is_integer() else float(x)


def compute(results: list[ImageResult], cfg: Config, keys: list[str] | None = None, ci: bool = True) -> dict[str, Metric]:
    out: dict[str, Metric] = {}
    for m in METRICS:
        if keys is not None and m.key not in keys:
            continue
        pairs = [m.per_image(r) for r in results]
        num = np.array([p[0] for p in pairs], dtype=np.float64)
        den = np.array([p[1] for p in pairs], dtype=np.float64)
        sn, sd = num.sum(), den.sum()
        val = m.scale * sn / sd if sd > 0 else None
        lo = hi = None
        if ci and results and sd > 0:
            lo, hi = ratio_ci(num, den, cfg.bootstrap_n, cfg.seed, cfg.confidence, m.scale)
        out[m.key] = Metric(m.key, None if val is None else float(val), _num(sn), _num(sd), lo, hi, len(results))
    return out


# --------------------------------------------------------------------------- 分组

SCOPE_TITLES = {
    "main": "主统计池（不含 T6、T7）",
    "flag_legend": "第 3、6 条口径（不含 T7）",
    "hard": "Hard 集",
    "clean": "Clean 集",
    "all": "全部图纸",
}


def groups(results: list[ImageResult]) -> list[tuple[str, str, list[ImageResult]]]:
    """返回 [(分组类型, 分组名, 结果列表)]。尺寸档、标签分层只在主统计池里做；类型分层覆盖全部图纸（T6、T7 在这里单独报告）。"""
    main = [r for r in results if r.in_main]
    out: list[tuple[str, str, list[ImageResult]]] = [
        ("scope", "main", main),
        ("scope", "flag_legend", [r for r in results if r.in_flag_scope]),
        ("scope", "hard", [r for r in main if r.hard]),
        ("scope", "clean", [r for r in main if not r.hard]),
        ("scope", "all", list(results)),
    ]
    for b in SIZE_BUCKETS:
        out.append(("size", b, [r for r in main if r.size_bucket == b]))
    for t in TYPES:
        out.append(("type", t, [r for r in results if r.type == t]))
    tags = sorted({t for r in main for t in r.tags}, key=lambda t: (t not in KNOWN_TAGS, KNOWN_TAGS.index(t) if t in KNOWN_TAGS else 0, t))
    for tg in tags:
        out.append(("tag", tg, [r for r in main if tg in r.tags]))
    out.append(("tag", "（无标签）", [r for r in main if not r.tags]))
    return out


def summarize(results: list[ImageResult], cfg: Config, ci: bool = True) -> list[dict]:
    """全部分组 × 全部指标的长表（summary.csv 的内容）。"""
    rows: list[dict] = []
    for kind, name, rs in groups(results):
        ms = compute(rs, cfg, ci=ci)
        for m in METRICS:
            x = ms[m.key]
            rows.append({"group_kind": kind, "group": name, "metric": m.key, "value": x.value,
                         "ci_low": x.ci_low, "ci_high": x.ci_high, "num": x.num, "den": x.den,
                         "n_images": x.n_images})
    return rows
