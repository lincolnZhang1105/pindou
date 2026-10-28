"""dev 集质量门禁（03 §3.6）：四个确定性指标和基线比，都不允许变差；耗时只报告。

比较用分子 / 分母的精确分数（fractions.Fraction），不受浮点误差影响。
"""
from __future__ import annotations

import statistics
from fractions import Fraction

from bench.eval.report import md_table
from bench.eval.scoring import ImageResult

# (key, 标题, 口径, 越大越好)
GATE_METRICS = [
    ("grid_accuracy", "网格正确率", "主统计池", True),
    ("zero_error_rate", "零错误率", "主统计池", True),
    ("clean_per_mille", "Clean 集每千颗错误数", "Clean", False),
    ("flag_recall", "不确定标记召回（上限内）", "不含 T7", True),
]


def gate_metrics(results: list[ImageResult]) -> dict[str, dict]:
    main = [r for r in results if r.in_main]
    clean = [r for r in main if not r.hard]
    fl = [r for r in results if r.in_flag_scope]

    def m(num: int, den: int, scale: float = 1.0) -> dict:
        return {"num": int(num), "den": int(den), "value": (scale * num / den) if den else None}

    return {
        "grid_accuracy": m(sum(r.grid_ok for r in main), len(main)),
        "zero_error_rate": m(sum(r.zero_error for r in main), len(main)),
        "clean_per_mille": m(sum(r.errors for r in clean), sum(r.n_beads for r in clean), 1000.0),
        "flag_recall": m(sum(r.flag_hits for r in fl), sum(r.errors for r in fl)),
    }


def _frac(key: str, x: dict) -> Fraction | None:
    if x["den"] == 0:
        # 没有错误格时召回视为 1（所有错误都被标出了，空真）；其他指标分母为 0 表示口径为空
        return Fraction(1) if key == "flag_recall" else None
    return Fraction(x["num"], x["den"])


def timing_median(results: list[ImageResult]) -> float | None:
    t = [r.timing_ms for r in results if r.timing_ms is not None and r.status == "ok"]
    return float(statistics.median(t)) if t else None


def compare_to_baseline(base: dict, cur: dict) -> list[dict]:
    """逐项比较，返回 [{key, title, base, cur, regressed}]。"""
    out = []
    for key, title, scope, higher in GATE_METRICS:
        fb, fc = _frac(key, base[key]), _frac(key, cur[key])
        if fb is None and fc is None:
            regressed = False
        elif fb is None or fc is None:
            regressed = True  # 口径从有到无（或反过来）：数据集变了，按退化处理
        else:
            regressed = fc < fb if higher else fc > fb
        out.append({"key": key, "title": title, "scope": scope, "base": base[key], "cur": cur[key],
                    "regressed": regressed})
    return out


def _v(key: str, x: dict) -> str:
    if x["den"] == 0:
        return "1（无错误格）" if key == "flag_recall" else "—"
    if key == "clean_per_mille":
        return f"{1000.0 * x['num'] / x['den']:.3f}（{x['num']}/{x['den']}）"
    return f"{100.0 * x['num'] / x['den']:.2f}%（{x['num']}/{x['den']}）"


def render(rows: list[dict], base_timing: float | None, cur_timing: float | None) -> str:
    table = [[r["title"], r["scope"], _v(r["key"], r["base"]), _v(r["key"], r["cur"]),
              "**退化**" if r["regressed"] else "通过"] for r in rows]
    s = md_table(["指标", "口径", "基线", "当前", "结论"], table)
    bt = "—" if base_timing is None else f"{base_timing:.0f} ms"
    ct = "—" if cur_timing is None else f"{cur_timing:.0f} ms"
    return s + f"\n\n耗时中位数（只报告，不设门禁）：基线 {bt}，当前 {ct}\n"
