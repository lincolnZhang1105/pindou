"""两次运行在同一批图上的配对比较（04 §3）：对每张图的错误格数差做 bootstrap 重采样。

Δ = 每千颗错误数(A) − 每千颗错误数(B)。Δ < 0 表示 A 的错误更少；
95% CI 整个在 0 以下 → A 显著更好；整个在 0 以上 → B 显著更好；跨过 0 → 不显著。
"""
from __future__ import annotations

from bench.common.formats import SIZE_BUCKETS
from bench.eval.config import Config
from bench.eval.report import md_table
from bench.eval.runner import UsageError
from bench.eval.scoring import ImageResult
from bench.eval.stats import paired_per_mille_diff


def pair_results(ra: list[ImageResult], rb: list[ImageResult]) -> tuple[list[tuple[ImageResult, ImageResult]], list[str]]:
    a = {r.image_id: r for r in ra}
    b = {r.image_id: r for r in rb}
    warns: list[str] = []
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        warns.append(f"两次运行的图纸集合不同，只比较共同的 {len(set(a) & set(b))} 张"
                     f"（只在 A：{only_a[:5]}；只在 B：{only_b[:5]}）")
    pairs = []
    for i in sorted(set(a) & set(b)):
        x, y = a[i], b[i]
        if x.label_sha != y.label_sha:
            raise UsageError(f"{i}: 两次运行用的真值不同（label sha256 不一致），不能配对比较")
        if (x.hard, x.in_main, x.size_bucket, x.n_beads) != (y.hard, y.in_main, y.size_bucket, y.n_beads):
            raise UsageError(f"{i}: 两次运行的 Hard/口径划分不同（评测参数变了？），不能配对比较")
        pairs.append((x, y))
    return pairs, warns


def verdict(d: dict) -> str:
    if d["ci_low"] is None or d["ci_high"] is None:
        return "—"
    if d["ci_high"] < 0:
        return "A 显著更好"
    if d["ci_low"] > 0:
        return "B 显著更好"
    return "不显著"


def paired_groups(pairs: list[tuple[ImageResult, ImageResult]], cfg: Config) -> list[tuple[str, dict]]:
    main = [p for p in pairs if p[0].in_main]
    groups = [("主统计池（不含 T6、T7）", main),
              ("Hard 集", [p for p in main if p[0].hard]),
              ("Clean 集", [p for p in main if not p[0].hard])]
    groups += [(f"尺寸档 {b}", [p for p in main if p[0].size_bucket == b]) for b in SIZE_BUCKETS]
    out = []
    for name, ps in groups:
        d = paired_per_mille_diff([p[0].errors for p in ps], [p[1].errors for p in ps], [p[0].n_beads for p in ps],
                                  cfg.bootstrap_n, cfg.seed, cfg.confidence)
        d["zero_a"] = sum(p[0].zero_error for p in ps)
        d["zero_b"] = sum(p[1].zero_error for p in ps)
        d["grid_a"] = sum(p[0].grid_ok for p in ps)
        d["grid_b"] = sum(p[1].grid_ok for p in ps)
        out.append((name, d))
    return out


def _f(v, nd=3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def render(name_a: str, name_b: str, split: str, rows: list[tuple[str, dict]], warns: list[str], cfg: Config) -> str:
    L = [f"# 配对比较：A = {name_a}，B = {name_b}（{split} 集）", "",
         f"Δ = 每千颗错误数(A) − 每千颗错误数(B)，**Δ < 0 表示 A 更好**。配对 bootstrap：{cfg.bootstrap_n} 次，"
         f"按图纸重采样（两组用同一批重采样的图），种子 {cfg.seed}，{cfg.confidence * 100:g}% 百分位区间。", ""]
    for w in warns:
        L.append(f"- 警告：{w}")
    if warns:
        L.append("")
    table = []
    for name, d in rows:
        table.append([name, str(d["n"]), str(d["beads"]), _f(d["pm_a"]), _f(d["pm_b"]), _f(d["diff"]),
                      f"[{_f(d['ci_low'])}, {_f(d['ci_high'])}]", verdict(d),
                      f"{d['zero_a']}/{d['zero_b']}", f"{d['grid_a']}/{d['grid_b']}"])
    L.append(md_table(["口径", "图数", "豆数", "‰ A", "‰ B", "Δ（A−B）", "95% CI", "结论", "零错误图数 A/B",
                       "网格正确图数 A/B"], table))
    L.append("")
    return "\n".join(L)
