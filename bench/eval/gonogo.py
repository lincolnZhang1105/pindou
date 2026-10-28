"""go / no-go 判定（04 §0）。

- 第 1、2、4 条同时达标 → GO；否则 NO-GO。第 3、5、6 条只报告（MVP 上线门槛）。
- 第 1 条：在「Hard 集 ∩ 主统计池 ∩ 最强竞品有 Tier B 结果」的图上配对比较：
  我方每千颗错误数 ≤ ratio_max × 竞品，且 Δ = 竞品 − 我方 的配对 bootstrap 95% CI 下界 > 0。
- 最强竞品：Tier A 里用量清单误差（微平均，失败按「没有输出」计）最小的竞品条目，
  在主统计池里所有条目都覆盖到的图上比较。
- 门槛比较用精确分数，避免浮点误差把边界情况判反。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from bench.eval.competitors import CompetitorDataError, TierAEntry, TierAResult, score_tier_a, unknown_images
from bench.eval.config import Config
from bench.eval.dataset import Unit
from bench.eval.report import md_table
from bench.eval.scoring import ImageResult
from bench.eval.stats import paired_per_mille_diff, ratio_ci


@dataclass
class Criterion:
    no: int
    title: str
    scope: str
    value: str
    threshold: str
    passed: bool | None
    required: bool
    notes: list[str] = field(default_factory=list)


def _q(x: float) -> Fraction:
    return Fraction(str(x))


def tier_a_table(entries: dict[str, TierAEntry], units: list[Unit], ours: dict[str, ImageResult]) -> dict:
    """Tier A 计分与最强竞品。只用主统计池的图。"""
    if not entries:
        raise CompetitorDataError("没有任何 Tier A 数据（--tier-a 或 $PINDOU_DATA_DIR/competitors/*/tier_a*.csv）")
    all_ids = {u.image_id for u in units}
    for e in entries.values():
        bad = unknown_images(e, all_ids)
        if bad:
            raise CompetitorDataError(f"Tier A {e.entry_id} 里有不属于 test 集的 image_id：{bad[:10]}")
    main = [u for u in units if u.in_main]
    scores = {eid: score_tier_a(e, main) for eid, e in entries.items()}
    common = sorted(set.intersection(*(set(s) for s in scores.values())))
    if not common:
        raise CompetitorDataError("各 Tier A 竞品条目在主统计池里没有共同覆盖的图纸，无法确定最强竞品")
    warns: list[str] = []
    rows = []
    for eid in sorted(scores):
        s = scores[eid]
        if len(s) < len(main):
            warns.append(f"Tier A {eid} 只覆盖了主统计池 {len(main)} 张中的 {len(s)} 张")
        cs: list[TierAResult] = [s[i] for i in common]
        beads = sum(r.n_beads for r in cs)
        rows.append({
            "entry": eid, "app_version": entries[eid].app_version, "tested_at": entries[eid].tested_at,
            "covered": len(s), "n": len(cs),
            "failures": sum(r.status != "ok" for r in cs),
            "grid_ok": sum(r.grid_ok for r in cs),
            "bom_abs": sum(r.bom_abs for r in cs), "bom_abs_raw": sum(r.bom_abs_raw for r in cs), "beads": beads,
        })
    for r in rows:
        r["bom_error"] = r["bom_abs"] / r["beads"] if r["beads"] else None
        r["bom_error_raw"] = r["bom_abs_raw"] / r["beads"] if r["beads"] else None
    key = lambda r: (Fraction(r["bom_abs"], r["beads"] or 1), Fraction(r["bom_abs_raw"], r["beads"] or 1), r["entry"])  # noqa: E731
    ranked = sorted(rows, key=key)
    strongest = ranked[0]["entry"]
    ties = [r["entry"] for r in ranked[1:] if Fraction(r["bom_abs"], r["beads"] or 1) == Fraction(ranked[0]["bom_abs"], ranked[0]["beads"] or 1)]
    if ties:
        warns.append(f"用量清单误差并列最小：{[strongest] + ties}；按原始误差、再按名称取 {strongest}")
    raw_best = min(rows, key=lambda r: (Fraction(r["bom_abs_raw"], r["beads"] or 1), r["entry"]))["entry"]
    if raw_best != strongest:
        warns.append(f"敏感性检查：若不惩罚网格错误，最强竞品会变成 {raw_best}（当前口径为 {strongest}）")
    ours_cs = [ours[i] for i in common if i in ours]
    ob = sum(r.n_beads for r in ours_cs)
    ours_row = {"entry": "我方", "n": len(ours_cs), "failures": sum(r.failed for r in ours_cs),
                "grid_ok": sum(r.grid_ok for r in ours_cs),
                "bom_error": (sum(r.bom_abs for r in ours_cs) / ob) if ob else None}
    return {"rows": ranked, "strongest": strongest, "common": common, "warnings": warns, "ours": ours_row}


def evaluate(cfg: Config, units: list[Unit], ours: list[ImageResult], tier_a: dict, tierb_entry: str,
             tierb: list[ImageResult], tierb_not_covered: list[str], allow_partial: bool,
             device_timing_ms: float | None, device_name: str | None) -> dict:
    ours_by = {r.image_id: r for r in ours}
    warns: list[str] = list(tier_a["warnings"])
    crits: list[Criterion] = []

    # ---- 第 1 条
    if tierb_entry != tier_a["strongest"]:
        raise CompetitorDataError(f"Tier B 必须是最强竞品 {tier_a['strongest']} 的结果，提供的是 {tierb_entry}")
    if tierb_not_covered:
        msg = f"最强竞品的 Tier B 缺 {len(tierb_not_covered)} 张 Hard 图纸的结果：{tierb_not_covered[:10]}"
        if not allow_partial:
            raise CompetitorDataError(msg + "（竞品处理失败要记成 error JSON；确实没测的，补测后再判定，"
                                      "或加 --allow-partial-tier-b 只在已有的图上比较）")
        warns.append(msg + "（--allow-partial-tier-b：只在已有结果的图上配对）")
    pairs = [(ours_by[c.image_id], c) for c in tierb]
    e_ours = sum(o.errors for o, _ in pairs)
    e_comp = sum(c.errors for _, c in pairs)
    beads = sum(o.n_beads for o, _ in pairs)
    d = paired_per_mille_diff([c.errors for _, c in pairs], [o.errors for o, _ in pairs], [o.n_beads for o, _ in pairs],
                              cfg.bootstrap_n, cfg.seed, cfg.confidence)  # Δ = 竞品 − 我方，> 0 表示我方更好
    c1 = Criterion(1, "和最强竞品比格子错误率", f"Hard ∩ Tier B（{len(pairs)} 张，{beads} 颗）", "", "", None, True)
    if not pairs or beads == 0:
        c1.passed = False
        c1.value = "无可比较的图纸"
        c1.notes.append("没有可配对的 Hard 图纸，无法证明优于竞品")
    else:
        ratio_ok = Fraction(e_ours) <= _q(cfg.ratio_max) * e_comp
        sig_ok = d["ci_low"] is not None and d["ci_low"] > 0
        c1.passed = bool(ratio_ok and sig_ok)
        ratio_txt = f"{e_ours / e_comp:.3f}" if e_comp else ("∞" if e_ours else "—")
        c1.value = (f"我方 {d['pm_b']:.3f}‰ vs 竞品 {d['pm_a']:.3f}‰（比值 {ratio_txt}）；"
                    f"Δ（竞品−我方）= {d['diff']:.3f}‰，95% CI [{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
        c1.threshold = f"我方 ≤ 竞品 × {cfg.ratio_max:g}，且 CI 下界 > 0"
        c1.notes.append(f"比值条件：{'满足' if ratio_ok else '不满足'}（错误格 我方 {e_ours} / 竞品 {e_comp}）；"
                        f"显著性条件：{'满足' if sig_ok else '不满足'}")
        if ratio_ok and not sig_ok:
            c1.notes.append(f"点估计达到了比值要求，但配对置信区间跨过 0（不显著）。按 04 §7 第 5 条：把 test 集里的 Hard 图纸"
                            f"扩充到 ≥ {cfg.hard_images_target} 张后再判定，**而不是降低门槛**（当前 {len(pairs)} 张）")
    crits.append(c1)

    # ---- 第 2 条
    clean = [r for r in ours if r.in_main and not r.hard]
    ce, cb = sum(r.errors for r in clean), sum(r.n_beads for r in clean)
    c2 = Criterion(2, "Clean 集每千颗错误数（微平均）", f"Clean（{len(clean)} 张，{cb} 颗）", "", f"≤ {cfg.clean_per_mille_max:g}",
                   None, True)
    if cb == 0:
        c2.passed, c2.value = False, "Clean 集为空"
        c2.notes.append("Clean 集没有图纸，无法证明达标")
    else:
        lo, hi = ratio_ci([r.errors for r in clean], [r.n_beads for r in clean], cfg.bootstrap_n, cfg.seed, cfg.confidence, 1000.0)
        c2.passed = Fraction(1000 * ce, cb) <= _q(cfg.clean_per_mille_max)
        c2.value = f"{1000 * ce / cb:.3f}‰（{ce}/{cb}），95% CI [{lo:.3f}, {hi:.3f}]"
    crits.append(c2)

    # ---- 第 3 条（只报告）
    fl = [r for r in ours if r.in_flag_scope]
    hits, errs = sum(r.flag_hits for r in fl), sum(r.errors for r in fl)
    ok_fl = [r for r in fl if r.status == "ok"]
    kept, raw, nb = sum(r.flag_kept for r in ok_fl), sum(r.flag_raw for r in ok_fl), sum(r.n_beads for r in ok_fl)
    trunc = sum(r.flag_raw > r.flag_cap for r in ok_fl)
    rec = Fraction(hits, errs) if errs else Fraction(1)
    c3 = Criterion(3, "不确定标记召回（上限内）", f"全部 test 图纸（不含 T7，{len(fl)} 张）",
                   f"{float(rec) * 100:.2f}%（{hits}/{errs}）" + ("，无错误格" if not errs else ""),
                   f"≥ {cfg.flag_recall_min * 100:g}%（上限：{cfg.cap_rule_text()}）",
                   rec >= _q(cfg.flag_recall_min), False)
    c3.notes.append(f"截断后标记率 {(kept / nb * 100) if nb else 0:.3f}%，引擎原始标记率 {(raw / nb * 100) if nb else 0:.3f}%；"
                    f"超上限被截断的图 {trunc} 张")
    crits.append(c3)

    # ---- 第 4 条
    main = [r for r in ours if r.in_main]
    g = sum(r.grid_ok for r in main)
    c4 = Criterion(4, "网格尺寸正确率", f"主统计池（不含 T6、T7，{len(main)} 张）", "", f"≥ {cfg.grid_accuracy_min * 100:g}%",
                   None, True)
    if not main:
        c4.passed, c4.value = False, "主统计池为空"
    else:
        c4.passed = Fraction(g, len(main)) >= _q(cfg.grid_accuracy_min)
        c4.value = f"{g / len(main) * 100:.2f}%（{g}/{len(main)}）"
    crits.append(c4)

    # ---- 第 5 条（只报告）
    if device_timing_ms is None:
        c5 = Criterion(5, "端侧耗时（104×104，中端机）", "T0.14 实测", "未提供（--device-timing-ms）",
                       f"≤ {cfg.timing_ms_max:g} ms", None, False)
    else:
        c5 = Criterion(5, "端侧耗时（104×104，中端机）", device_name or "T0.14 实测", f"{device_timing_ms:g} ms",
                       f"≤ {cfg.timing_ms_max:g} ms", device_timing_ms <= cfg.timing_ms_max, False)
    crits.append(c5)

    # ---- 第 6 条（只报告）
    lg = [r for r in fl if r.has_legend]
    lok = sum(bool(r.legend_ok) for r in lg)
    c6 = Criterion(6, "图例解析完全正确率", f"带图例的 test 图纸（不含 T7，{len(lg)} 张）",
                   f"{lok / len(lg) * 100:.2f}%（{lok}/{len(lg)}）" if lg else "没有带图例的图纸",
                   f"≥ {cfg.legend_accuracy_min * 100:g}%",
                   (Fraction(lok, len(lg)) >= _q(cfg.legend_accuracy_min)) if lg else None, False)
    crits.append(c6)

    go = all(c.passed is True for c in crits if c.required)
    return {
        "verdict": "GO" if go else "NO-GO",
        "criteria": crits,
        "paired": d,
        "pairs": pairs,
        "warnings": warns,
    }


def _ptxt(c: Criterion) -> str:
    if c.passed is None:
        return "—"
    return "达标" if c.passed else "**未达标**"


def render(meta: dict, tier_a: dict, res: dict, tierb_entry: str, cfg: Config) -> str:
    L = [f"# go / no-go 报告（04 §0）", "",
         f"## 结论：**{res['verdict']}**", "",
         "判定规则：第 1、2、4 条同时达标为 GO；第 3、5、6 条是 MVP 上线门槛，只报告、不影响结论。", "",
         f"- 我方：run = `{meta['run']}`，test_reason = `{meta['test_reason']}`，git `{meta['git_rev']}`，"
         f"results.json 已核对 test_usage.jsonl 的登记",
         f"- 最强竞品（Tier A 用量清单误差最小）：`{tier_a['strongest']}`；Tier B 数据：`{tierb_entry}`",
         f"- 评测器参数指纹 `{meta['config_sha256'][:12]}`；θ = {cfg.theta:g}（已冻结）；bootstrap {cfg.bootstrap_n} 次，种子 {cfg.seed}",
         ""]
    if res["warnings"]:
        L.append("## 警告")
        L.append("")
        L += [f"- {w}" for w in res["warnings"]]
        L.append("")
    L.append("## 各条指标")
    L.append("")
    L.append(md_table(["#", "指标", "口径", "我方结果", "门槛", "结论", "性质"],
                      [[str(c.no), c.title, c.scope, c.value, c.threshold, _ptxt(c), "go 必要条件" if c.required else "MVP 上线门槛（只报告）"]
                       for c in res["criteria"]]))
    L.append("")
    for c in res["criteria"]:
        for n in c.notes:
            L.append(f"- 第 {c.no} 条：{n}")
    L.append("")
    L.append("## Tier A：最强竞品的确定")
    L.append("")
    L.append(f"在主统计池里所有竞品条目都覆盖到的 {len(tier_a['common'])} 张图上比较。用量清单误差 = Σ|识别颗数 − 真值颗数| / 总豆数"
             "（微平均）；网格尺寸错或处理失败的图按「没有输出」计（误差 = 该图豆数）。")
    L.append("")
    rows = []
    for r in tier_a["rows"]:
        rows.append([r["entry"] + (" ← 最强" if r["entry"] == tier_a["strongest"] else ""), r["app_version"], r["tested_at"],
                     f"{r['n']}（共覆盖 {r['covered']}）", str(r["failures"]), f"{r['grid_ok']}/{r['n']}",
                     "—" if r["bom_error"] is None else f"{r['bom_error'] * 100:.3f}%",
                     "—" if r["bom_error_raw"] is None else f"{r['bom_error_raw'] * 100:.3f}%"])
    o = tier_a["ours"]
    rows.append(["我方（参考）", "", "", str(o["n"]), str(o["failures"]), f"{o['grid_ok']}/{o['n']}",
                 "—" if o["bom_error"] is None else f"{o['bom_error'] * 100:.3f}%", ""])
    L.append(md_table(["竞品条目", "版本", "测试日期", "图数", "失败", "网格正确", "用量清单误差", "不惩罚网格错误时"], rows))
    L.append("")
    L.append("## Tier B：Hard 集逐图配对（错误格数）")
    L.append("")
    d = res["paired"]
    if d["diff"] is not None:
        L.append(f"Δ（竞品 − 我方）= {d['diff']:.3f}‰，95% CI [{d['ci_low']:.3f}, {d['ci_high']:.3f}]；"
                 f"重采样中我方更差的比例 {d['p_a_better']:.4f}（Δ < 0 的比例）。")
        L.append("")
    rows = []
    for o, c in sorted(res["pairs"], key=lambda p: (p[1].errors - p[0].errors, p[0].image_id)):
        rows.append([o.image_id, o.type, o.size_bucket, ",".join(o.hard_reasons), str(o.n_beads),
                     f"{o.errors}（{o.status}）", f"{c.errors}（{c.status}）", str(c.errors - o.errors)])
    L.append(md_table(["image_id", "类型", "尺寸", "Hard 原因", "豆数", "我方错误格（状态）", "竞品错误格（状态）", "竞品 − 我方"], rows))
    L.append("")
    fails = [o for o, _ in res["pairs"] if o.failed]
    L.append(f"## 我方失败案例（Hard ∩ Tier B 中 {len(fails)} 张；全部 test 集的失败列表见 run 报告）")
    L.append("")
    if fails:
        L.append(md_table(["image_id", "失败类型", "说明"], [[r.image_id, r.status, r.detail] for r in fails]))
    else:
        L.append("无。")
    L.append("")
    return "\n".join(L)
