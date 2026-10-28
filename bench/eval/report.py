"""报告输出：Markdown + CSV（逐图明细、汇总长表）+ results.json（供 compare / gonogo 复用）。"""
from __future__ import annotations

import csv
import io
import statistics
from pathlib import Path

from bench.eval.aggregate import METRIC_BY_KEY, METRICS, SCOPE_TITLES, compute, groups
from bench.eval.config import Config
from bench.eval.scoring import ImageResult

PER_IMAGE_COLUMNS = [
    "image_id", "type", "size_bucket", "tags", "in_main", "in_flag_scope", "hard", "hard_reasons", "clean",
    "blurry", "min_cell_px", "lap_var", "gt_cols", "gt_rows", "n_beads", "n_empty", "status", "error_code",
    "pred_cols", "pred_rows", "errors", "per_mille", "wrong_code", "bead_to_empty", "empty_to_bead",
    "correct_beads", "nonempty_acc", "zero_error", "bom_abs", "bom_abs_raw", "bom_error", "has_legend",
    "legend_ok", "flag_source", "flag_raw", "flag_cap", "flag_kept", "flag_hits", "timing_ms", "detail", "warnings",
]
SUMMARY_COLUMNS = ["group_kind", "group", "metric", "value", "ci_low", "ci_high", "num", "den", "n_images"]

STATUS_TITLES = {"ok": "成功", "grid_wrong": "网格尺寸错", "error": "处理失败（error）",
                 "missing": "缺预测结果（崩溃）", "invalid": "输出不合法（崩溃）"}


# --------------------------------------------------------------------------- 格式化

def fmt(key: str, v: float | None) -> str:
    if v is None:
        return "—"
    kind = METRIC_BY_KEY[key].kind if key in METRIC_BY_KEY else "ratio"
    if kind == "per_mille":
        return f"{v:.3f}"
    if kind == "rate":
        return f"{v * 100:.2f}%"
    return f"{v:.4g}"


def fmt_ci(key: str, lo: float | None, hi: float | None) -> str:
    if lo is None or hi is None:
        return "—"
    return f"[{fmt(key, lo)}, {fmt(key, hi)}]"


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    return str(v)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in r) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- CSV

def per_image_rows(results: list[ImageResult]) -> list[dict]:
    rows = []
    for r in results:
        d = r.to_dict()
        d["clean"] = r.clean
        d["per_mille"] = 1000.0 * r.errors / r.n_beads if r.n_beads else None
        d["nonempty_acc"] = r.nonempty_acc
        d["zero_error"] = r.zero_error
        d["bom_error"] = r.bom_abs / r.n_beads if r.n_beads else None
        rows.append({c: d.get(c) for c in PER_IMAGE_COLUMNS})
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8-sig")  # 带 BOM，Excel 直接打开不乱码


# --------------------------------------------------------------------------- Markdown

_STRATA_KEYS = ["grid_accuracy", "per_mille", "nonempty_acc_macro", "zero_error_rate", "bom_error",
                "flag_recall", "legend_accuracy"]


def _strata_table(rs_groups: list[tuple[str, list[ImageResult]]], cfg: Config, title_of=lambda n: n) -> str:
    header = ["分组", "图数", "失败", "网格正确率", "每千颗错误数", "95% CI", "非空格准确率（宏）", "零错误率",
              "用量清单误差", "召回（上限内）", "图例完全正确"]
    rows = []
    for name, rs in rs_groups:
        if not rs:
            rows.append([title_of(name), "0"] + ["—"] * (len(header) - 2))
            continue
        ms = compute(rs, cfg, keys=_STRATA_KEYS)
        pm = ms["per_mille"]
        rows.append([title_of(name), str(len(rs)), str(sum(r.failed for r in rs)),
                     fmt("grid_accuracy", ms["grid_accuracy"].value), fmt("per_mille", pm.value),
                     fmt_ci("per_mille", pm.ci_low, pm.ci_high),
                     fmt("nonempty_acc_macro", ms["nonempty_acc_macro"].value),
                     fmt("zero_error_rate", ms["zero_error_rate"].value), fmt("bom_error", ms["bom_error"].value),
                     fmt("flag_recall", ms["flag_recall"].value), fmt("legend_accuracy", ms["legend_accuracy"].value)])
    return md_table(header, rows)


def threshold_rows(results: list[ImageResult], cfg: Config) -> list[dict]:
    """与 04 §0.1 门槛对照（只是参考；正式判定用 gonogo）。"""
    main = [r for r in results if r.in_main]
    clean = [r for r in main if not r.hard]
    fl = [r for r in results if r.in_flag_scope]
    m_clean = compute(clean, cfg, keys=["per_mille"])["per_mille"]
    m_grid = compute(main, cfg, keys=["grid_accuracy"])["grid_accuracy"]
    m_flag = compute(fl, cfg, keys=["flag_recall", "flag_rate"])
    m_leg = compute(fl, cfg, keys=["legend_accuracy"])["legend_accuracy"]

    def ok(v, thr, higher):
        if v is None:
            return None
        return v >= thr if higher else v <= thr

    rec = m_flag["flag_recall"].value
    rec_eff = 1.0 if rec is None and m_flag["flag_recall"].den == 0 else rec
    return [
        {"no": 2, "title": "Clean 集每千颗错误数", "scope": "Clean", "key": "per_mille", "metric": m_clean,
         "threshold": f"≤ {cfg.clean_per_mille_max:g}", "pass": ok(m_clean.value, cfg.clean_per_mille_max, False),
         "nature": "go 必要条件"},
        {"no": 3, "title": "不确定标记召回（上限内）", "scope": "不含 T7", "key": "flag_recall", "metric": m_flag["flag_recall"],
         "threshold": f"≥ {cfg.flag_recall_min * 100:g}%", "pass": ok(rec_eff, cfg.flag_recall_min, True),
         "nature": "MVP 上线门槛"},
        {"no": 4, "title": "网格尺寸正确率", "scope": "主统计池", "key": "grid_accuracy", "metric": m_grid,
         "threshold": f"≥ {cfg.grid_accuracy_min * 100:g}%", "pass": ok(m_grid.value, cfg.grid_accuracy_min, True),
         "nature": "go 必要条件"},
        {"no": 6, "title": "图例解析完全正确率", "scope": "带图例、不含 T7", "key": "legend_accuracy", "metric": m_leg,
         "threshold": f"≥ {cfg.legend_accuracy_min * 100:g}%", "pass": ok(m_leg.value, cfg.legend_accuracy_min, True),
         "nature": "MVP 上线门槛"},
    ]


def _pass_text(p) -> str:
    return "—" if p is None else ("达标" if p else "未达标")


def render_run_report(meta: dict, results: list[ImageResult], cfg: Config, warnings: list[str]) -> str:
    main = [r for r in results if r.in_main]
    hard = [r for r in main if r.hard]
    clean = [r for r in main if not r.hard]
    L: list[str] = []
    L.append(f"# 评测报告：{meta['run']}（{meta['split']} 集）")
    L.append("")
    L.append(f"- 生成时间：{meta['created_at']}；git：`{meta['git_rev']}`；评测器参数指纹：`{meta['config_sha256'][:12]}`")
    L.append(f"- 预测结果：`{meta['preds_dir']}`（指纹 `{meta['preds_sha256'][:12]}`）；真值：`{meta['labels_dir']}`")
    if meta.get("test_reason"):
        L.append(f"- **test 集使用记录**：reason = `{meta['test_reason']}`（已追加到 test_usage.jsonl）")
    L.append(f"- θ（拉普拉斯方差阈值）= {cfg.theta:g}（{'已冻结' if cfg.theta_frozen else '**占位值，未冻结**'}）；"
             f"最小格边长阈值 {cfg.min_cell_px:g} px；τ = {cfg.tau:g}")
    L.append(f"- 不确定标记上限：{cfg.cap_rule_text()}")
    L.append(f"- bootstrap：{cfg.bootstrap_n} 次，按图纸重采样，种子 {cfg.seed}，{cfg.confidence * 100:g}% 百分位区间")
    n_t6 = sum(r.type == "T6" for r in results)
    n_t7 = sum(r.type == "T7" for r in results)
    L.append(f"- 图纸数：共 {len(results)}；主统计池（不含 T6、T7）{len(main)}，其中 Hard {len(hard)}、Clean {len(clean)}；"
             f"T6 {n_t6}、T7 {n_t7}（单独报告）")
    L.append("- 指标定义见 `bench/eval/README.md`。失败的图纸（网格尺寸错、处理失败、缺预测结果）：所有非空真值格按错误计，零错误记「否」。")
    L.append("")
    if warnings:
        L.append("## 警告")
        L.append("")
        for w in warnings:
            L.append(f"- {w}")
        L.append("")

    # ---- 1. 主指标
    L.append("## 1. 主指标（主统计池：不含 T6、T7）")
    L.append("")
    ms = compute(main, cfg)
    rows = []
    for m in METRICS:
        if m.key in ("flag_recall", "flag_recall_ok_only", "flag_precision", "flag_rate", "flag_rate_raw", "legend_accuracy"):
            continue
        x = ms[m.key]
        rows.append([m.title, fmt(m.key, x.value), fmt_ci(m.key, x.ci_low, x.ci_high), f"{x.num:g} / {x.den:g}"])
    L.append(md_table(["指标", "值", "95% CI", "分子 / 分母"], rows))
    L.append("")
    e2b = sum(r.empty_to_bead or 0 for r in main if r.status == "ok")
    b2e = sum(r.bead_to_empty or 0 for r in main if r.status == "ok")
    L.append(f"空格误判（只统计成功的图）：空格→豆子 **{e2b}** 次，豆子→空格 **{b2e}** 次。")
    L.append("")

    # ---- 2. 门槛对照
    L.append("## 2. 与 go/no-go 门槛对照（只作参考；正式判定用 `gonogo` 子命令，第 1 条需要竞品数据）")
    L.append("")
    rows = []
    for t in threshold_rows(results, cfg):
        x = t["metric"]
        rows.append([str(t["no"]), t["title"], t["scope"], fmt(t["key"], x.value), fmt_ci(t["key"], x.ci_low, x.ci_high),
                     t["threshold"], _pass_text(t["pass"]), t["nature"]])
    L.append(md_table(["#", "指标", "口径", "值", "95% CI", "门槛", "结论", "性质"], rows))
    L.append("")

    # ---- 3. Hard / Clean
    L.append("## 3. Hard / Clean（只含主统计池）")
    L.append("")
    L.append("Hard = 尺寸档 L/XL，或带 watermark / blurry 标签，或类型 T4；Clean = 主统计池里其余图纸。")
    L.append("")
    L.append(_strata_table([("hard", hard), ("clean", clean), ("main", main)], cfg, lambda n: SCOPE_TITLES[n]))
    L.append("")

    # ---- 4. 分层
    gs = groups(results)
    L.append("## 4. 分层")
    L.append("")
    L.append("### 4.1 尺寸档（主统计池）")
    L.append("")
    L.append(_strata_table([(n, rs) for k, n, rs in gs if k == "size"], cfg))
    L.append("")
    L.append("### 4.2 类型（全部图纸；T6、T7 不计入第 1、2、4 条，在这里单独报告）")
    L.append("")
    L.append(_strata_table([(n, rs) for k, n, rs in gs if k == "type" and rs], cfg,
                           lambda n: n + ("（单独报告）" if n in cfg.main_excluded_types else "")))
    t7 = [r for r in results if r.type == "T7"]
    if t7:
        n_crop = sum(r.status == "error" and r.error_code == "needs_user_crop" for r in t7)
        L.append("")
        L.append(f"T7 中返回 `needs_user_crop` 的：{n_crop} / {len(t7)}。")
    L.append("")
    L.append("### 4.3 难点标签（主统计池；一张图可以有多个标签；blurry 按规则判定）")
    L.append("")
    L.append(_strata_table([(n, rs) for k, n, rs in gs if k == "tag" and rs], cfg))
    L.append("")

    # ---- 5. 不确定标记
    fl = [r for r in results if r.in_flag_scope]
    L.append("## 5. 不确定标记（口径：不含 T7）")
    L.append("")
    L.append(f"截断规则：{cfg.cap_rule_text()}。")
    L.append("召回 = 保留的标记中落在错误格上的个数 / 全部错误格（失败图的错误格 = 全部非空真值格，且没有可用的标记）；"
             "精确率、标记率只在成功的图上统计。")
    L.append("")
    fm = compute(fl, cfg, keys=["flag_recall", "flag_recall_ok_only", "flag_precision", "flag_rate", "flag_rate_raw"])
    rows = [[METRIC_BY_KEY[k].title, fmt(k, fm[k].value), fmt_ci(k, fm[k].ci_low, fm[k].ci_high), f"{fm[k].num:g} / {fm[k].den:g}"]
            for k in ("flag_recall", "flag_recall_ok_only", "flag_precision", "flag_rate", "flag_rate_raw")]
    L.append(md_table(["指标", "值", "95% CI", "分子 / 分母"], rows))
    trunc = [r for r in fl if r.status == "ok" and r.flag_raw > r.flag_cap]
    src = sorted({r.flag_source for r in fl if r.status == "ok"})
    L.append("")
    L.append(f"超出上限被截断的图：{len(trunc)} 张{('：' + ', '.join(r.image_id for r in trunc[:20])) if trunc else ''}。"
             f"标记来源：{', '.join(src) if src else '—'}（uncertain = 预测结果的 uncertain 字段；tau = 由 confidence < τ 推出；none = 没有标记）。")
    L.append("")

    # ---- 6. 失败
    failed = [r for r in results if r.failed]
    L.append(f"## 6. 失败的图纸（{len(failed)} 张）")
    L.append("")
    if failed:
        rows = [[r.image_id, r.type, r.size_bucket, STATUS_TITLES.get(r.status, r.status), r.error_code or "", r.detail,
                 str(r.n_beads)] for r in failed]
        L.append(md_table(["image_id", "类型", "尺寸", "失败类型", "错误码", "说明", "计为错误的格数"], rows))
    else:
        L.append("无。")
    L.append("")

    # ---- 7. 错误最多的图
    worst = sorted([r for r in results if r.n_beads > 0], key=lambda r: (-r.errors / r.n_beads, r.image_id))[:15]
    L.append("## 7. 每千颗错误数最高的图纸（前 15）")
    L.append("")
    rows = [[r.image_id, r.type, r.size_bucket, "H" if r.hard else ("C" if r.in_main else "—"), r.status,
             str(r.errors), str(r.n_beads), f"{1000 * r.errors / r.n_beads:.1f}",
             "" if r.wrong_code is None else f"{r.wrong_code}/{r.bead_to_empty}/{r.empty_to_bead}",
             f"{r.flag_hits}/{r.flag_kept}"] for r in worst]
    L.append(md_table(["image_id", "类型", "尺寸", "H/C", "状态", "错误格", "豆数", "‰", "色号错/豆→空/空→豆", "标记命中/保留"], rows))
    L.append("")

    # ---- 8. 耗时
    L.append("## 8. 耗时（只报告，不设门禁；端侧实测见 T0.14）")
    L.append("")
    tm = [r.timing_ms for r in results if r.timing_ms is not None and r.status == "ok"]
    if tm:
        tm_sorted = sorted(tm)
        p90 = tm_sorted[min(len(tm_sorted) - 1, int(0.9 * len(tm_sorted)))]
        L.append(f"成功图纸 {len(tm)} 张（预测结果 timing_ms 各阶段之和）：中位数 {statistics.median(tm):.0f} ms，"
                 f"P90 {p90:.0f} ms，最大 {max(tm):.0f} ms。")
        rows = []
        for b in ("S", "M", "L", "XL"):
            v = [r.timing_ms for r in results if r.size_bucket == b and r.timing_ms is not None and r.status == "ok"]
            if v:
                rows.append([b, str(len(v)), f"{statistics.median(v):.0f}", f"{max(v):.0f}"])
        if rows:
            L.append("")
            L.append(md_table(["尺寸档", "图数", "中位数 ms", "最大 ms"], rows))
    else:
        L.append("预测结果里没有 timing_ms。")
    L.append("")

    # ---- 9. 输出文件（附：人工 blurry 标签与规则不一致的图数；只有不一致时 make_unit 才会写这条警告）
    dis = [r for r in results if any("人工 blurry 标签" in w for w in r.warnings)]
    L.append("## 9. 输出文件")
    L.append("")
    L.append("- `per_image.csv`：逐图明细（列定义见 bench/eval/README.md）")
    L.append("- `summary.csv`：各口径、各分层 × 各指标的长表，含置信区间、分子、分母")
    L.append("- `results.json`：逐图结果和运行元数据（compare / gonogo 复用）")
    if dis:
        L.append("")
        L.append(f"注：{len(dis)} 张图的人工 blurry 标签与规则判定不一致，见逐图 warnings 列。")
    L.append("")
    return "\n".join(L)
