"""命令行入口：python -m bench.eval <run|compare|gate|gonogo|calibrate-theta> ...

退出码：0 成功 / GO / 门禁通过；1 NO-GO / 门禁退化 / 无法标定；2 参数或数据错误（含 test 集纪律拒绝）；
3 门禁基线缺失或已失效（数据集或参数变了，需要 gate --update）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from bench.common import formats as F
from bench.eval import calibrate, compare, gate, gonogo
from bench.eval.competitors import CompetitorDataError, parse_tier_a, score_tier_b
from bench.eval.config import ConfigError, load_config
from bench.eval.dataset import DatasetError, load_units
from bench.eval.discipline import (REASONS, TestDisciplineError, append_usage, check_label_hashes,
                                   check_reason_available, check_test_lock, git_rev, now_iso)
from bench.eval.report import PER_IMAGE_COLUMNS, SUMMARY_COLUMNS, per_image_rows, render_run_report, write_csv
from bench.eval.aggregate import summarize
from bench.eval.runner import (UsageError, build_results_json, default_report_dir, evaluate, load_results,
                               resolve_paths, resolve_preds_dir, resolve_results_file, units_fingerprint,
                               verify_test_results)
from bench.eval.scoring import score_units


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--labels", help="真值目录（默认 $PINDOU_DATA_DIR/labels）")
    p.add_argument("--images", help="原图目录（默认 $PINDOU_DATA_DIR/images；blurry 判定要用）")
    p.add_argument("--manifest", help="manifest.jsonl（默认 bench/datasets/manifest.jsonl）")
    p.add_argument("--config", help="评测参数（默认 bench/eval/config.yaml）")


def _test_preflight(args, cfg, paths) -> dict:
    """test 集的前置检查（不写任何东西）：reason 未用过、θ 已冻结、test.lock 与图片一致。"""
    check_reason_available(args.test_reason)
    if not cfg.theta_frozen:
        raise TestDisciplineError("config.yaml 的 θ 还是占位值（theta_frozen: false）。Hard/Clean 的定义必须先在 dev 集上标定并冻结，"
                                  "才能使用 test 集")
    return check_test_lock(F.read_split("test"), paths.images, paths.manifest)


# --------------------------------------------------------------------------- run

def cmd_run(args) -> int:
    cfg = load_config(args.config)
    paths = resolve_paths(args.labels, args.images, args.manifest)
    preds = resolve_preds_dir(args.preds)
    name = args.name or preds.name
    out = Path(args.out) if args.out else default_report_dir(name, args.split)
    lock = None
    if args.split == "test":
        if args.allow_missing_labels:
            raise UsageError("test 集不允许 --allow-missing-labels")
        lock = _test_preflight(args, cfg, paths)
        if (out / "results.json").exists():
            raise UsageError(f"{out} 里已有 test 集的结果，拒绝覆盖（换一个 --name 或 --out）")
    elif args.test_reason:
        raise UsageError("--test-reason 只用于 --split test")

    units, results, warns, preds_sha = evaluate(args.split, preds, cfg, paths, args.allow_missing_labels)
    if lock is not None:
        check_label_hashes(units, lock)
    meta, text = build_results_json(run=name, split=args.split, preds_dir=preds, preds_sha=preds_sha,
                                    labels_dir=paths.labels, cfg=cfg, units=units, results=results, warnings=warns,
                                    test_reason=args.test_reason if args.split == "test" else None)
    if args.split == "test":
        # 先完成全部计算，再登记使用记录，最后才写出 / 显示结果：计算出错不会白白消耗 reason，结果也不会在登记前被看到
        append_usage(args.test_reason, name, {
            "split": "test", "tool": "bench.eval run", "preds_sha256": preds_sha,
            "results_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "config_sha256": cfg.fingerprint(), "n_units": len(units),
        })
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_bytes(text.encode("utf-8"))
    write_csv(out / "per_image.csv", PER_IMAGE_COLUMNS, per_image_rows(results))
    write_csv(out / "summary.csv", SUMMARY_COLUMNS, summarize(results, cfg))
    md = render_run_report(meta, results, cfg, warns)
    (out / "report.md").write_text(md, encoding="utf-8")
    main = [r for r in results if r.in_main]
    e, b = sum(r.errors for r in main), sum(r.n_beads for r in main)
    print(f"[{name} / {args.split}] 图纸 {len(results)}（主统计池 {len(main)}），失败 {sum(r.failed for r in results)}，"
          f"每千颗错误数 {1000 * e / b if b else float('nan'):.3f}，警告 {len(warns)} 条")
    print(f"报告：{out / 'report.md'}")
    return 0


# --------------------------------------------------------------------------- compare

def _load_side(arg: str, split: str, cfg, paths, units):
    if split == "dev":
        preds = resolve_preds_dir(arg)
        results, warns, _ = score_units(units, preds, cfg, paths.manifest)
        return preds.name, results, warns
    f = resolve_results_file(arg, "test")
    meta, results, sha = load_results(f)
    verify_test_results(meta, sha)
    if meta["config_sha256"] != cfg.fingerprint():
        raise UsageError(f"{meta['run']} 的评测参数指纹与当前 config 不同，不能配对比较")
    return meta["run"], results, []


def cmd_compare(args) -> int:
    cfg = load_config(args.config)
    units = None
    paths = None
    if args.split == "dev":
        paths = resolve_paths(args.labels, args.images, args.manifest)
        units, _ = load_units("dev", cfg, paths.labels, paths.images, paths.manifest, args.allow_missing_labels)
    name_a, ra, wa = _load_side(args.a, args.split, cfg, paths, units)
    name_b, rb, wb = _load_side(args.b, args.split, cfg, paths, units)
    pairs, warns = compare.pair_results(ra, rb)
    warns = [f"A：{w}" for w in wa] + [f"B：{w}" for w in wb] + warns
    rows = compare.paired_groups(pairs, cfg)
    md = compare.render(name_a, name_b, args.split, rows, warns, cfg)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- gate

def cmd_gate(args) -> int:
    cfg = load_config(args.config)
    paths = resolve_paths(args.labels, args.images, args.manifest)
    preds = resolve_preds_dir(args.preds)
    units, results, warns, preds_sha = evaluate("dev", preds, cfg, paths)
    cur = gate.gate_metrics(results)
    fp = units_fingerprint(units, cfg)
    base_path = Path(args.baseline) if args.baseline else cfg.gate_baseline
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else None
    for w in warns:
        print(f"警告：{w}")
    if args.update:
        if base is not None and base.get("units_sha256") == fp:
            rows = gate.compare_to_baseline(base["metrics"], cur)
            print(gate.render(rows, base.get("timing_ms_median"), gate.timing_median(results)))
            if any(r["regressed"] for r in rows) and not args.allow_regression:
                print("拒绝更新：当前结果相对基线有退化。确实要接受退化，加 --allow-regression。")
                return 1
        new = {"created_at": now_iso(), "git_rev": git_rev(), "preds_dir": str(preds), "preds_sha256": preds_sha,
               "units_sha256": fp, "config_sha256": cfg.fingerprint(), "n_units": len(units), "metrics": cur,
               "timing_ms_median": gate.timing_median(results)}
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_path.write_text(json.dumps(new, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"已更新基线：{base_path}")
        return 0
    if base is None:
        print(f"没有门禁基线 {base_path}：先在主干版本上运行 `gate --preds ... --update` 建立基线。")
        return 3
    if base.get("units_sha256") != fp:
        print("门禁基线已失效：dev 集的图纸、真值、Hard/Clean 划分或评测参数和建立基线时不同。"
              "请用主干版本的预测结果重新运行 `gate --update` 建立基线。")
        return 3
    rows = gate.compare_to_baseline(base["metrics"], cur)
    print(gate.render(rows, base.get("timing_ms_median"), gate.timing_median(results)))
    bad = [r for r in rows if r["regressed"]]
    if bad:
        print(f"门禁未通过：{', '.join(r['title'] for r in bad)} 变差了")
        return 1
    print("门禁通过")
    return 0


# --------------------------------------------------------------------------- gonogo

def cmd_gonogo(args) -> int:
    cfg = load_config(args.config)
    if not cfg.theta_frozen:
        raise TestDisciplineError("θ 尚未冻结，不能做 go/no-go 判定")
    paths = resolve_paths(args.labels, args.images, args.manifest)
    f = resolve_results_file(args.run, "test")
    meta, ours, sha = load_results(f)
    verify_test_results(meta, sha, reason="final_w6")
    if meta["config_sha256"] != cfg.fingerprint():
        raise UsageError("我方 final_w6 结果的评测参数指纹与当前 config 不同（参数在冻结后被改过？）")
    lock = check_test_lock(F.read_split("test"), paths.images, paths.manifest)
    units, _ = load_units("test", cfg, paths.labels, paths.images, paths.manifest)
    check_label_hashes(units, lock)
    if meta["units_sha256"] != units_fingerprint(units, cfg):
        raise UsageError("test 集的图纸 / 真值 / Hard 划分和我方 final_w6 运行时不同，不能判定")
    if args.tier_a:
        ta_files = [Path(p) for p in args.tier_a]
    else:
        data = F.data_dir(required=False)
        ta_files = sorted((data / "competitors").glob("*/tier_a*.csv")) if data else []
    entries = parse_tier_a(ta_files)
    ours_by = {r.image_id: r for r in ours}
    tier_a = gonogo.tier_a_table(entries, units, ours_by)
    hard_units = [u for u in units if u.in_main and u.hard]
    tierb, not_cov = score_tier_b(hard_units, Path(args.tier_b), cfg, paths.manifest)
    res = gonogo.evaluate(cfg, units, ours, tier_a, args.tier_b_entry, tierb, not_cov, args.allow_partial_tier_b,
                          args.device_timing_ms, args.device_name)
    md = gonogo.render(meta, tier_a, res, args.tier_b_entry, cfg)
    out = Path(args.out) if args.out else f.parent / "gonogo"
    out.mkdir(parents=True, exist_ok=True)
    (out / "gonogo.md").write_text(md, encoding="utf-8")
    doc = {"verdict": res["verdict"], "created_at": now_iso(), "git_rev": git_rev(), "run": meta["run"],
           "strongest": tier_a["strongest"], "tier_b_entry": args.tier_b_entry,
           "criteria": [c.__dict__ for c in res["criteria"]], "paired": res["paired"],
           "tier_a": tier_a["rows"], "tier_a_common": tier_a["common"], "warnings": res["warnings"],
           "pairs": [{"image_id": o.image_id, "ours_errors": o.errors, "ours_status": o.status, "comp_errors": c.errors,
                      "comp_status": c.status, "n_beads": o.n_beads} for o, c in res["pairs"]]}
    (out / "gonogo.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(md)
    print(f"结论：{res['verdict']}（报告：{out / 'gonogo.md'}）")
    return 0 if res["verdict"] == "GO" else 1


# --------------------------------------------------------------------------- calibrate-theta

def cmd_calibrate(args) -> int:
    cfg = load_config(args.config)
    paths = resolve_paths(args.labels, args.images, args.manifest)
    units, _ = load_units("dev", cfg, paths.labels, paths.images, paths.manifest, args.allow_missing_labels)
    res = calibrate.suggest_theta(units, cfg)
    md = calibrate.render(res, cfg)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md, encoding="utf-8")
    return 0 if res["theta"] is not None else 1


# --------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m bench.eval", description="识别 benchmark 评测器（T0.7）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="评测一个预测结果目录，输出 Markdown 报告和 CSV")
    p.add_argument("--preds", required=True, help="预测结果目录，或 $PINDOU_DATA_DIR/preds/ 下的 run 名")
    p.add_argument("--split", required=True, choices=["dev", "test", "synth"],
                   help="synth：评测 --manifest 指定的合成集全部样本（配合 --labels/--images 指向合成目录）")
    p.add_argument("--test-reason", choices=list(REASONS), help="--split test 时必填，每个 reason 只能用一次")
    p.add_argument("--out", help="报告目录（默认 $PINDOU_DATA_DIR/reports/<run>/<split>/）")
    p.add_argument("--name", help="run 名（默认取预测结果目录名）")
    p.add_argument("--allow-missing-labels", action="store_true", help="只用于 dev：跳过还没标注的图")
    _common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("compare", help="两次运行在同一批图上的配对 bootstrap 比较")
    p.add_argument("--a", required=True, help="dev：预测结果目录或 run 名；test：run 名或报告目录（读 results.json）")
    p.add_argument("--b", required=True)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--out", help="把 Markdown 结果另存到这个文件")
    p.add_argument("--allow-missing-labels", action="store_true")
    _common(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("gate", help="dev 集质量门禁（CI 用）")
    p.add_argument("--preds", required=True)
    p.add_argument("--update", action="store_true", help="用这次结果更新基线")
    p.add_argument("--allow-regression", action="store_true", help="和 --update 一起用：接受退化")
    p.add_argument("--baseline", help="基线文件（默认 config 里的 gate.baseline）")
    _common(p)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("gonogo", help="在 test 集上给出 go / no-go 判定（04 §0）")
    p.add_argument("--run", required=True, help="我方 final_w6 的 run 名或 results.json 路径")
    p.add_argument("--tier-a", nargs="+", help="竞品 Tier A CSV（默认 $PINDOU_DATA_DIR/competitors/*/tier_a*.csv）")
    p.add_argument("--tier-b", required=True, help="最强竞品的 Tier B 结果目录（RecognitionResult 格式）")
    p.add_argument("--tier-b-entry", required=True, help="Tier B 结果对应的竞品条目：app/platform/import_path")
    p.add_argument("--allow-partial-tier-b", action="store_true", help="Tier B 缺图时只在已有结果的图上比较")
    p.add_argument("--device-timing-ms", type=float, help="T0.14 中端机实测 104×104 耗时（只报告）")
    p.add_argument("--device-name", help="实测机型")
    p.add_argument("--out", help="输出目录（默认在我方 test 报告目录下的 gonogo/）")
    _common(p)
    p.set_defaults(func=cmd_gonogo)

    p = sub.add_parser("calibrate-theta", help="只用 dev 集给出 θ 的建议值和依据")
    p.add_argument("--out", help="把 Markdown 结果另存到这个文件")
    p.add_argument("--allow-missing-labels", action="store_true")
    _common(p)
    p.set_defaults(func=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (UsageError, DatasetError, TestDisciplineError, CompetitorDataError, ConfigError) as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2
