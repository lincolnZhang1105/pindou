"""bench/eval 命令行端到端：私有数据目录、报告文件、test 集纪律、门禁、配对比较、go/no-go、θ 标定。"""
import csv
import json

import pytest

from bench.common import formats as F
from bench.eval.cli import main
from bench.eval.competitors import CompetitorDataError, parse_tier_a
from bench.eval.testing import Env, grid_of, label, pred

LAB_A = label("a", [["A1", None, "B2"], [None, "A1", "C3"]], legend=[("A1", 2), ("B2", 1), ("C3", 1)])
PRED_A = pred([["A01", None, "B3"], ["A1", None, "C3"]], legend=[("A1", 2), ("B2", 1), ("C3", 1)],
              timing={"grid": 10, "classify": 30})
LAB_B = label("b", [["A1", "A1"], ["B2", "B2"]], legend=[("A1", 2), ("B2", 2)])
PRED_B = pred([["A1", "A1"], ["B2", "B2"]], legend=[("A1", 2), ("B2", 1)])
LAB_C = label("c", [["A1", None], ["C3", "C3"]])
PRED_C = pred([["A1", None, "A1"], ["C3", "C3", None]])


def three(env: Env, run="r1", split="dev"):
    for lab in (LAB_A, LAB_B, LAB_C):
        env.add(lab)
    for iid, p in (("a", PRED_A), ("b", PRED_B), ("c", PRED_C)):
        env.put_pred(run, iid, p)
    env.split(split, ["a", "b", "c"])


def fails(capsys, argv, needle):
    """命令必须以退出码 2 拒绝，且 stderr 里有指定的原因。"""
    capsys.readouterr()
    rc = main(argv)
    err = capsys.readouterr().err
    assert rc == 2 and needle in err, (rc, err)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- run

def test_run_dev_writes_reports(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    three(env)
    env.split("test", ["zzz"])
    assert main(env.args("run", "--preds", "r1", "--split", "dev")) == 0
    out = env.root / "reports" / "r1" / "dev"
    rows = {r["image_id"]: r for r in read_csv(out / "per_image.csv")}
    assert rows["a"]["errors"] == "3" and rows["a"]["status"] == "ok" and rows["a"]["wrong_code"] == "1"
    assert rows["c"]["status"] == "grid_wrong" and rows["c"]["errors"] == "3" and rows["c"]["zero_error"] == "0"
    assert rows["a"]["clean"] == "1" and rows["a"]["timing_ms"] == "40"
    summ = [r for r in read_csv(out / "summary.csv") if r["group"] == "main" and r["metric"] == "per_mille"]
    assert float(summ[0]["value"]) == pytest.approx(1000 * 6 / 11) and summ[0]["num"] == "6" and summ[0]["den"] == "11"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "545.455" in md and "网格尺寸错" in md and "Clean" in md
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["split"] == "dev" and res["test_reason"] is None and len(res["images"]) == 3
    assert env.usage() == []  # dev 集不登记使用记录
    # 结果可复现：再跑一次，summary.csv 完全相同
    first = (out / "summary.csv").read_bytes()
    assert main(env.args("run", "--preds", "r1", "--split", "dev")) == 0
    assert (out / "summary.csv").read_bytes() == first


def test_run_missing_pred_and_unmatched_file(tmp_path, monkeypatch):
    env = Env(tmp_path, monkeypatch)
    three(env)
    (env.root / "preds" / "r1" / "b.json").unlink()
    env.put_pred("r1", "b_v2", PRED_B)  # 命名不一致的文件
    assert main(env.args("run", "--preds", "r1", "--split", "dev")) == 0
    res = json.loads((env.root / "reports" / "r1" / "dev" / "results.json").read_text(encoding="utf-8"))
    b = [r for r in res["images"] if r["image_id"] == "b"][0]
    assert b["status"] == "missing" and b["errors"] == 4
    assert any("没有预测结果文件" in w for w in res["warnings"])
    assert any("没有对应本 split" in w for w in res["warnings"])


def test_invalid_label_or_overlap_refused(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    three(env)
    bad = dict(LAB_B, legend=[{"code": "A1", "count": 3}, {"code": "B2", "count": 2}])  # 图例校验和不对
    F.save_json(env.root / "labels" / "b.json", bad)
    fails(capsys, env.args("run", "--preds", "r1", "--split", "dev"), "图例校验不通过")
    F.save_json(env.root / "labels" / "b.json", LAB_B)
    env.split("test", ["c"])
    fails(capsys, env.args("run", "--preds", "r1", "--split", "dev"), "重叠")


def test_missing_label_and_allow_flag(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    three(env)
    env.split("dev", ["a", "b", "c", "nolabel"])
    fails(capsys, env.args("run", "--preds", "r1", "--split", "dev"), "找不到真值")
    assert main(env.args("run", "--preds", "r1", "--split", "dev", "--allow-missing-labels")) == 0


def test_paged_t4_unit_via_manifest_group(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    cells = grid_of("A1", 4, 4)
    cells[3][3] = "B2"
    lab = label("grp", cells, type="T4")
    lab["images"] = [{"file": "grp_0.png", "sha256": ""}, {"file": "grp_1.png", "sha256": ""}]
    p0 = dict(lab["pages"][0], image_index=0, rows=2, y_lines=[0, 10, 20])
    p1 = dict(lab["pages"][0], image_index=1, rows=2, row_offset=2, y_lines=[0, 10, 20])
    lab["pages"] = [p0, p1]
    from bench.eval.testing import write_png
    for i in (0, 1):
        write_png(env.root / "images" / f"grp_{i}.png", 44, 24)
    F.save_json(env.root / "labels" / "grp.json", lab)
    F.write_jsonl(env.datasets / "manifest.jsonl", [{"image_id": f"grp_{i}", "file": f"grp_{i}.png", "group": "grp"}
                                                     for i in (0, 1)])
    env.split("dev", ["grp_0", "grp_1"])
    env.put_pred("r1", "grp", pred(grid_of("A1", 4, 4)))  # 合成后的整张图纸，按单元 key 命名
    assert main(env.args("run", "--preds", "r1", "--split", "dev")) == 0
    res = json.loads((env.root / "reports" / "r1" / "dev" / "results.json").read_text(encoding="utf-8"))
    (r,) = res["images"]
    assert r["image_id"] == "grp" and r["size_bucket"] == "XL" and r["hard"] and r["errors"] == 1 and r["n_beads"] == 16
    env.put_pred("r1", "grp_0", pred(grid_of("A1", 4, 2)))  # 又按分页名写了一份：有歧义，拒绝
    fails(capsys, env.args("run", "--preds", "r1", "--split", "dev"), "有歧义")


# --------------------------------------------------------------------------- test 集纪律

def _frozen_test_env(tmp_path, monkeypatch, **kw):
    env = Env(tmp_path, monkeypatch, blurry__theta_frozen=True, **kw)
    three(env, split="test")
    env.split("dev", ["zzz"])
    return env


def test_test_split_requires_lock_reason_and_frozen_theta(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)  # θ 未冻结
    three(env, split="test")
    env.lock(["a", "b", "c"])
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2"), "占位值")
    env2 = _frozen_test_env(tmp_path / "x", monkeypatch)
    fails(capsys, env2.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2"), "尚未锁定")
    env2.lock(["a", "b", "c"])
    fails(capsys, env2.args("run", "--preds", "r1", "--split", "test"), "必须带 --test-reason")
    with pytest.raises(SystemExit):
        main(env2.args("run", "--preds", "r1", "--split", "test", "--test-reason", "tuning"))
    fails(capsys, env2.args("run", "--preds", "r1", "--split", "dev", "--test-reason", "baseline_w2"),
          "只用于 --split test")
    assert env2.usage() == []


def test_test_split_each_reason_once(tmp_path, monkeypatch, capsys):
    env = _frozen_test_env(tmp_path, monkeypatch)
    env.lock(["a", "b", "c"])
    assert main(env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2")) == 0
    (rec,) = env.usage()
    assert rec["reason"] == "baseline_w2" and rec["run"] == "r1" and {"at", "git_rev"} <= set(rec)
    out = env.root / "reports" / "r1" / "test" / "results.json"
    assert rec["results_sha256"] == F.sha256_file(out)
    # 同一个 reason 再用一次：拒绝，不追加记录、不写报告
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2", "--name", "r2"),
          "只能用一次")
    assert len(env.usage()) == 1 and not (env.root / "reports" / "r2").exists()
    # 另一个 reason，但会覆盖已有的 test 报告：拒绝
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "final_w6"), "拒绝覆盖")
    assert len(env.usage()) == 1
    assert main(env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "final_w6", "--name", "v1")) == 0
    assert [r["reason"] for r in env.usage()] == ["baseline_w2", "final_w6"]


def test_test_lock_tampering_refused(tmp_path, monkeypatch, capsys):
    env = _frozen_test_env(tmp_path, monkeypatch)
    env.lock(["a", "b", "c"])
    env.split("test", ["a", "b"])  # 锁定后改了 test.txt
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2"), "ids_sha256")
    env.split("test", ["a", "b", "c"])
    from bench.eval.testing import write_png
    write_png(env.root / "images" / "a.png", 34, 24, sharp=False)  # 锁定后改了图片
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2"),
          "sha256 与 test.lock 不一致")
    fails(capsys, env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2",
                           "--allow-missing-labels"), "不允许 --allow-missing-labels")
    assert env.usage() == []


# --------------------------------------------------------------------------- gate

def test_gate_lifecycle(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    three(env)
    assert main(env.args("gate", "--preds", "r1")) == 3  # 没有基线
    assert main(env.args("gate", "--preds", "r1", "--update")) == 0
    base = json.loads((tmp_path / "baseline_dev.json").read_text(encoding="utf-8"))
    assert base["metrics"]["clean_per_mille"] == {"num": 6, "den": 11, "value": pytest.approx(1000 * 6 / 11)}
    assert base["metrics"]["grid_accuracy"]["num"] == 2
    assert main(env.args("gate", "--preds", "r1")) == 0
    # 退化：b 多一个错误
    for iid, p in (("a", PRED_A), ("c", PRED_C)):
        env.put_pred("r2", iid, p)
    env.put_pred("r2", "b", pred([["A1", "B2"], ["B2", "B2"]]))
    capsys.readouterr()
    assert main(env.args("gate", "--preds", "r2")) == 1
    assert "退化" in capsys.readouterr().out
    assert main(env.args("gate", "--preds", "r2", "--update")) == 1  # 有退化时拒绝更新
    assert main(env.args("gate", "--preds", "r2", "--update", "--allow-regression")) == 0
    # 改进：c 的网格修对了
    for iid, p in (("a", PRED_A), ("b", PRED_B)):
        env.put_pred("r3", iid, p)
    env.put_pred("r3", "c", pred([["A1", None], ["C3", "C3"]]))
    assert main(env.args("gate", "--preds", "r3")) == 0
    # 真值变了：基线失效
    F.save_json(env.root / "labels" / "c.json", label("c", [["A1", "A1"], ["C3", "C3"]]))
    assert main(env.args("gate", "--preds", "r3")) == 3


# --------------------------------------------------------------------------- compare

def test_compare_dev(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    three(env)
    for lab in (LAB_A, LAB_B, LAB_C):
        env.put_pred("perfect", lab["image_id"], pred(lab["cells"]))
    capsys.readouterr()
    assert main(env.args("compare", "--a", "perfect", "--b", "r1", "--split", "dev")) == 0
    out = capsys.readouterr().out
    assert "-545.455" in out  # Δ = 0 − 545.455


def test_compare_test_uses_recorded_results(tmp_path, monkeypatch, capsys):
    env = _frozen_test_env(tmp_path, monkeypatch)
    env.lock(["a", "b", "c"])
    for lab in (LAB_A, LAB_B, LAB_C):
        env.put_pred("v2", lab["image_id"], pred(lab["cells"]))
    assert main(env.args("run", "--preds", "r1", "--split", "test", "--test-reason", "baseline_w2")) == 0
    assert main(env.args("run", "--preds", "v2", "--split", "test", "--test-reason", "final_w6")) == 0
    capsys.readouterr()
    assert main(env.args("compare", "--a", "v2", "--b", "r1", "--split", "test")) == 0
    assert "-545.455" in capsys.readouterr().out
    assert len(env.usage()) == 2  # compare 只读已登记的结果，不消耗 reason
    p = env.root / "reports" / "r1" / "test" / "results.json"
    p.write_text(p.read_text(encoding="utf-8").replace('"errors": 3', '"errors": 0', 1), encoding="utf-8")
    fails(capsys, env.args("compare", "--a", "v2", "--b", "r1", "--split", "test"), "被改动过")


# --------------------------------------------------------------------------- gonogo

def _write_tier_a(path, app, platform, rows, version="1.0"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["app", "app_version", "platform", "import_path", "tested_at", "image_id", "status", "cols", "rows",
                    "code", "count"])
        for iid, status, cols, rows_, counts in rows:
            if status == "error":
                w.writerow([app, version, platform, "image", "2026-11-01", iid, "error", "", "", "", ""])
            for code, n in counts.items():
                w.writerow([app, version, platform, "image", "2026-11-01", iid, status, cols, rows_, code, n])


def _gonogo_env(tmp_path, monkeypatch, comp_errors):
    env = Env(tmp_path, monkeypatch, blurry__theta_frozen=True)
    ids = [f"h{i}" for i in range(8)] + [f"c{i}" for i in range(4)]
    for iid in ids:
        env.add(label(iid, grid_of("A1", 10, 10), tags=["watermark"] if iid.startswith("h") else []))
        env.put_pred("ours", iid, pred(grid_of("A1", 10, 10)))
    cells = grid_of("A1", 10, 10)
    cells[0][0] = "B2"
    env.put_pred("ours", "h0", pred(cells))  # 我方 Hard 共 1 个错误
    for i in range(8):
        cells = grid_of("A1", 10, 10)
        for k in range(comp_errors[i]):
            cells[k // 10][k % 10] = "B2"
        F.save_json(env.root / "competitors" / "X" / "tierb" / f"h{i}.json", pred(cells))
    env.split("test", ids)
    env.split("dev", ["zzz"])
    env.lock(ids)
    _write_tier_a(env.root / "competitors" / "X" / "tier_a_image.csv", "X", "android",
                  [(i, "ok", 10, 10, {"A1": 98, "B2": 2}) for i in ids])
    _write_tier_a(env.root / "competitors" / "Y" / "tier_a_image.csv", "Y", "ios",
                  [(i, "ok", 10, 10, {"A1": 95, "B02": 5}) for i in ids[:-1]] + [(ids[-1], "error", "", "", {})])
    return env


def test_gonogo_end_to_end_go(tmp_path, monkeypatch, capsys):
    env = _gonogo_env(tmp_path, monkeypatch, [5, 6, 7, 8, 9, 10, 11, 12])
    tierb = str(env.root / "competitors" / "X" / "tierb")
    # baseline_w2 的结果不能用来判定
    assert main(env.args("run", "--preds", "ours", "--split", "test", "--test-reason", "baseline_w2", "--name", "w2")) == 0
    fails(capsys, env.args("gonogo", "--run", "w2", "--tier-b", tierb, "--tier-b-entry", "X/android/image"),
          "需要 final_w6")
    assert main(env.args("run", "--preds", "ours", "--split", "test", "--test-reason", "final_w6")) == 0
    capsys.readouterr()
    rc = main(env.args("gonogo", "--run", "ours", "--tier-b", tierb, "--tier-b-entry", "X/android/image",
                       "--device-timing-ms", "2500", "--device-name", "骁龙 7 Gen 1"))
    out = capsys.readouterr().out
    assert rc == 0 and "结论：GO" in out
    doc = json.loads((env.root / "reports" / "ours" / "test" / "gonogo" / "gonogo.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == "GO" and doc["strongest"] == "X/android/image"
    assert doc["paired"]["diff"] == pytest.approx(1000 * (68 - 1) / 800)
    assert len(env.usage()) == 2  # gonogo 不追加使用记录
    # Tier B 给的不是最强竞品
    fails(capsys, env.args("gonogo", "--run", "ours", "--tier-b", tierb, "--tier-b-entry", "Y/ios/image"),
          "必须是最强竞品")


def test_gonogo_end_to_end_nogo(tmp_path, monkeypatch, capsys):
    env = _gonogo_env(tmp_path, monkeypatch, [1, 1, 0, 0, 0, 0, 0, 0])  # 竞品只比我方多 1 个错：比值不满足
    assert main(env.args("run", "--preds", "ours", "--split", "test", "--test-reason", "final_w6")) == 0
    tierb = str(env.root / "competitors" / "X" / "tierb")
    assert main(env.args("gonogo", "--run", "ours", "--tier-b", tierb, "--tier-b-entry", "X/android/image")) == 1
    md = (env.root / "reports" / "ours" / "test" / "gonogo" / "gonogo.md").read_text(encoding="utf-8")
    assert "NO-GO" in md
    # Tier B 缺图：默认拒绝判定
    (env.root / "competitors" / "X" / "tierb" / "h7.json").unlink()
    fails(capsys, env.args("gonogo", "--run", "ours", "--tier-b", tierb, "--tier-b-entry", "X/android/image"), "缺 1 张")
    # 真值在 final_w6 之后被改：拒绝判定
    F.save_json(env.root / "labels" / "c0.json", label("c0", grid_of("B2", 10, 10)))
    fails(capsys, env.args("gonogo", "--run", "ours", "--tier-b", tierb, "--tier-b-entry", "X/android/image",
                           "--allow-partial-tier-b"), "运行时不同")


# --------------------------------------------------------------------------- calibrate-theta

def test_calibrate_theta(tmp_path, monkeypatch, capsys):
    env = Env(tmp_path, monkeypatch)
    ids = []
    for i in range(3):
        env.add(label(f"s{i}", grid_of("A1", 10, 10)), sharp=True)
        env.add(label(f"g{i}", grid_of("A1", 10, 10), tags=["blurry"]), sharp=False)  # 纯灰：方差 0
        ids += [f"s{i}", f"g{i}"]
    env.split("dev", ids)
    capsys.readouterr()
    assert main(env.args("calibrate-theta")) == 0
    out = capsys.readouterr().out
    assert "建议 θ" in out and "Youden J = 1.000" in out
    env2 = Env(tmp_path / "y", monkeypatch)
    for i in range(2):
        env2.add(label(f"s{i}", grid_of("A1", 10, 10)))
    env2.split("dev", ["s0", "s1"])
    assert main(env2.args("calibrate-theta")) == 1  # 没有人工 blurry 标签：无法标定


# --------------------------------------------------------------------------- Tier A CSV

def test_tier_a_csv_parsing(tmp_path):
    p = tmp_path / "a.csv"
    _write_tier_a(p, "拼豆大王", "android", [("x", "ok", 10, 10, {"A01": 3, "COCO-291:B2": 1}),
                                            ("y", "error", "", "", {})])
    (e,) = parse_tier_a([p]).values()
    assert e.entry_id == "拼豆大王/android/image" and e.images["x"].counts == {"A1": 3, "COCO-291:B2": 1}
    assert e.images["y"].status == "error"
    bad = tmp_path / "b.csv"
    _write_tier_a(bad, "X", "android", [("x", "ok", 10, 10, {"A1": 3, "A01": 1})])
    with pytest.raises(CompetitorDataError, match="重复"):
        parse_tier_a([bad])
    _write_tier_a(bad, "X", "android", [("x", "ok", 10, 10, {"A1": 3})])
    _write_tier_a(tmp_path / "c.csv", "X", "android", [("y", "ok", 10, 10, {"A1": 3})], version="2.0")
    with pytest.raises(CompetitorDataError, match="不一致"):
        parse_tier_a([bad, tmp_path / "c.csv"])
    bad.write_text("app,image_id\nX,x\n", encoding="utf-8")
    with pytest.raises(CompetitorDataError, match="缺少列"):
        parse_tier_a([bad])
