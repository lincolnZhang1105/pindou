"""bench/eval 的 bootstrap、配对比较、门禁比较、Tier A 计分与 go/no-go 判定逻辑（不经过文件）。"""
import numpy as np
import pytest

from bench.eval import gate, gonogo
from bench.eval.competitors import CompetitorDataError, TierAEntry, TierAImage, score_tier_a
from bench.eval.compare import pair_results, paired_groups, verdict
from bench.eval.config import load_config
from bench.eval.dataset import make_unit
from bench.eval.scoring import score_image
from bench.eval.stats import boot_indices, paired_per_mille_diff, ratio_ci
from bench.eval.testing import grid_of, label, pred

CFG = load_config()


# --------------------------------------------------------------------------- bootstrap

def test_boot_indices_deterministic_and_seed_sensitive():
    a = boot_indices(7, 500, 1)
    assert np.array_equal(a, boot_indices(7, 500, 1))
    assert not np.array_equal(a, boot_indices(7, 500, 2))
    assert a.shape == (500, 7) and a.min() >= 0 and a.max() <= 6


def test_ratio_ci_degenerate_and_bernoulli():
    lo, hi = ratio_ci([2, 4, 6], [1000, 2000, 3000], 1000, 0, 0.95, 1000.0)  # 每张图都是 2‰
    assert lo == pytest.approx(2.0) and hi == pytest.approx(2.0)
    lo, hi = ratio_ci([1, 0], [1, 1], 10000, 0, 0.95)  # 两张图，重采样均值只可能是 0、0.5、1
    assert (lo, hi) == (0.0, 1.0)
    assert ratio_ci([], [], 100, 0, 0.95) == (None, None)
    assert ratio_ci([1], [0], 100, 0, 0.95) == (None, None)


def test_ratio_ci_is_micro_average():
    # 大图（1000 颗、10 个错）和小图（10 颗、5 个错）：微平均点估计是 15/1010，不是两图比例的平均
    lo, hi = ratio_ci([10, 5], [1000, 10], 10000, 0, 0.95, 1000.0)
    assert lo <= 1000 * 15 / 1010 <= hi


def test_paired_direction_and_sign():
    a = [0, 1, 0, 2, 1, 0, 0, 1]
    b = [5, 6, 4, 9, 7, 5, 6, 8]
    w = [1000] * 8
    d = paired_per_mille_diff(a, b, w, 5000, 3, 0.95)
    assert d["diff"] == pytest.approx(1000 * (sum(a) - sum(b)) / sum(w))
    assert d["diff"] < 0 and d["ci_high"] < 0 and d["p_a_better"] == 1.0  # A 每张图都更好
    assert verdict(d) == "A 显著更好"
    e = paired_per_mille_diff(b, a, w, 5000, 3, 0.95)
    assert e["diff"] == pytest.approx(-d["diff"]) and e["ci_low"] == pytest.approx(-d["ci_high"])
    assert verdict(e) == "B 显著更好"
    same = paired_per_mille_diff(a, a, w, 5000, 3, 0.95)
    assert (same["diff"], same["ci_low"], same["ci_high"]) == (0.0, 0.0, 0.0)
    assert verdict(same) == "不显著"


def test_paired_not_significant_when_one_image_drives_difference():
    a = [0] * 8
    b = [0] * 7 + [50]  # 只有一张图有差别
    d = paired_per_mille_diff(a, b, [1000] * 8, 5000, 3, 0.95)
    assert d["diff"] < 0 and d["ci_high"] == 0.0  # 重采样里经常抽不到那张图：上界为 0，不显著
    assert verdict(d) == "不显著"


# --------------------------------------------------------------------------- compare 的配对

def _res(iid, errors_cells, n=10, tags=()):
    lab = label(iid, grid_of("A1", n, 10), tags=tags)
    cells = grid_of("A1", n, 10)
    for k in range(errors_cells):
        cells[k // n][k % n] = "B2"
    return score_image(make_unit(lab, CFG, 10.0, 1000.0, label_sha=iid), pred(cells), CFG)


def test_compare_pairs_and_groups():
    ra = [_res("x", 1), _res("y", 0, tags=["watermark"])]
    rb = [_res("x", 5), _res("y", 3, tags=["watermark"]), _res("z", 1)]
    pairs, warns = pair_results(ra, rb)
    assert [p[0].image_id for p in pairs] == ["x", "y"] and warns  # z 只在 B 里：警告并只比较共同的图
    rows = dict(paired_groups(pairs, CFG))
    main = rows["主统计池（不含 T6、T7）"]
    assert main["pm_a"] == pytest.approx(1000 * 1 / 200) and main["pm_b"] == pytest.approx(1000 * 8 / 200)
    assert main["diff"] == pytest.approx(-35.0)
    assert rows["Hard 集"]["n"] == 1 and rows["Clean 集"]["n"] == 1


def test_compare_refuses_different_truth():
    a = _res("x", 1)
    b = _res("x", 1)
    b.label_sha = "other"
    with pytest.raises(Exception, match="真值不同"):
        pair_results([a], [b])


# --------------------------------------------------------------------------- gate

def test_gate_metrics_and_regressions():
    clean_ok = _res("c1", 0)
    clean_bad = _res("c2", 2)
    hard = _res("h", 3, tags=["watermark"])
    base = gate.gate_metrics([clean_ok, clean_bad, hard])
    assert base["grid_accuracy"] == {"num": 3, "den": 3, "value": 1.0}
    assert base["zero_error_rate"]["num"] == 1
    assert (base["clean_per_mille"]["num"], base["clean_per_mille"]["den"]) == (2, 200)  # Hard 图不进 Clean
    assert base["flag_recall"]["den"] == 5
    assert not any(r["regressed"] for r in gate.compare_to_baseline(base, base))

    worse_clean = gate.gate_metrics([clean_ok, _res("c2", 3), hard])  # Clean 多 1 个错误
    rows = {r["key"]: r["regressed"] for r in gate.compare_to_baseline(base, worse_clean)}
    assert rows == {"grid_accuracy": False, "zero_error_rate": False, "clean_per_mille": True, "flag_recall": False}

    worse_zero = gate.gate_metrics([_res("c1", 1), clean_bad, hard])
    rows = {r["key"]: r["regressed"] for r in gate.compare_to_baseline(base, worse_zero)}
    assert rows["zero_error_rate"] and rows["clean_per_mille"]

    failed = score_image(make_unit(label("c1", grid_of("A1", 10, 10)), CFG, 10.0, 1000.0), {"error": {"code": "x"}}, CFG)
    rows = {r["key"]: r["regressed"] for r in gate.compare_to_baseline(base, gate.gate_metrics([failed, clean_bad, hard]))}
    assert rows["grid_accuracy"] and rows["flag_recall"] is False  # 召回 0/… 与基线 0/5 相同，不算退化

    better = gate.gate_metrics([clean_ok, _res("c2", 0), _res("h", 0, tags=["watermark"])])
    assert not any(r["regressed"] for r in gate.compare_to_baseline(base, better))


def test_gate_recall_vacuous_and_exact_fractions():
    b = {"num": 0, "den": 0, "value": None}
    assert gate._frac("flag_recall", b) == 1
    base = {"grid_accuracy": {"num": 1, "den": 3}, "zero_error_rate": {"num": 1, "den": 3},
            "clean_per_mille": {"num": 1, "den": 3}, "flag_recall": {"num": 1, "den": 3}}
    cur = {"grid_accuracy": {"num": 2, "den": 6}, "zero_error_rate": {"num": 2, "den": 6},
           "clean_per_mille": {"num": 2, "den": 6}, "flag_recall": {"num": 0, "den": 0}}
    assert not any(r["regressed"] for r in gate.compare_to_baseline(base, cur))  # 1/3 与 2/6 精确相等


# --------------------------------------------------------------------------- Tier A 与 go/no-go 逻辑

def _units(specs):
    """specs: [(image_id, tags, type)]，10×10 全是 A1 的图（100 颗）。"""
    return [make_unit(label(i, grid_of("A1", 10, 10), tags=t, type=ty), CFG, 10.0, 1000.0, label_sha=i)
            for i, t, ty in specs]


def _entry(eid, rows):
    app, platform, path = eid.split("/")
    e = TierAEntry(eid, app, "1.0", platform, path, "2026-11-01")
    for iid, status, cols, rows_, counts in rows:
        e.images[iid] = TierAImage(iid, status, cols, rows_, counts)
    return e


def test_tier_a_bom_scoring():
    u = _units([("a", [], "T1")])
    e = _entry("X/android/image", [("a", "ok", 10, 10, {"A1": 95, "B2": 5})])
    r = score_tier_a(e, u)["a"]
    assert (r.status, r.bom_abs, r.bom_abs_raw) == ("ok", 10, 10)  # |95−100| + |5−0|
    e = _entry("X/android/image", [("a", "ok", 10, 9, {"A1": 100})])
    r = score_tier_a(e, u)["a"]
    assert (r.status, r.bom_abs, r.bom_abs_raw) == ("grid_wrong", 100, 0)  # 网格错：按没有输出计
    e = _entry("X/android/image", [("a", "error", None, None, {})])
    assert score_tier_a(e, u)["a"].bom_abs == 100


def _ours(units, errors):
    out = []
    for u in units:
        cells = grid_of("A1", 10, 10)
        for k in range(errors[u.image_id]):
            cells[k // 10][k % 10] = "B2"
        out.append(score_image(u, pred(cells), CFG))
    return out


def _decide(units, ours_err, comp_err, tier_a_entries, tierb_entry="X/android/image", not_cov=()):
    ours = _ours(units, ours_err)
    ta = gonogo.tier_a_table(tier_a_entries, units, {r.image_id: r for r in ours})
    hard = [u for u in units if u.in_main and u.hard and u.image_id not in not_cov]
    comp = _ours(hard, comp_err)
    return ta, gonogo.evaluate(CFG, units, ours, ta, tierb_entry, comp, list(not_cov), bool(not_cov), None, None)


SPECS = [(f"h{i}", ["watermark"], "T1") for i in range(8)] + [(f"c{i}", [], "T1") for i in range(4)] + \
        [("p6", [], "T6"), ("p7", [], "T7")]


def _tier_a(units, x_err=2, y_err=5):
    rows_x = [(u.image_id, "ok", 10, 10, {"A1": 100 - x_err, "B2": x_err}) for u in units if u.in_main]
    rows_y = [(u.image_id, "ok", 10, 10, {"A1": 100 - y_err, "B2": y_err}) for u in units if u.in_main]
    return {"X/android/image": _entry("X/android/image", rows_x), "Y/ios/image": _entry("Y/ios/image", rows_y)}


def test_gonogo_go():
    units = _units(SPECS)
    ours_err = {u.image_id: 0 for u in units} | {"h0": 1, "p6": 30}  # T6 的错误不影响第 1、2、4 条
    comp_err = {f"h{i}": 5 + i for i in range(8)}
    ta, res = _decide(units, ours_err, comp_err, _tier_a(units))
    assert ta["strongest"] == "X/android/image" and len(ta["common"]) == 12  # T6、T7 不参与 Tier A 排名
    assert res["verdict"] == "GO"
    c = {x.no: x for x in res["criteria"]}
    assert c[1].passed and c[2].passed and c[4].passed
    assert res["paired"]["pm_b"] == pytest.approx(1000 * 1 / 800)  # 我方（B 位置）Hard 8 张共 1 个错误
    assert res["paired"]["diff"] > 0 and res["paired"]["ci_low"] > 0  # Δ = 竞品 − 我方


def test_gonogo_ratio_boundary_is_inclusive():
    units = _units(SPECS)
    ours_err = {u.image_id: 0 for u in units} | {f"h{i}": 1 for i in range(8)}
    comp_err = {f"h{i}": 2 for i in range(8)}  # 我方恰好是竞品的 50%
    _, res = _decide(units, ours_err, comp_err, _tier_a(units))
    assert res["verdict"] == "GO"
    comp_err = {f"h{i}": 2 for i in range(7)} | {"h7": 1}  # 竞品少 1 个错：我方 8 > 0.5 × 15
    _, res = _decide(units, ours_err, comp_err, _tier_a(units))
    assert res["verdict"] == "NO-GO" and res["criteria"][0].passed is False


def test_gonogo_not_significant_gives_nogo_with_expand_note():
    units = _units(SPECS)
    ours_err = {u.image_id: 0 for u in units}
    comp_err = {f"h{i}": 0 for i in range(7)} | {"h7": 40}  # 只有一张图有差别：比值满足、差值不显著
    _, res = _decide(units, ours_err, comp_err, _tier_a(units))
    c1 = res["criteria"][0]
    assert res["verdict"] == "NO-GO" and c1.passed is False
    assert any("而不是降低门槛" in n for n in c1.notes)


def test_gonogo_clean_and_grid_thresholds():
    units = _units(SPECS)
    comp_err = {f"h{i}": 10 for i in range(8)}
    ours_err = {u.image_id: 0 for u in units} | {"c0": 1}  # Clean 400 颗 1 个错 = 2.5‰ > 1.0
    _, res = _decide(units, ours_err, comp_err, _tier_a(units))
    assert res["verdict"] == "NO-GO" and res["criteria"][1].passed is False and res["criteria"][0].passed

    ours = _ours(units, {u.image_id: 0 for u in units})
    bad = score_image(units[8], {"error": {"code": "internal"}}, CFG)  # c0 失败：网格正确率 11/12 < 98%
    ours = [bad if r.image_id == "c0" else r for r in ours]
    ta = gonogo.tier_a_table(_tier_a(units), units, {r.image_id: r for r in ours})
    comp = _ours([u for u in units if u.hard and u.in_main], comp_err)
    res = gonogo.evaluate(CFG, units, ours, ta, "X/android/image", comp, [], False, None, None)
    c = {x.no: x for x in res["criteria"]}
    assert res["verdict"] == "NO-GO" and c[4].passed is False and c[2].passed is False


def test_gonogo_requires_strongest_and_full_tier_b():
    units = _units(SPECS)
    ours_err = {u.image_id: 0 for u in units}
    comp_err = {f"h{i}": 5 for i in range(8)}
    with pytest.raises(CompetitorDataError, match="最强竞品"):
        _decide(units, ours_err, comp_err, _tier_a(units), tierb_entry="Y/ios/image")
    ours = _ours(units, ours_err)
    ta = gonogo.tier_a_table(_tier_a(units), units, {r.image_id: r for r in ours})
    comp = _ours([u for u in units if u.hard and u.image_id != "h7"], comp_err)
    with pytest.raises(CompetitorDataError, match="缺 1 张"):
        gonogo.evaluate(CFG, units, ours, ta, "X/android/image", comp, ["h7"], False, None, None)
    res = gonogo.evaluate(CFG, units, ours, ta, "X/android/image", comp, ["h7"], True, None, None)
    assert res["paired"]["n"] == 7 and any("allow-partial" in w for w in res["warnings"])


def test_tier_a_strongest_uses_common_coverage_and_penalized_bom():
    units = _units(SPECS)
    ta = _tier_a(units, x_err=3, y_err=2)
    assert gonogo.tier_a_table(ta, units, {})["strongest"] == "Y/ios/image"
    # Y 有一张图网格错：按没有输出计（100 颗），总误差反而更大；敏感性检查给出警告
    ta["Y/ios/image"].images["c0"] = TierAImage("c0", "ok", 10, 9, {"A1": 98, "B2": 2})
    t = gonogo.tier_a_table(ta, units, {})
    assert t["strongest"] == "X/android/image"
    assert any("敏感性检查" in w for w in t["warnings"])
    # X 少测一张：只在共同覆盖的图上比较
    del ta["X/android/image"].images["c1"]
    t = gonogo.tier_a_table(ta, units, {})
    assert len(t["common"]) == 11 and any("只覆盖了" in w for w in t["warnings"])
    ta["X/android/image"].images["zzz"] = TierAImage("zzz", "error", None, None, {})
    with pytest.raises(CompetitorDataError, match="不属于 test 集"):
        gonogo.tier_a_table(ta, units, {})
