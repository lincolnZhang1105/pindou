"""bench/eval 逐图计分与汇总：用手工构造、手算结果的小样例逐个验证指标（T0.7 DoD）。"""
from fractions import Fraction

import pytest

from bench.common.formats import validate_label
from bench.eval.aggregate import compute
from bench.eval.codes import CodeMapper, brand_of
from bench.eval.config import load_config
from bench.eval.dataset import make_unit, min_cell_px_of
from bench.eval.scoring import score_image, truncate_flags
from bench.eval.testing import grid_of, label, pred

CFG = load_config()  # θ = 100（占位），min_cell_px = 8，上限 2% / 30 格


def unit(lab, min_cell=10.0, lap=1000.0):
    assert validate_label(lab) == [], validate_label(lab)
    return make_unit(lab, CFG, min_cell, lap)


def score(lab, p, **kw):
    return score_image(unit(lab, **kw), p, CFG)


# --------------------------------------------------------------------------- 手算的 3 张图样例

LAB_A = label("a", [["A1", None, "B2"], [None, "A1", "C3"]], legend=[("A1", 2), ("B2", 1), ("C3", 1)])
# (0,2) 色号错；(1,0) 空格→豆子；(1,1) 豆子→空格。图例和真值一致（色号写成 A01、品牌写 MARD-291 也算同一个）
PRED_A = pred([["A01", None, "B3"], ["A1", None, "C3"]], palette="MARD-291", uncertain=[2, 3],
              legend=[("A01", 2), ("B2", 1), ("C3", 1)])
LAB_B = label("b", [["A1", "A1"], ["B2", "B2"]], legend=[("A1", 2), ("B2", 2)])
PRED_B = pred([["A1", "A1"], ["B2", "B2"]], legend=[("A1", 2), ("B2", 1)])  # 格子全对，图例颗数错
LAB_C = label("c", [["A1", None], ["C3", "C3"]])
PRED_C = pred([["A1", None, "A1"], ["C3", "C3", None]])  # 3 列 × 2 行，真值 2 × 2：网格尺寸错 → 失败


def three():
    return [score(LAB_A, PRED_A), score(LAB_B, PRED_B), score(LAB_C, PRED_C)]


def test_per_image_counts_hand_computed():
    a, b, c = three()
    assert (a.status, a.n_beads, a.n_empty) == ("ok", 4, 2)
    assert (a.wrong_code, a.bead_to_empty, a.empty_to_bead, a.errors, a.correct_beads) == (1, 1, 1, 3, 2)
    assert a.bom_abs == 2  # |B2: 0−1| + |B3: 1−0|，A1、C3 颗数都对
    assert a.legend_ok is True
    assert (b.errors, b.zero_error, b.legend_ok, b.bom_abs) == (0, True, False, 0)
    assert c.status == "grid_wrong" and c.failed and not c.zero_error
    assert (c.errors, c.correct_beads, c.bom_abs) == (3, 0, 3)  # 失败：所有非空真值格（3 颗）按错误计
    assert c.bom_abs_raw == 1  # 原始用量误差（A1 多 1 颗）单独保留，不进主指标
    assert c.legend_ok is None and c.has_legend is False


def test_aggregate_hand_computed():
    m = compute(three(), CFG, ci=False)
    assert m["per_mille"].value == pytest.approx(1000 * 6 / 11)  # (3+0+3) / (4+4+3)
    assert (m["per_mille"].num, m["per_mille"].den) == (6, 11)
    assert m["nonempty_acc_macro"].value == pytest.approx((2 / 4 + 4 / 4 + 0) / 3)
    assert m["zero_error_rate"].value == pytest.approx(1 / 3)
    assert m["grid_accuracy"].value == pytest.approx(2 / 3)
    assert m["failure_rate"].value == pytest.approx(1 / 3)
    assert m["bom_error"].value == pytest.approx(5 / 11)  # (2+0+3)/11，失败图按「没有输出」计
    # 空格误判只在成功的图上统计：空格→豆子 1 次 / 空格 2 个；豆子→空格 1 次 / 豆子 8 颗
    assert m["empty_to_bead_rate"].value == pytest.approx(1 / 2)
    assert m["bead_to_empty_rate"].value == pytest.approx(1 / 8)
    assert m["legend_accuracy"].value == pytest.approx(1 / 2)  # 只统计真值带图例的 a、b
    # 小图的标记上限：4 颗 × 2% 向下取整 = 0，a 标记的 2 格全部被截掉
    assert m["flag_recall"].value == 0 and m["flag_rate_raw"].value == pytest.approx(2 / 8)
    assert m["flag_precision"].value is None  # 截断后没有保留任何标记


def test_ci_brackets_value_and_is_deterministic():
    rs = three()
    m1 = compute(rs, CFG)
    m2 = compute(rs, CFG)
    for k, x in m1.items():
        assert (x.ci_low, x.ci_high) == (m2[k].ci_low, m2[k].ci_high), k
        if x.value is not None and x.ci_low is not None:
            assert x.ci_low - 1e-9 <= x.value <= x.ci_high + 1e-9, k


# --------------------------------------------------------------------------- 失败计分

@pytest.mark.parametrize("p,status", [
    ({"error": {"code": "no_grid_found", "message": "x"}}, "error"),
    (None, "missing"),
    ({"grid": {"cols": 2, "rows": 2}, "codebook": [{"code": "A1"}], "cells": [1, 1, 1]}, "invalid"),  # 长度不对
    ({"grid": {"cols": 2, "rows": 2}, "codebook": [{"code": "A1"}], "cells": [1, 1, 1, 2]}, "invalid"),  # 索引越界
    ({"grid": {"cols": 2, "rows": 2}, "cells": [1, 1, 1, 1]}, "invalid"),  # 没有 codebook
])
def test_failures_count_all_beads_as_errors(p, status):
    r = score(LAB_B, p)
    assert r.status == status and r.failed
    assert (r.errors, r.correct_beads, r.bom_abs, r.zero_error) == (4, 0, 4, False)
    assert r.legend_ok is False  # 真值有图例，失败的输出图例记为不正确
    assert r.flag_hits == 0 and r.flag_kept == 0


def test_error_code_recorded():
    r = score(LAB_B, {"error": {"code": "needs_user_crop", "message": ""}})
    assert r.error_code == "needs_user_crop"


def test_grid_transposed_is_wrong():
    lab = label("t", [["A1", "A1", "A1"], ["A1", "A1", "A1"]])
    r = score(lab, pred(grid_of("A1", 2, 3)))  # 2 列 × 3 行，真值 3 列 × 2 行
    assert r.status == "grid_wrong" and r.errors == 6
    assert r.bom_abs_raw == 0 and r.bom_abs == 6


def test_empty_image_zero_beads():
    lab = label("e", [[None, None], [None, None]])
    ok = score(lab, pred([[None, None], [None, None]]))
    assert ok.n_beads == 0 and ok.zero_error and ok.nonempty_acc is None
    bad = score(lab, pred([[None, "A1"], [None, None]]))
    assert bad.errors == 1 and bad.empty_to_bead == 1 and not bad.zero_error
    m = compute([ok, bad], CFG, ci=False)
    assert m["nonempty_acc_macro"].value is None  # 没有豆子的图不进宏平均
    assert m["per_mille"].value is None  # 分母为 0


# --------------------------------------------------------------------------- 色号与品牌

def test_code_normalization_and_brand_rules():
    assert brand_of("MARD-221") == brand_of("MARD-291") == "MARD"
    assert brand_of("HAMA-MIDI") == "HAMA" and brand_of(None) is None
    m = CodeMapper("MARD-221")
    assert m.raw_key("a01") == "A1"
    assert m.raw_key("MARD:A01") == "A1"  # 主品牌前缀去掉
    assert m.raw_key("COCO-291:a01") == "COCO:A1"
    assert m.pred_key("COCO-291", "A01") == "COCO:A1"
    assert m.pred_key(None, "A01") == "A1"
    assert m.pred_key("MARD-291", "A1") == "A1"
    assert CodeMapper(None).pred_key("COCO-291", "A1") == "A1"  # 真值缺 palette_hint：不区分品牌


def test_mixed_brand_cells_and_wrong_brand():
    lab = label("m", [["A1", "COCO-291:B2"]])
    r = score(lab, pred([["A1", ("COCO-291", "B02")]]))
    assert r.errors == 0
    r = score(lab, pred([["A1", "B2"]]))  # 漏了品牌：MARD 的 B2 ≠ COCO 的 B2
    assert r.wrong_code == 1
    r = score(lab, pred([["A1", "B2"]], palette="COCO-291"))  # 整张认成 COCO：MARD 的 A1 那格错，COCO 的 B2 那格对
    assert r.wrong_code == 1
    r = score(label("mm", [["A1", "B2"]]), pred([["A1", "B2"]], palette="COCO-291"))  # 纯 MARD 图纸整张认成 COCO
    assert r.wrong_code == 2 and r.bom_abs == 4


# --------------------------------------------------------------------------- 图例

def test_legend_rules():
    lab = LAB_B  # A1×2, B2×2
    ok = pred(lab["cells"], legend=[("A1", 1), ("A01", 1), ("B2", 2)])  # 同一色号多行：颗数相加
    assert score(lab, ok).legend_ok is True
    extra = pred(lab["cells"], legend=[("A1", 2), ("B2", 2), ("H2", 0)])  # 多出一个色号（即使颗数是 0）
    assert score(lab, extra).legend_ok is False
    none = pred(lab["cells"])
    assert score(lab, none).legend_ok is False  # 没有输出图例
    gw = pred(grid_of("A1", 3, 2), legend=[("A1", 2), ("B2", 2)], grid=(3, 2))
    r = score(lab, gw)
    assert r.status == "grid_wrong" and r.legend_ok is True  # 网格错了，图例照样单独评
    bad = pred(lab["cells"])
    bad["legend"] = [{"code": "A1", "count": -1}]
    assert score(lab, bad).legend_ok is False


# --------------------------------------------------------------------------- 不确定标记（上限内的召回）

def test_flag_cap_values():
    assert CFG.flag_cap(49) == 0 and CFG.flag_cap(50) == 1 and CFG.flag_cap(100) == 2
    assert CFG.flag_cap(1500) == 30 and CFG.flag_cap(3000) == 30  # 2% 与 30 取小


def _hundred(uncertain=None, confidence=None):
    lab = label("h", grid_of("A1", 10, 10))
    cells = grid_of("A1", 10, 10)
    cells[0][0] = cells[0][1] = cells[0][2] = "B2"  # 错误格：编号 0、1、2
    return lab, pred(cells, uncertain=uncertain, confidence=confidence)


def test_recall_truncated_by_confidence():
    conf = [0.99] * 100
    conf[0], conf[1], conf[50], conf[60] = 0.9, 0.2, 0.1, 0.3
    lab, p = _hundred(uncertain=[0, 1, 50, 60], confidence=conf)
    r = score(lab, p)
    # 上限 2 格，按 confidence 升序保留 50（0.1）和 1（0.2）：命中 1 个错误格
    assert (r.flag_raw, r.flag_cap, r.flag_kept, r.flag_hits) == (4, 2, 2, 1)
    m = compute([r], CFG, ci=False)
    assert m["flag_recall"].value == pytest.approx(1 / 3)
    assert m["flag_precision"].value == pytest.approx(1 / 2)
    assert m["flag_rate"].value == pytest.approx(2 / 100)
    assert m["flag_rate_raw"].value == pytest.approx(4 / 100)


def test_recall_within_cap_no_truncation():
    lab, p = _hundred(uncertain=[2, 99])
    r = score(lab, p)
    assert (r.flag_kept, r.flag_hits) == (2, 1)


def test_flag_everything_cannot_inflate_recall():
    lab, p = _hundred(uncertain=list(range(100)))  # 全部标为不确定、没有 confidence：按编号保留 0、1
    r = score(lab, p)
    assert (r.flag_kept, r.flag_hits) == (2, 2)
    lab, p = _hundred(uncertain=list(range(99, -1, -1)), confidence=[0.5] * 100)  # 置信度相同：按编号
    assert score(lab, p).flag_hits == 2


def test_truncate_tie_break_and_bad_indices():
    import numpy as np
    assert truncate_flags([5, 3, 9], 2, np.array([0, 0, 0, 0.5, 0, 0.5, 0, 0, 0, 0.1])) == [3, 9]  # 9（0.1）最低；3 和 5 并列 0.5，按编号取 3
    lab, p = _hundred(uncertain=[0, 0, -1, 1000, "x", 1])
    r = score(lab, p)
    assert r.flag_raw == 2 and r.flag_hits == 2  # 去重、去掉越界和非法编号
    assert any("非法或越界" in w for w in r.warnings)


def test_tau_fallback_when_no_uncertain_field():
    conf = [0.99] * 100
    conf[2] = 0.3   # < τ = 0.5，是错误格
    conf[70] = 0.4  # < τ，不是错误格
    lab, p = _hundred(confidence=conf)
    r = score(lab, p)
    assert r.flag_source == "tau" and (r.flag_kept, r.flag_hits) == (2, 1)


def test_recall_counts_failed_images_in_denominator():
    lab, p = _hundred(uncertain=[0, 1])
    ok = score(lab, p)
    failed = score(label("f", grid_of("A1", 10, 10)), {"error": {"code": "internal", "message": ""}})
    m = compute([ok, failed], CFG, ci=False)
    assert m["flag_recall"].value == pytest.approx(2 / (3 + 100))  # 失败图 100 个错误格都没标出
    assert m["flag_recall_ok_only"].value == pytest.approx(2 / 3)
    assert m["flag_rate"].value == pytest.approx(2 / 100)  # 标记率、精确率只看成功的图


def test_recall_vacuous_when_no_errors():
    lab, _ = _hundred()
    r = score(lab, pred(grid_of("A1", 10, 10)))
    assert compute([r], CFG, ci=False)["flag_recall"].value is None  # 没有错误格：召回无定义（gate 里按 1 处理）


# --------------------------------------------------------------------------- Hard / Clean / 口径

def test_hard_clean_partition():
    s = label("s", grid_of("A1", 10, 10))
    assert unit(s).hard is False and unit(s).clean and unit(s).size_bucket == "S"
    big = label("big", grid_of("A1", 90, 85))
    u = unit(big)
    assert u.size_bucket == "L" and u.hard and u.hard_reasons == ("size:L",)
    assert unit(label("w", grid_of("A1", 10, 10), tags=["watermark"])).hard_reasons == ("tag:watermark",)
    small = unit(label("sm", grid_of("A1", 10, 10), cell_px=7.5), min_cell=7.5)
    assert small.blurry and small.hard_reasons == ("tag:blurry",) and "blurry" in small.tags
    lowvar = unit(s, lap=99.9)  # < θ = 100
    assert lowvar.blurry and lowvar.hard
    edge = unit(s, min_cell=8.0, lap=100.0)  # 恰好等于阈值：不算 blurry（规则是严格小于）
    assert not edge.blurry and not edge.hard
    human = unit(label("hu", grid_of("A1", 10, 10), tags=["blurry"]))  # 人工标了 blurry，但规则判定不糊
    assert not human.blurry and not human.hard and "blurry" not in human.tags
    assert any("人工 blurry 标签" in w for w in human.warnings)


def test_paged_t4_is_hard_xl():
    lab = label("p", grid_of("A1", 20, 20), type="T4")
    u = unit(lab)
    assert u.size_bucket == "XL" and u.paged and set(u.hard_reasons) == {"size:XL", "type:T4"}


def test_t6_t7_scopes():
    t6 = unit(label("t6", grid_of("A1", 10, 10), type="T6"))
    t7 = unit(label("t7", grid_of("A1", 10, 10), type="T7"))
    assert (t6.in_main, t6.in_flag_scope, t6.clean) == (False, True, False)
    assert (t7.in_main, t7.in_flag_scope, t7.clean) == (False, False, False)


def test_min_cell_px_uses_mean_pitch_and_min_over_axes():
    lab = label("x", grid_of("A1", 4, 2), cell_px=10)
    lab["pages"][0]["y_lines"] = [0, 7, 15]  # 行方向平均格距 7.5 < 列方向 10
    assert min_cell_px_of(lab) == pytest.approx(7.5)


def test_fraction_cap_no_float_error():
    # 0.02 × 1450 = 29.000000000000004（浮点），用 Fraction 计算应得 29
    assert CFG.flag_cap(1450) == 29 and Fraction(CFG.flag_rate) == Fraction(1, 50)
