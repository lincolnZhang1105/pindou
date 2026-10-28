"""bootstrap 置信区间（按图纸重采样）与配对 bootstrap。

- 重采样单位是「图纸」：每轮从 n 张图里有放回地抽 n 张。
- 统计量一律写成「分子之和 / 分母之和」（比率型），所以微平均（每千颗错误数）和
  各种按图平均的比例（网格正确率、零错误率……分母取 1）用同一个函数。
- 区间用百分位法；某一轮分母之和为 0 时该轮统计量记为 NaN，不参与分位数计算。
- 随机数用 numpy.random.RandomState(seed)：这个旧接口的随机流在 numpy 各版本间保持不变，结果可复现。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=64)
def _indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    rs = np.random.RandomState(seed)
    idx = rs.randint(0, n, size=(n_boot, n))
    idx.setflags(write=False)
    return idx


def boot_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) 的重采样下标矩阵。同样的 (n, n_boot, seed) 总是得到同一个矩阵。"""
    if n <= 0:
        raise ValueError("n 必须为正")
    return _indices(n, n_boot, seed)


def _pct(stats: np.ndarray, confidence: float) -> tuple[float | None, float | None]:
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return None, None
    a = (1.0 - confidence) / 2.0 * 100.0
    lo, hi = np.percentile(stats, [a, 100.0 - a])
    return float(lo), float(hi)


def ratio_ci(num, den, n_boot: int, seed: int, confidence: float,
             scale: float = 1.0) -> tuple[float | None, float | None]:
    """scale × Σnum / Σden 的 bootstrap 百分位置信区间。"""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if num.size == 0 or den.sum() == 0:
        return None, None
    idx = boot_indices(num.size, n_boot, seed)
    sn = num[idx].sum(axis=1)
    sd = den[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.where(sd > 0, scale * sn / np.where(sd > 0, sd, 1.0), np.nan)
    return _pct(st, confidence)


def paired_per_mille_diff(err_a, err_b, beads, n_boot: int, seed: int, confidence: float) -> dict:
    """配对 bootstrap：Δ = 每千颗错误数(A) − 每千颗错误数(B)，在同一批图上按图纸重采样。

    每张图的错误格数差 d_i = err_a_i − err_b_i；同一轮重采样里两组用同样的图，
    Δ* = 1000 × Σd* / Σbeads*。Δ < 0 表示 A 的错误更少。
    """
    a = np.asarray(err_a, dtype=np.float64)
    b = np.asarray(err_b, dtype=np.float64)
    w = np.asarray(beads, dtype=np.float64)
    if not (a.shape == b.shape == w.shape):
        raise ValueError("配对数组长度不一致")
    n = a.size
    out = {"n": int(n), "beads": int(w.sum()), "pm_a": None, "pm_b": None, "diff": None,
           "ci_low": None, "ci_high": None, "p_a_better": None}
    if n == 0 or w.sum() == 0:
        return out
    d = a - b
    out["pm_a"] = 1000.0 * a.sum() / w.sum()
    out["pm_b"] = 1000.0 * b.sum() / w.sum()
    out["diff"] = 1000.0 * d.sum() / w.sum()
    idx = boot_indices(n, n_boot, seed)
    sd = d[idx].sum(axis=1)
    sw = w[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        st = np.where(sw > 0, 1000.0 * sd / np.where(sw > 0, sw, 1.0), np.nan)
    out["ci_low"], out["ci_high"] = _pct(st, confidence)
    fin = st[np.isfinite(st)]
    out["p_a_better"] = float((fin < 0).mean()) if fin.size else None  # 重采样里 A 更好的比例（仅供参考）
    return out
