"""逐图计分：一张图纸的真值 + 预测结果 → ImageResult。指标的精确定义见 bench/eval/README.md。

计分规则摘要：
- 状态 status：ok / grid_wrong / error（预测结果是 {"error": ...}）/ missing（缺文件，视为程序崩溃）/
  invalid（JSON 读不出来、结构不对，同样视为崩溃）。除 ok 以外都是「失败」。
- 失败的图纸（04 §0.2）：错误格数 = 真值豆数（所有非空真值格都按错误计），零错误记「否」，
  非空格准确率记 0，用量清单按「没有输出」计（Σ 真值颗数），不确定标记命中数为 0。
- ok 的图纸逐格比较：色号错误、豆子→空格、空格→豆子都算错误格。
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bench.common import formats as F
from bench.eval.config import Config
from bench.eval.dataset import DatasetError, Unit

FAIL_STATUSES = ("grid_wrong", "error", "missing", "invalid")


@dataclass
class ImageResult:
    image_id: str
    type: str
    size_bucket: str
    tags: list[str]
    hard: bool
    hard_reasons: list[str]
    in_main: bool
    in_flag_scope: bool
    blurry: bool
    min_cell_px: float
    lap_var: float
    label_sha: str
    gt_cols: int
    gt_rows: int
    n_cells: int
    n_beads: int                     # 非空真值格数（总豆数）
    n_empty: int                     # 空真值格数
    status: str
    error_code: str | None = None
    detail: str = ""
    pred_cols: int | None = None
    pred_rows: int | None = None
    errors: int = 0                  # 错误格数（失败图 = n_beads）
    wrong_code: int | None = None    # 以下三项只对 ok 的图有意义
    bead_to_empty: int | None = None
    empty_to_bead: int | None = None
    correct_beads: int = 0
    bom_abs: int = 0                 # Σ|识别颗数 − 真值颗数|，失败图按「没有输出」计 = n_beads
    bom_abs_raw: int | None = None   # 不做失败惩罚的原始值（有可解析 cells 时才有）
    has_legend: bool = False
    legend_ok: bool | None = None    # 只有真值带图例时才有值
    flag_source: str = "none"        # uncertain / tau / none
    flag_raw: int = 0                # 引擎标记的格子数（去重、去掉越界之后）
    flag_cap: int = 0                # 本图上限
    flag_kept: int = 0               # 截断后保留的标记数
    flag_hits: int = 0               # 保留的标记里落在错误格上的个数
    timing_ms: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status != "ok"

    @property
    def grid_ok(self) -> bool:
        return self.status == "ok"

    @property
    def zero_error(self) -> bool:
        return self.status == "ok" and self.errors == 0

    @property
    def clean(self) -> bool:
        return self.in_main and not self.hard

    @property
    def nonempty_acc(self) -> float | None:
        return None if self.n_beads == 0 else self.correct_beads / self.n_beads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ImageResult":
        return cls(**d)


# --------------------------------------------------------------------------- 预测结果读取

def pred_candidates(unit: Unit, manifest: dict[str, dict]) -> list[str]:
    """预测结果文件名候选：真值的 image_id，其次是分页所属的 group id 和各页 image_id。"""
    out = [unit.image_id]
    for pid in unit.page_ids:
        g = (manifest.get(pid) or {}).get("group")
        if g:
            out.append(g)
    out.extend(unit.page_ids)
    seen: dict[str, None] = {}
    for x in out:
        seen.setdefault(x, None)
    return list(seen)


def find_pred(preds_dir: Path, unit: Unit, manifest: dict[str, dict]) -> Path | None:
    hits = [preds_dir / f"{c}.json" for c in pred_candidates(unit, manifest)]
    hits = [h for h in hits if h.exists()]
    if len(hits) > 1:
        raise DatasetError(f"{unit.image_id}: 预测结果文件有歧义 {[h.name for h in hits]}（同一张图纸只能有一个预测结果文件）")
    return hits[0] if hits else None


def load_pred_file(path: Path | None) -> tuple[dict | None, str | None]:
    """返回 (pred, 读取失败的原因)。缺文件时 pred 为 None、原因为 'missing'。"""
    if path is None:
        return None, "missing"
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return None, f"invalid: 读取失败 {type(e).__name__}: {e}"
    if not isinstance(obj, dict):
        return None, "invalid: 顶层不是 JSON 对象"
    return obj, None


# --------------------------------------------------------------------------- 计分

def _int(v: Any) -> int | None:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def _decode_cells(pred: dict, unit: Unit) -> tuple[int, int, list[str | None]]:
    """预测结果 → (cols, rows, 行优先的比较键列表)。结构不对时抛 ValueError。"""
    grid = pred.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("缺少 grid")
    cols, rows = _int(grid.get("cols")), _int(grid.get("rows"))
    if cols is None or rows is None or cols <= 0 or rows <= 0:
        raise ValueError(f"grid 不合法：{grid!r}")
    book = pred.get("codebook")
    cells = pred.get("cells")
    if not isinstance(book, list) or not isinstance(cells, list):
        raise ValueError("缺少 codebook 或 cells")
    keys: list[str] = []
    for i, e in enumerate(book):
        if not isinstance(e, dict) or not isinstance(e.get("code"), str) or not e["code"].strip():
            raise ValueError(f"codebook[{i}] 不合法")
        keys.append(unit.mapper.pred_key(e.get("palette_id"), e["code"]))
    if len(cells) != rows * cols:
        raise ValueError(f"cells 长度 {len(cells)} ≠ {rows}×{cols}")
    out: list[str | None] = []
    for k in cells:
        ki = _int(k)
        if ki is None or ki < 0 or ki > len(keys):
            raise ValueError(f"cells 里有非法索引 {k!r}")
        out.append(None if ki == 0 else keys[ki - 1])
    return cols, rows, out


def _legend_counter(items: Any, key_fn) -> Counter | None:
    """图例列表 → Counter（同一色号出现多次时颗数相加）。格式不对返回 None。"""
    if not isinstance(items, list):
        return None
    c: Counter = Counter()
    for it in items:
        if not isinstance(it, dict) or not isinstance(it.get("code"), str):
            return None
        n = it.get("count")
        if not isinstance(n, (int, float)) or isinstance(n, bool) or n < 0 or int(n) != n:
            return None
        c[key_fn(it)] += int(n)
    return c


def _timing(pred: dict) -> float | None:
    t = pred.get("timing_ms")
    if not isinstance(t, dict):
        return None
    vals = [float(v) for v in t.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return sum(vals) if vals else None


def _flags(pred: dict, n: int, cfg: Config, warnings: list[str]) -> tuple[list[int], str, np.ndarray | None]:
    """被标记的格子（行优先编号，去重、去掉越界），来源，以及 confidence（可用时）。"""
    conf = None
    if pred.get("confidence") is not None:
        try:
            conf = F.pred_confidence(pred).ravel()
            if not np.all(np.isfinite(conf)):
                raise ValueError("含 NaN/Inf")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"confidence 不可用（{e}），截断时只按格子编号")
            conf = None
    unc = pred.get("uncertain")
    if unc is not None:
        if not isinstance(unc, list):
            warnings.append("uncertain 不是数组，按没有标记处理")
            return [], "uncertain", conf
        idx: set[int] = set()
        bad = 0
        for v in unc:
            vi = _int(v)
            if vi is None or not (0 <= vi < n):
                bad += 1
                continue
            idx.add(vi)
        if bad:
            warnings.append(f"uncertain 里有 {bad} 个非法或越界的编号，已忽略")
        if len(idx) != len(unc) - bad:
            warnings.append("uncertain 里有重复编号，已去重")
        return sorted(idx), "uncertain", conf
    if conf is not None:
        return [int(i) for i in np.flatnonzero(conf < cfg.tau)], "tau", conf
    return [], "none", None


def truncate_flags(flagged: list[int], cap: int, conf: np.ndarray | None) -> list[int]:
    """超过上限时按 confidence 升序（相同按编号升序；没有 confidence 只按编号）保留前 cap 个。"""
    if len(flagged) <= cap:
        return sorted(flagged)
    if conf is None:
        order = sorted(flagged)
    else:
        order = sorted(flagged, key=lambda i: (float(conf[i]), i))
    return sorted(order[:cap])


def score_image(unit: Unit, pred: dict | None, cfg: Config, load_problem: str | None = None) -> ImageResult:
    lab = unit.label
    gt = F.label_cells(lab)  # (rows, cols)，元素是 code_key 或 None
    gt_flat = [unit.mapper.gt_key(v) for v in gt.ravel()]
    rows, cols = gt.shape
    n_beads = sum(v is not None for v in gt_flat)
    gt_counts = Counter(v for v in gt_flat if v is not None)
    has_legend = bool(lab.get("legend"))
    r = ImageResult(
        image_id=unit.image_id, type=unit.type, size_bucket=unit.size_bucket, tags=list(unit.tags),
        hard=unit.hard, hard_reasons=list(unit.hard_reasons), in_main=unit.in_main, in_flag_scope=unit.in_flag_scope,
        blurry=unit.blurry, min_cell_px=unit.min_cell_px, lap_var=unit.lap_var, label_sha=unit.label_sha,
        gt_cols=cols, gt_rows=rows, n_cells=rows * cols, n_beads=n_beads, n_empty=rows * cols - n_beads,
        status="ok", has_legend=has_legend, flag_cap=cfg.flag_cap(n_beads),
        legend_ok=False if has_legend else None,  # 缺文件、error JSON 时图例记为不正确
    )

    def fail(status: str, detail: str = "", code: str | None = None) -> ImageResult:
        r.status, r.detail, r.error_code = status, detail, code
        r.errors = n_beads
        r.correct_beads = 0
        r.bom_abs = n_beads  # 失败图一律按「没有输出」计；能解析出 cells 时，原始值另存在 bom_abs_raw
        r.flag_kept = r.flag_hits = 0
        return r

    # ---- 读取失败 / error JSON
    if pred is None:
        if load_problem == "missing" or load_problem is None:
            return fail("missing", "缺少预测结果文件（视为程序崩溃）")
        return fail("invalid", load_problem)
    if F.is_error(pred):
        err = pred.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        msg = err.get("message", "") if isinstance(err, dict) else str(err)
        return fail("error", str(msg), str(code) if code is not None else None)

    r.timing_ms = _timing(pred)

    # ---- 图例（与网格无关：网格错了也照样评图例；结构坏掉的预测结果记为不正确）
    def legend_eval() -> None:
        if not has_legend:
            return
        want = _legend_counter(lab["legend"], lambda it: unit.mapper.raw_key(it["code"]))
        got = _legend_counter(pred.get("legend"), lambda it: unit.mapper.pred_key(it.get("palette_id"), it["code"]))
        if got is None and pred.get("legend") is not None:
            r.warnings.append("预测结果的 legend 格式不对，按不正确计")
        r.legend_ok = got is not None and want is not None and got == want and set(got) == set(want)

    try:
        pcols, prows, pflat = _decode_cells(pred, unit)
    except ValueError as e:
        legend_eval()
        return fail("invalid", f"预测结果结构不对：{e}")
    r.pred_cols, r.pred_rows = pcols, prows
    legend_eval()
    pred_counts = Counter(v for v in pflat if v is not None)
    r.bom_abs_raw = sum(abs(pred_counts.get(k, 0) - gt_counts.get(k, 0)) for k in set(pred_counts) | set(gt_counts))
    if (pcols, prows) != (cols, rows):
        return fail("grid_wrong", f"网格 {pcols}×{prows}，真值 {cols}×{rows}（列×行）")

    # ---- 逐格比较
    wrong = b2e = e2b = correct = 0
    err_mask = np.zeros(rows * cols, dtype=bool)
    for i, (g, p) in enumerate(zip(gt_flat, pflat)):
        if g is None:
            if p is not None:
                e2b += 1
                err_mask[i] = True
        elif p is None:
            b2e += 1
            err_mask[i] = True
        elif p != g:
            wrong += 1
            err_mask[i] = True
        else:
            correct += 1
    r.wrong_code, r.bead_to_empty, r.empty_to_bead = wrong, b2e, e2b
    r.errors = wrong + b2e + e2b
    r.correct_beads = correct
    r.bom_abs = r.bom_abs_raw

    # ---- 不确定标记（上限内）
    flagged, src, conf = _flags(pred, rows * cols, cfg, r.warnings)
    r.flag_source = src
    r.flag_raw = len(flagged)
    kept = truncate_flags(flagged, r.flag_cap, conf)
    r.flag_kept = len(kept)
    r.flag_hits = int(err_mask[kept].sum()) if kept else 0
    return r


def score_units(units: list[Unit], preds_dir: Path, cfg: Config,
                manifest: dict[str, dict]) -> tuple[list[ImageResult], list[str], str]:
    """对一个预测结果目录计分。返回 (逐图结果, 警告, 预测结果指纹)。"""
    results: list[ImageResult] = []
    warns: list[str] = []
    used: set[Path] = set()
    fp = hashlib.sha256()
    for u in units:
        path = find_pred(preds_dir, u, manifest)
        pred, problem = load_pred_file(path)
        if path is not None:
            used.add(path.resolve())
            fp.update(f"{u.image_id}\0{F.sha256_file(path)}\n".encode("utf-8"))
        else:
            fp.update(f"{u.image_id}\0missing\n".encode("utf-8"))
        res = score_image(u, pred, cfg, problem)
        res.warnings = list(u.warnings) + res.warnings
        results.append(res)
    missing = [r.image_id for r in results if r.status == "missing"]
    if missing:
        warns.append(f"{len(missing)} 张图没有预测结果文件，已按失败计分：{missing[:10]}{' …' if len(missing) > 10 else ''}")
    extra = sorted(p.name for p in preds_dir.glob("*.json") if p.resolve() not in used)
    if extra:
        warns.append(f"预测结果目录里有 {len(extra)} 个文件没有对应本 split 的图纸（命名不一致？）：{extra[:10]}")
    return results, warns, fp.hexdigest()
