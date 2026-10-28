"""竞品结果的导入与计分（04 §4.2）。

- Tier A：每个竞品条目一份 CSV（长表，一行一个色号），记录网格尺寸和用量清单，用来确定「最强竞品」。
  格式见 bench/eval/README.md「竞品 Tier A CSV」。
- Tier B：最强竞品的格子级结果，用我们自己的解析器读取竞品导出的干净 PNG 后，存成 RecognitionResult
  （和我方预测结果同一格式），直接复用 scoring.score_image 计分。
"""
from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from bench.common import formats as F
from bench.eval.dataset import Unit
from bench.eval.scoring import ImageResult, find_pred, load_pred_file, score_image

TIER_A_COLUMNS = ("app", "app_version", "platform", "import_path", "tested_at",
                  "image_id", "status", "cols", "rows", "code", "count")
TIER_A_STATUSES = ("ok", "error")


class CompetitorDataError(ValueError):
    pass


@dataclass
class TierAImage:
    image_id: str
    status: str
    cols: int | None
    rows: int | None
    counts: dict[str, int]  # 原始色号写法（经 code_key 规范化） → 颗数


@dataclass
class TierAEntry:
    entry_id: str            # app/platform/import_path
    app: str
    app_version: str
    platform: str
    import_path: str
    tested_at: str
    files: list[str] = field(default_factory=list)
    images: dict[str, TierAImage] = field(default_factory=dict)


def entry_id_of(app: str, platform: str, import_path: str) -> str:
    return f"{app}/{platform}/{import_path}"


def _pos_int(s: str, what: str, where: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise CompetitorDataError(f"{where}: {what} 不是整数：{s!r}") from None
    if v <= 0:
        raise CompetitorDataError(f"{where}: {what} 必须为正：{v}")
    return v


def parse_tier_a(paths: list[Path]) -> dict[str, TierAEntry]:
    entries: dict[str, TierAEntry] = {}
    for path in sorted(Path(p) for p in paths):
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in TIER_A_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise CompetitorDataError(f"{path}: 缺少列 {missing}")
            for ln, row in enumerate(reader, start=2):
                where = f"{path.name}:{ln}"
                row = {k: (v or "").strip() for k, v in row.items() if k is not None}
                if not any(row.values()):
                    continue
                for c in ("app", "platform", "import_path", "app_version", "tested_at", "image_id", "status"):
                    if not row[c]:
                        raise CompetitorDataError(f"{where}: {c} 不能为空")
                eid = entry_id_of(row["app"], row["platform"], row["import_path"])
                e = entries.get(eid)
                if e is None:
                    e = entries[eid] = TierAEntry(eid, row["app"], row["app_version"], row["platform"],
                                                  row["import_path"], row["tested_at"])
                if (row["app_version"], row["tested_at"]) != (e.app_version, e.tested_at):
                    raise CompetitorDataError(f"{where}: 同一竞品条目 {eid} 的 app_version / tested_at 不一致"
                                              "（不同版本请分开记录为不同的 platform 或 import_path）")
                if path.name not in e.files:
                    e.files.append(path.name)
                st = row["status"]
                if st not in TIER_A_STATUSES:
                    raise CompetitorDataError(f"{where}: status 只能是 {TIER_A_STATUSES}")
                iid = row["image_id"]
                if st == "error":
                    cols = rows = None
                    if row["code"] or row["count"]:
                        raise CompetitorDataError(f"{where}: status=error 的行不能有 code / count")
                else:
                    cols = _pos_int(row["cols"], "cols", where)
                    rows = _pos_int(row["rows"], "rows", where)
                img = e.images.get(iid)
                if img is None:
                    img = e.images[iid] = TierAImage(iid, st, cols, rows, {})
                elif (img.status, img.cols, img.rows) != (st, cols, rows):
                    raise CompetitorDataError(f"{where}: {eid} 的 {iid} 各行的 status / cols / rows 不一致")
                if st == "ok":
                    if bool(row["code"]) != bool(row["count"]):
                        raise CompetitorDataError(f"{where}: code 和 count 要么都填，要么都空（都空表示这张图没有豆子）")
                    if row["code"]:
                        try:
                            n = int(row["count"])
                        except ValueError:
                            raise CompetitorDataError(f"{where}: count 不是整数：{row['count']!r}") from None
                        if n < 0:
                            raise CompetitorDataError(f"{where}: count 不能为负")
                        k = F.code_key(row["code"])
                        if k in img.counts:
                            raise CompetitorDataError(f"{where}: {eid} 的 {iid} 里色号 {row['code']} 重复")
                        img.counts[k] = n
    return entries


@dataclass
class TierAResult:
    image_id: str
    status: str          # ok / grid_wrong / error
    n_beads: int
    bom_abs: int         # 失败（error、网格错）按「没有输出」计 = n_beads
    bom_abs_raw: int     # 不惩罚网格错（error 仍为 n_beads）
    grid_ok: bool


def gt_counts(unit: Unit) -> Counter:
    return Counter(k for k in (unit.mapper.gt_key(v) for v in F.label_cells(unit.label).ravel()) if k is not None)


def score_tier_a(entry: TierAEntry, units: list[Unit]) -> dict[str, TierAResult]:
    """对 units 中被该条目覆盖的图计分。返回 image_id → TierAResult。"""
    out: dict[str, TierAResult] = {}
    for u in units:
        ti = entry.images.get(u.image_id)
        if ti is None:
            continue
        g = gt_counts(u)
        nb = sum(g.values())
        if ti.status == "error":
            out[u.image_id] = TierAResult(u.image_id, "error", nb, nb, nb, False)
            continue
        p: Counter = Counter()
        for code, n in ti.counts.items():
            p[u.mapper.gt_key(code)] += n
        raw = sum(abs(p.get(k, 0) - g.get(k, 0)) for k in set(p) | set(g))
        ok = (ti.cols, ti.rows) == (u.label["grid"]["cols"], u.label["grid"]["rows"])
        out[u.image_id] = TierAResult(u.image_id, "ok" if ok else "grid_wrong", nb, raw if ok else nb, raw, ok)
    return out


def unknown_images(entry: TierAEntry, all_ids: set[str]) -> list[str]:
    return sorted(i for i in entry.images if i not in all_ids)


def score_tier_b(units: list[Unit], preds_dir: Path, cfg, manifest: dict[str, dict]) -> tuple[list[ImageResult], list[str]]:
    """Tier B：竞品结果目录里缺文件的图视为「没有测」（不是竞品失败），单独返回；error JSON 才算竞品失败。"""
    results: list[ImageResult] = []
    not_covered: list[str] = []
    for u in units:
        path = find_pred(preds_dir, u, manifest)
        if path is None:
            not_covered.append(u.image_id)
            continue
        pred, problem = load_pred_file(path)
        results.append(score_image(u, pred, cfg, problem))
    return results, not_covered
