"""标注任务、草稿、真值的读写和导出校验（服务端逻辑，与 HTTP 无关，便于测试）。

- 任务 = 标注单元（`bench/common/dataset.py`）：普通图纸 key = image_id，分页图 key = group id；
- 草稿：`$PINDOU_DATA_DIR/labels/drafts/<key>.json`，格式是真值的超集（多了取色点、锚点、可疑格等编辑状态）；
- 导出：用 `validate_label` 校验，按 04 §2.2 格式写 `labels/<key>.json`。图例校验不通过时拒绝导出；
- 复核：XL 档和分页图要第二个人复核（04 §2.2），复核通过时写 `reviewed_by`。
"""
from __future__ import annotations

import re
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path

from bench.common import dataset as ds
from bench.common import formats

PALETTE_FILES = (formats.REPO / "bench" / "palettes_unverified" / "build" / "palettes_bench.json",
                 formats.REPO / "palettes" / "build" / "palettes.json")
_CODE_OK = re.compile(r"^(?:[A-Z0-9]+:)?[A-Z0-9][A-Z0-9-]*$")
STATUS_TEXT = {"unlabeled": "未标", "draft": "草稿", "legend_fail": "图例校验失败", "ready": "草稿（可导出）",
               "done": "完成", "needs_review": "待复核", "invalid": "真值无效"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def norm_code(code) -> str | None:
    """格子 / 图例里的色号：None、空串 → None；其余按 code_key 规范化（A01 → A1，zg:zg01 → ZG:ZG1）。"""
    if code is None:
        return None
    s = str(code).strip()
    if not s:
        return None
    return formats.code_key(s)


def legend_diff(cells: list[list], legend: list[dict]) -> list[dict]:
    """图例颗数与格子实际颗数的差异（和 validate_label 的图例校验同一套规则）。"""
    actual = Counter(norm_code(c) for row in cells for c in row)
    actual.pop(None, None)
    expected: Counter = Counter()
    for item in legend:
        code = norm_code(item.get("code"))
        if code is None:
            continue
        try:
            expected[code] += int(item.get("count"))
        except (TypeError, ValueError):
            expected[code] += 0
    return [{"code": c, "expected": expected.get(c, 0), "actual": actual.get(c, 0)}
            for c in sorted(set(actual) | set(expected)) if actual.get(c, 0) != expected.get(c, 0)]


def needs_review(type_: str | None, size_bucket: str | None, n_images: int, n_pages: int) -> bool:
    """04 §2.2 第 4 步：XL 档和分页图要第二个人复核。"""
    return size_bucket == "XL" or type_ == "T4" or n_images > 1 or n_pages > 1


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.RLock()
        self._mcache: tuple | None = None
        self._palettes = None

    # ------------------------------------------------------------------ manifest / 划分
    def manifest(self) -> dict[str, dict]:
        p = formats.MANIFEST
        mt = p.stat().st_mtime_ns if p.exists() else None
        if self._mcache is None or self._mcache[0] != (str(p), mt):
            self._mcache = ((str(p), mt), formats.load_manifest(p))
        return self._mcache[1]

    def units(self) -> dict[str, list[dict]]:
        return ds.build_units(self.manifest())

    def split_of(self) -> dict[str, str]:
        out = {}
        m = self.manifest()
        for name in ("dev", "test"):
            for iid in formats.read_split(name):
                if iid in m:
                    out[ds.unit_key(m[iid])] = name
        return out

    def pages_of(self, key: str) -> list[dict] | None:
        if not ds.valid_id(key):
            return None
        return self.units().get(key)

    def palettes(self) -> dict:
        if self._palettes is None:
            for p in PALETTE_FILES:
                if p.exists():
                    data = formats.load_json(p)
                    self._palettes = {"source": str(p.relative_to(formats.REPO)), "palettes": [
                        {"id": pal["id"], "colors": [[c["code"], *c["rgb"]] for c in pal["colors"]]}
                        for pal in data.get("palettes", [])]}
                    break
            else:
                self._palettes = {"source": None, "palettes": []}
        return self._palettes

    def palette_codes(self, pid: str | None) -> set[str] | None:
        if not pid:
            return None
        for pal in self.palettes()["palettes"]:
            if pal["id"].upper() == pid.upper():
                return {formats.normalize_code(c[0]) for c in pal["colors"]}
        return None

    # ------------------------------------------------------------------ 草稿 / 真值
    def load_draft(self, key: str) -> dict | None:
        p = ds.draft_path(key, self.root)
        return formats.load_json(p) if p.is_file() else None

    def load_label(self, key: str) -> dict | None:
        p = ds.label_path(key, self.root)
        return formats.load_json(p) if p.is_file() else None

    def save_draft(self, key: str, draft: dict) -> dict:
        if not isinstance(draft, dict):
            raise ValueError("草稿必须是 JSON 对象")
        with self._lock:
            old = self.load_draft(key) or {}
            draft = dict(draft)
            draft["key"] = key
            draft["rev"] = int(old.get("rev", 0)) + 1
            draft["updated_at"] = now_iso()
            if "exported_rev" in old and "exported_rev" not in draft:
                draft["exported_rev"] = old["exported_rev"]
            formats.save_json(ds.draft_path(key, self.root), draft, indent=None)
        return draft

    def status(self, key: str, pages: list[dict]) -> dict:
        lab_p, dr_p = ds.label_path(key, self.root), ds.draft_path(key, self.root)
        draft = None
        if dr_p.is_file():
            try:
                draft = formats.load_json(dr_p)
            except ValueError:
                draft = None
        out = {"code": "unlabeled", "draft_newer": False, "legend_diff": None, "reviewed_by": None,
               "annotator": None}
        if lab_p.is_file():
            try:
                lab = formats.load_json(lab_p)
                errs = formats.validate_label(lab)
            except (ValueError, KeyError, TypeError) as e:
                lab, errs = {}, [str(e)]
            out["annotator"] = lab.get("annotator")
            out["reviewed_by"] = lab.get("reviewed_by")
            if errs:
                out["code"], out["errors"] = "invalid", errs[:5]
            elif lab.get("needs_review") and not lab.get("reviewed_by"):
                out["code"] = "needs_review"
            else:
                out["code"] = "done"
            if draft is not None:
                out["draft_newer"] = int(draft.get("rev", 0)) > int(draft.get("exported_rev", -1))
        elif draft is not None:
            out["annotator"] = draft.get("annotator")
            legend = draft.get("legend") or []
            cells = draft.get("cells") or []
            if draft.get("no_legend") or not legend:
                out["code"] = "draft" if not draft.get("no_legend") else "ready"
            else:
                diff = legend_diff(cells, legend)
                out["legend_diff"] = len(diff)
                out["code"] = "legend_fail" if diff else "ready"
        out["text"] = STATUS_TEXT[out["code"]] + ("（有未导出的修改）" if out["draft_newer"] else "")
        return out

    def task_list(self) -> list[dict]:
        split = self.split_of()
        out = []
        for key, pages in self.units().items():
            a = ds.unit_attrs(pages)
            st = self.status(key, pages)
            out.append({"key": key, "image_ids": [p["image_id"] for p in pages], "split": split.get(key),
                        "type": a["type"], "size_bucket": a["size_bucket"], "tags": a["tags"],
                        "palette_hint": a["palette_hint"], "n_images": a["n_images"],
                        "needs_review": needs_review(a["type"], a["size_bucket"], a["n_images"], 1),
                        "status": st})
        return out

    def task(self, key: str) -> dict | None:
        pages = self.pages_of(key)
        if pages is None:
            return None
        a = ds.unit_attrs(pages)
        first = pages[0]
        images = []
        for p in pages:
            readable = ds.readable_image_path(p, self.root)
            images.append({"image_id": p["image_id"], "file": p["file"], "width": p.get("width"),
                           "height": p.get("height"), "format": p.get("format"),
                           "exif_orientation": p.get("exif_orientation"),
                           "url": f"/img/{p['image_id']}", "readable": readable.is_file()})
        draft, source = self.load_draft(key), "draft"
        label = self.load_label(key)
        if draft is None and label is not None:
            draft, source = label_to_draft(label), "label"
        elif draft is None:
            source = None
        return {
            "key": key, "split": self.split_of().get(key), "images": images,
            "meta": {"type": a["type"], "size_bucket": a["size_bucket"], "palette_hint": a["palette_hint"],
                     "tags": a["tags"], "grid": first.get("grid"), "notes": first.get("notes") or "",
                     "source": {k: first.get(k) for k in ("platform", "note_id", "author", "url", "saved_at",
                                                         "source_path")}},
            "needs_review": needs_review(a["type"], a["size_bucket"], a["n_images"], 1),
            "draft": draft, "draft_source": source,
            "label": None if label is None else {k: label.get(k) for k in (
                "annotator", "labeled_at", "reviewed_by", "reviewed_at", "needs_review", "size_bucket")},
            "status": self.status(key, pages),
        }

    # ------------------------------------------------------------------ 导出
    def build_label(self, key: str, draft: dict) -> tuple[dict | None, list[str], list[str]]:
        """草稿 → 04 §2.2 真值。返回 (label, 错误, 警告)；有错误时 label 为 None。"""
        pages_m = self.pages_of(key)
        if pages_m is None:
            return None, [f"未知任务 {key}"], []
        errs: list[str] = []
        warns: list[str] = []
        a = ds.unit_attrs(pages_m)
        first = pages_m[0]
        grid = draft.get("grid") or {}
        try:
            cols, rows = int(grid.get("cols")), int(grid.get("rows"))
        except (TypeError, ValueError):
            return None, ["请先填写网格的列数和行数"], []
        if cols <= 0 or rows <= 0:
            return None, ["网格的列数和行数必须是正整数"], []
        raw_cells = draft.get("cells") or []
        if len(raw_cells) != rows or any(not isinstance(r, list) or len(r) != cols for r in raw_cells):
            return None, [f"cells 尺寸应为 {rows}×{cols}"], []
        cells = [[norm_code(c) for c in row] for row in raw_cells]
        bad = sorted({c for row in cells for c in row if c is not None and not _CODE_OK.match(c)})
        if bad:
            errs.append(f"格子里有不合法的色号：{bad[:10]}")
        pages = []
        covered = [[False] * cols for _ in range(rows)]
        for i, p in enumerate(draft.get("pages") or []):
            try:
                pc, pr = int(p["cols"]), int(p["rows"])
                xl = [round(float(v), 2) for v in p["x_lines"]]
                yl = [round(float(v), 2) for v in p["y_lines"]]
                co, ro = int(p.get("col_offset", 0)), int(p.get("row_offset", 0))
                ii = int(p.get("image_index", 0))
            except (KeyError, TypeError, ValueError):
                errs.append(f"第 {i + 1} 页的网格不完整")
                continue
            if any(b <= a_ for a_, b in zip(xl, xl[1:])) or any(b <= a_ for a_, b in zip(yl, yl[1:])):
                errs.append(f"第 {i + 1} 页的网格线必须严格递增")
            if co < 0 or ro < 0:
                errs.append(f"第 {i + 1} 页的偏移不能为负")
            for r in range(max(0, ro), min(rows, ro + pr)):
                for c in range(max(0, co), min(cols, co + pc)):
                    covered[r][c] = True
            pages.append({"image_index": ii, "bbox": [xl[0], yl[0], xl[-1], yl[-1]] if xl and yl else [0, 0, 0, 0],
                          "x_lines": xl, "y_lines": yl, "col_offset": co, "row_offset": ro, "cols": pc, "rows": pr})
        if not pages:
            errs.append("至少要有一页网格")
        uncovered = sum(1 for row in covered for v in row if not v)
        if pages and uncovered:
            warns.append(f"有 {uncovered} 格不在任何一页的网格里")
        legend: list[dict] = []
        if draft.get("no_legend"):
            if draft.get("legend"):
                warns.append("已勾选「无图例」，导出时忽略图例")
        else:
            seen = set()
            for item in draft.get("legend") or []:
                code = norm_code(item.get("code"))
                cnt = item.get("count")
                if code is None:
                    if cnt not in (None, ""):
                        errs.append("图例里有没填色号的行")
                    continue
                try:
                    cnt = int(cnt)
                    if cnt < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    errs.append(f"图例 {code} 的颗数没填或不是非负整数")
                    continue
                if code in seen:
                    errs.append(f"图例里 {code} 出现了两次")
                seen.add(code)
                legend.append({"code": code, "count": cnt})
            if not legend:
                errs.append("图例为空：请按图上的图例录入色号和颗数；图纸本身没有图例时勾选「无图例」")
        typ = (draft.get("type") or a["type"] or "").upper()
        n_images = len(pages_m)
        paged = typ == "T4" or n_images > 1 or len(pages) > 1
        bucket = formats.size_bucket_of(cols, rows, paged=paged)
        if a["size_bucket"] and bucket != a["size_bucket"]:
            warns.append(f"尺寸档按真值是 {bucket}，manifest 里预估的是 {a['size_bucket']}")
        hint = (draft.get("palette_hint") or a["palette_hint"] or "").upper() or None
        known = self.palette_codes(hint)
        if known is not None:
            unknown = sorted({c for row in cells for c in row if c and ":" not in c and c not in known})
            if unknown:
                warns.append(f"这些色号不在 {hint} 色卡里：{unknown[:10]}（混用其他品牌请写成 品牌:色号）")
        n_flags = len(draft.get("flags") or [])
        if n_flags:
            warns.append(f"还有 {n_flags} 个可疑格没有处理")
        label = {
            "image_id": key,
            "source": {k: first.get(k) for k in ("platform", "note_id", "author", "saved_at", "source_path")},
            "type": typ, "size_bucket": bucket, "palette_hint": hint,
            "images": [{"file": p["file"], "sha256": p["sha256"]} for p in pages_m],
            "grid": {"cols": cols, "rows": rows},
            "pages": pages, "legend": legend, "cells": cells,
            "tags": [t for t in (draft.get("tags") or []) if isinstance(t, str) and t.strip()],
        }
        errs.extend(e for e in formats.validate_label(label) if e not in errs)
        if errs:
            return None, errs, warns
        label["needs_review"] = needs_review(typ, bucket, n_images, len(pages))
        return label, [], warns

    def export(self, key: str, draft: dict, user: str | None, *, review: bool = False) -> dict:
        """导出真值。review=True 表示「复核通过」：user 记为 reviewed_by（必须和标注人不同）。"""
        user = (user or "").strip()
        if not user:
            return {"ok": False, "errors": ["请先在右上角填写你的名字（标注人 / 复核人）"], "warnings": []}
        with self._lock:
            label, errs, warns = self.build_label(key, draft)
            if label is None:
                return {"ok": False, "errors": errs, "warnings": warns}
            old = self.load_label(key) or {}
            if review:
                annotator = old.get("annotator") or draft.get("annotator")
                if not annotator:
                    return {"ok": False, "errors": ["还没有标注人导出过，不能复核"], "warnings": warns}
                if annotator == user:
                    return {"ok": False, "errors": [f"复核人必须是另一个人（标注人是 {annotator}）"],
                            "warnings": warns}
                label["annotator"] = annotator
                label["labeled_at"] = old.get("labeled_at") or now_iso()
                label["reviewed_by"] = user
                label["reviewed_at"] = now_iso()
            else:
                label["annotator"] = user
                label["labeled_at"] = now_iso()
                # 标注人重新导出（内容可能改了）→ 需要重新复核
                label["reviewed_by"] = None
                label["reviewed_at"] = None
            formats.save_json(ds.label_path(key, self.root), label)
            d = dict(draft)
            d["annotator"] = label["annotator"]
            saved = self.save_draft(key, d)
            saved["exported_rev"] = saved["rev"]
            formats.save_json(ds.draft_path(key, self.root), saved, indent=None)
        return {"ok": True, "errors": [], "warnings": warns, "path": f"labels/{key}.json",
                "needs_review": label["needs_review"], "reviewed_by": label["reviewed_by"]}

    # ------------------------------------------------------------------ 预测结果（初稿）
    def pred_runs(self, key: str) -> list[dict]:
        pages = self.pages_of(key)
        if pages is None:
            return []
        d = ds.preds_dir(self.root)
        if not d.is_dir():
            return []
        names = [f"{key}.json", f"{pages[0]['image_id']}.json"]
        out = []
        for run in sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run.is_dir() or not ds.valid_id(run.name):
                continue
            for n in names:
                f = run / n
                if f.is_file():
                    out.append({"run": run.name, "file": n, "mtime": datetime.fromtimestamp(
                        f.stat().st_mtime).astimezone().isoformat(timespec="seconds")})
                    break
        return out

    def pred_draft(self, key: str, run: str) -> dict:
        """把预测结果（RecognitionResult）转成草稿的一部分：grid、pages、cells、legend、flags。"""
        pages = self.pages_of(key)
        if pages is None or not ds.valid_id(run):
            raise KeyError(key)
        for n in (f"{key}.json", f"{pages[0]['image_id']}.json"):
            f = ds.preds_dir(self.root) / run / n
            if f.is_file():
                break
        else:
            raise KeyError(f"{run}/{key}")
        pred = formats.load_json(f)
        if formats.is_error(pred):
            e = pred["error"]
            return {"error": f"{e.get('code')}: {e.get('message')}"}
        cells = formats.pred_to_label_cells(pred)
        rows, cols = pred["grid"]["rows"], pred["grid"]["cols"]
        out_pages = []
        for p in pred.get("pages") or []:
            pc, pr = p.get("cols", cols), p.get("rows", rows)
            xl, yl = p.get("x_lines") or [], p.get("y_lines") or []
            if len(xl) != pc + 1 or len(yl) != pr + 1:
                bb = p.get("bbox") or [0, 0, 0, 0]
                if bb[2] > bb[0] and bb[3] > bb[1]:
                    xl = [bb[0] + (bb[2] - bb[0]) * i / pc for i in range(pc + 1)]
                    yl = [bb[1] + (bb[3] - bb[1]) * i / pr for i in range(pr + 1)]
                else:
                    xl, yl = [], []
            out_pages.append({"image_index": p.get("image_index", 0), "cols": pc, "rows": pr,
                              "col_offset": p.get("col_offset", 0), "row_offset": p.get("row_offset", 0),
                              "x_lines": xl, "y_lines": yl})
        flags = []
        for idx in pred.get("uncertain") or []:
            if 0 <= idx < rows * cols:
                flags.append([idx // cols, idx % cols, "uncertain"])
        legend = [{"code": formats.normalize_code(e["code"]), "count": e.get("count")}
                  for e in pred.get("legend") or [] if e.get("code")]
        guess = (pred.get("palette_guess") or {}).get("palette_id")
        return {"grid": {"cols": cols, "rows": rows}, "pages": out_pages, "cells": cells, "legend": legend,
                "flags": flags, "palette_guess": guess, "engine_version": pred.get("engine_version")}


def label_to_draft(label: dict) -> dict:
    """已导出的真值 → 草稿（没有草稿、只有真值时继续编辑用）。"""
    pages = []
    for p in label.get("pages", []):
        q = dict(p)
        q.pop("bbox", None)
        q["x_anchors"] = list(range(len(p.get("x_lines", []))))
        q["y_anchors"] = list(range(len(p.get("y_lines", []))))
        pages.append(q)
    return {"type": label.get("type"), "palette_hint": label.get("palette_hint"), "tags": label.get("tags", []),
            "grid": label.get("grid"), "pages": pages, "cells": label.get("cells"),
            "legend": [dict(x) for x in label.get("legend", [])], "no_legend": not label.get("legend"),
            "flags": [], "annotator": label.get("annotator")}
