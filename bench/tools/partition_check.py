"""诊断工具：不看色号文字、只看「格子分组」对不对（OCR 还没接上之前，用来单独衡量网格和颜色分类）。

对每张网格尺寸正确的图，把预测的每个类别按多数票映射到一个真值色号，统计分错的格子数
（含空格判断错误）。这不是 04 的正式指标，只用于开发期定位问题。

    python -m bench.tools.partition_check --data bench/synth/out/eval60 --run v0
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from bench.common import formats as F


def check(label: dict, pred: dict) -> dict:
    gt = F.label_cells(label)
    rows, cols = gt.shape
    out = {"grid_ok": False, "gt": f"{cols}x{rows}", "beads": int(sum(v is not None for v in gt.ravel()))}
    if "error" in pred:
        out["error"] = pred["error"]["code"]
        return out
    out["pred"] = f"{pred['grid']['cols']}x{pred['grid']['rows']}"
    if pred["grid"]["cols"] != cols or pred["grid"]["rows"] != rows:
        return out
    out["grid_ok"] = True
    cells = pred["cells"]
    votes: dict[int, Counter] = defaultdict(Counter)
    for i, k in enumerate(cells):
        votes[k][gt.ravel()[i]] += 1
    mapping = {k: (None if k == 0 else c.most_common(1)[0][0]) for k, c in votes.items()}
    errs = sum(1 for i, k in enumerate(cells) if mapping[k] != gt.ravel()[i])
    out["partition_errors"] = errs
    out["uncertain"] = len(pred.get("uncertain", []))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="含 labels/ 和 preds/ 的目录")
    ap.add_argument("--run", required=True)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows = []
    for lp in sorted((a.data / "labels").glob("*.json")):
        pp = a.data / "preds" / a.run / lp.name
        if not pp.exists():
            continue
        r = check(F.load_json(lp), F.load_json(pp))
        r["id"] = lp.stem
        man = a.data / "manifest.jsonl"
        if man.exists():
            m = F.load_manifest(man).get(lp.stem, {})
            r["tags"] = ",".join(m.get("tags") or [])
        rows.append(r)
    ok = [r for r in rows if r["grid_ok"]]
    beads = sum(r["beads"] for r in ok)
    errs = sum(r["partition_errors"] for r in ok)
    print(f"{len(rows)} 张：网格正确 {len(ok)}，报错 {sum('error' in r for r in rows)}，"
          f"网格错 {len(rows) - len(ok) - sum('error' in r for r in rows)}")
    if ok:
        print(f"网格正确的图：分组错误 {errs} / {beads} 颗（{1000 * errs / max(1, beads):.2f}‰），"
              f"零错误 {sum(r['partition_errors'] == 0 for r in ok)}/{len(ok)}")
    for r in rows:
        if not r["grid_ok"] or r.get("partition_errors"):
            print(f"  {r['id']}: gt {r['gt']} pred {r.get('pred', '-')} err {r.get('error', '')} "
                  f"part_err {r.get('partition_errors', '-')} tags {r.get('tags', '')}")
    if a.csv:
        with a.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
