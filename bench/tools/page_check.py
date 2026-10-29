"""诊断工具：分页图（T4）逐页对比检测到的网格和真值，找出少 / 多的行列在哪一侧、拼接偏移对不对。
不是 04 的正式指标，只用于开发期定位问题。

    python -m bench.tools.page_check --data bench/synth/out/evalT4 --run t4-base [--only-bad]

每页输出一行：真值 cols×rows 与 offset，预测的 cols×rows 与 offset，以及每个方向上
「真值线里没被检测到的」在开头还是结尾（lead / trail，单位：格），和多检测出来的线。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def align(gt: list[float], pr: list[float], tol: float) -> dict:
    """把预测线按最近距离（< tol 像素）配到真值线上。返回首尾漏掉的格数、中间漏的、多出的线数。"""
    matched = []
    used = set()
    for x in pr:
        j = min(range(len(gt)), key=lambda k: abs(gt[k] - x))
        if abs(gt[j] - x) <= tol and j not in used:
            used.add(j)
            matched.append(j)
    extra = len(pr) - len(matched)
    if not matched:
        return {"lead": None, "trail": None, "inner_missing": None, "extra": extra}
    lo, hi = min(matched), max(matched)
    inner = sum(1 for k in range(lo, hi + 1) if k not in used)
    return {"lead": lo, "trail": len(gt) - 1 - hi, "inner_missing": inner, "extra": extra}


def check_unit(label: dict, pred: dict) -> list[dict]:
    rows = []
    ppages = {p["image_index"]: p for p in pred.get("pages", [])} if "error" not in pred else {}
    for g in label["pages"]:
        p = ppages.get(g["image_index"])
        r = {"img": g["image_index"], "gt": f"{g['cols']}x{g['rows']}@{g['col_offset']},{g['row_offset']}"}
        if p is None:
            r["pred"] = "—"
            rows.append(r)
            continue
        r["pred"] = f"{p['cols']}x{p['rows']}@{p.get('col_offset')},{p.get('row_offset')}"
        pitch_x = (g["x_lines"][-1] - g["x_lines"][0]) / g["cols"]
        pitch_y = (g["y_lines"][-1] - g["y_lines"][0]) / g["rows"]
        r["x"] = align(g["x_lines"], p["x_lines"], 0.3 * pitch_x)
        r["y"] = align(g["y_lines"], p["y_lines"], 0.3 * pitch_y)
        r["size_ok"] = (p["cols"], p["rows"]) == (g["cols"], g["rows"])
        r["offset_ok"] = (p.get("col_offset"), p.get("row_offset")) == (g["col_offset"], g["row_offset"])
        r["pitch"] = round(min(pitch_x, pitch_y), 1)
        rows.append(r)
    return rows


def describe(a: dict | None) -> str:
    if a is None:
        return ""
    if a["lead"] is None:
        return "无匹配"
    parts = []
    if a["lead"]:
        parts.append(f"头-{a['lead']}")
    if a["trail"]:
        parts.append(f"尾-{a['trail']}")
    if a["inner_missing"]:
        parts.append(f"中-{a['inner_missing']}")
    if a["extra"]:
        parts.append(f"多{a['extra']}")
    return " ".join(parts) or "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--only-bad", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.data)
    stats: Counter = Counter()
    for lp in sorted((root / "labels").glob("*.json")):
        label = json.loads(lp.read_text(encoding="utf-8"))
        if len(label["pages"]) < 2:
            continue
        pp = root / "preds" / a.run / lp.name
        pred = json.loads(pp.read_text(encoding="utf-8")) if pp.exists() else {"error": {"code": "missing"}}
        rows = check_unit(label, pred)
        unit_ok = "error" not in pred and pred["grid"] == label["grid"]
        if a.only_bad and unit_ok:
            continue
        head = pred["error"]["code"] if "error" in pred else f"{pred['grid']['cols']}x{pred['grid']['rows']}"
        print(f"\n## {lp.stem}  真值 {label['grid']['cols']}x{label['grid']['rows']}  预测 {head}  "
              f"{'OK' if unit_ok else ''} {pred.get('warnings', '')}")
        for r in rows:
            if "x" not in r:
                print(f"  p{r['img']}: 真值 {r['gt']}  预测 {r['pred']}")
                stats["page_missing"] += 1
                continue
            bad = not (r["size_ok"] and r["offset_ok"])
            stats["pages"] += 1
            stats["size_bad"] += not r["size_ok"]
            stats["offset_bad"] += not r["offset_ok"]
            for ax in ("x", "y"):
                d = r[ax]
                if d["lead"]:
                    stats[f"{ax}_lead"] += 1
                if d["trail"]:
                    stats[f"{ax}_trail"] += 1
                if d["inner_missing"]:
                    stats[f"{ax}_inner"] += 1
                if d["extra"]:
                    stats[f"{ax}_extra"] += 1
            if bad or not a.only_bad:
                print(f"  p{r['img']}: 真值 {r['gt']}  预测 {r['pred']}  x[{describe(r['x'])}] y[{describe(r['y'])}]"
                      f"  格距 {r['pitch']}{'' if r['offset_ok'] else '  偏移错'}")
    print("\n汇总：", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
