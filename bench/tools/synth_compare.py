"""开发期回归检查：同一个合成集上两次运行逐张对比（网格对错、图例对错、错误格数）。
不是 04 的正式指标；真实 dev 集用 `python -m bench.eval compare` / `gate`。

先对两次运行各跑一遍 `bench.eval run --split synth`（报告默认写在 <data>/reports/<run>/synth/），再：

    python -m bench.tools.synth_compare --data bench/synth/out/eval60 bench/synth/out/evalT4 --a base --b new

改识别路径后要求「由对变错」为空、网格都对的图错误格数不变差（AGENTS 规则 3 的开发期替代，dev 集到位前用）。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load(data: Path, run: str) -> dict[str, dict]:
    p = data / "reports" / run / "synth" / "per_image.csv"
    if not p.exists():
        raise SystemExit(f"找不到 {p}：先运行 python -m bench.eval run --split synth ... --preds {data / 'preds' / run}")
    return {r["image_id"]: r for r in csv.DictReader(p.open(encoding="utf-8-sig"))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="合成集目录（可以给多个）")
    ap.add_argument("--a", required=True, help="对照运行（改动前）")
    ap.add_argument("--b", required=True, help="新运行（改动后）")
    a = ap.parse_args(argv)
    bad = 0
    for d in map(Path, a.data):
        ra, rb = load(d, a.a), load(d, a.b)
        ok = lambda t: sum(r["status"] == "ok" for r in t.values())  # noqa: E731
        leg = lambda t: sum(r["legend_ok"] == "1" for r in t.values())  # noqa: E731
        worse = [k for k in ra if ra[k]["status"] == "ok" and rb.get(k, {}).get("status") != "ok"]
        better = [k for k in ra if ra[k]["status"] != "ok" and rb.get(k, {}).get("status") == "ok"]
        changed = [(k, int(ra[k]["errors"]), int(rb[k]["errors"])) for k in ra
                   if ra[k]["status"] == "ok" and rb.get(k, {}).get("status") == "ok" and ra[k]["errors"] != rb[k]["errors"]]
        print(f"{d.name}: 网格正确 {ok(ra)} → {ok(rb)} / {len(ra)}；图例完全正确 {leg(ra)} → {leg(rb)}")
        print(f"   由对变错：{worse}")
        print(f"   由错变对：{better}")
        print(f"   网格都对、错误格数变化（改前, 改后）：{changed}")
        bad += len(worse) + sum(1 for _, x, y in changed if y > x)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
