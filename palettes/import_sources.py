"""从上游按固定 commit 拉取色卡原始数据，规范化后写入 palettes/data/<id>.csv（T0.2）。

用法（在仓库根目录）：python -m palettes.import_sources

规范化格式：code,name,r,g,b,series,transparent
- series：色号的字母前缀（A1 → A；H01 → H；80-15179 这类没有字母前缀的为空）
- transparent：上游没有标注时为 0
"""
from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def series_of(code: str) -> str:
    m = re.match(r"^([A-Za-z]+)", code)
    return m.group(1).upper() if m else ""


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pindou-palettes"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def import_beadcolors(spec: dict, upstream: dict) -> list[dict]:
    url = upstream["raw_url"].format(commit=upstream["commit"], file=spec["file"])
    rows = []
    for row in csv.reader(io.StringIO(fetch(url))):
        if not row or not row[0].strip():
            continue
        code, name, r, g, b = (x.strip() for x in row[:5])
        rows.append({"code": code, "name": name, "r": int(r), "g": int(g), "b": int(b),
                     "series": series_of(code), "transparent": 0})
    return rows


def main() -> None:
    cfg = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    DATA.mkdir(exist_ok=True)
    for spec in cfg["palettes"]:
        if spec.get("derived_from_zippland") is not False:
            raise SystemExit(f"{spec['id']}: 只允许引入 derived_from_zippland=false 的色卡（AGENTS 规则 1）")
        upstream = cfg["upstreams"][spec["upstream"]]
        if spec["upstream"] != "beadcolors":
            raise SystemExit(f"未知上游：{spec['upstream']}")
        rows = import_beadcolors(spec, upstream)
        codes = [r["code"] for r in rows]
        if len(codes) != len(set(codes)):
            raise SystemExit(f"{spec['id']}: 上游有重复色号")
        out = DATA / f"{spec['id']}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["code", "name", "r", "g", "b", "series", "transparent"])
            w.writeheader()
            w.writerows(rows)
        print(f"{spec['id']}: {len(rows)} 色 → {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
