"""导入来源存疑的色卡，只供 bench 内部使用（见同目录 sources.json 的说明）。

用法：python -m bench.palettes_unverified.import_unverified
产物：bench/palettes_unverified/data/<id>.csv，以及合并了 palettes/ 正式色卡的
      bench/palettes_unverified/build/palettes_bench.json（benchmark 跑 CLI 时用）。
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HB_RAW = "https://raw.githubusercontent.com/HansBug/pindou-color-data/{commit}/{dir}/colors.json"
BC_RAW = "https://raw.githubusercontent.com/maxcleme/beadcolors/{commit}/raw/{file}"


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "pindou-bench"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def norm(code: str) -> str:
    m = re.match(r"^([A-Za-z]+)0*(\d+)$", code.strip())
    return f"{m.group(1).upper()}{int(m.group(2))}" if m else code.strip().upper()


def series_of(code: str) -> str:
    m = re.match(r"^([A-Za-z]+)", code)
    return m.group(1).upper() if m else ""


def main() -> None:
    cfg = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    ups = cfg["upstreams"]
    (HERE / "data").mkdir(exist_ok=True)
    for spec in cfg["palettes"]:
        up = ups[spec["upstream"]]
        rows = []
        if spec["upstream"] == "beadcolors-mard":
            for row in csv.reader(io.StringIO(fetch(BC_RAW.format(commit=up["commit"], file=spec["file"])))):
                if row and row[0].strip():
                    rows.append({"code": row[0].strip(), "name": "", "r": int(row[2]), "g": int(row[3]),
                                 "b": int(row[4]), "series": series_of(row[0]), "transparent": 0})
            if "subset_codes_from" in spec:
                sub = json.loads(fetch(HB_RAW.format(commit=ups["hansbug"]["commit"], dir=spec["subset_codes_from"])))
                keep = {norm(c["code"]) for c in sub["colors"]}
                rows = [r for r in rows if norm(r["code"]) in keep]
                assert len(rows) == len(keep), f"{spec['id']}：子集色号对不上"
        else:
            data = json.loads(fetch(HB_RAW.format(commit=up["commit"], dir=spec["dir"])))
            for c in data["colors"]:
                if c.get("unidentified"):
                    continue  # 上游色号为「-」的颜色不猜测
                rgb = c["rgb"][:3]
                rows.append({"code": c["code"], "name": "", "r": rgb[0], "g": rgb[1], "b": rgb[2],
                             "series": series_of(c["code"]), "transparent": int(len(c["rgb"]) > 3 and c["rgb"][3] < 255)})
        out = HERE / "data" / f"{spec['id']}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["code", "name", "r", "g", "b", "series", "transparent"])
            w.writeheader()
            w.writerows(rows)
        print(f"{spec['id']}: {len(rows)} 色（未核实，只供 bench）")
    subprocess.check_call([sys.executable, "-m", "palettes.build_palettes", "--extra", str(HERE),
                           "--out", str(HERE / "build" / "palettes_bench.json")], cwd=REPO)


if __name__ == "__main__":
    main()
