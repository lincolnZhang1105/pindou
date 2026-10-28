"""生成 palettes_json（03 §3.5），供 beadcore 的 bc_create 使用。

用法（在仓库根目录）：
    python -m palettes.build_palettes                      # → palettes/build/palettes.json（只含 palettes/data）
    python -m palettes.build_palettes --extra DIR --out F  # 额外合并其他目录的色卡（bench 内部用，见 bench/palettes_unverified）

每个色卡目录里有 <id>.csv（code,name,r,g,b,series,transparent），色卡元数据来自同目录或上级的 sources.json。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bench.common.color import srgb8_to_lab

HERE = Path(__file__).resolve().parent


def load_dir(data_dir: Path, sources_json: Path) -> list[dict]:
    cfg = json.loads(sources_json.read_text(encoding="utf-8"))
    upstreams = cfg.get("upstreams", {})
    out = []
    for spec in cfg["palettes"]:
        path = data_dir / f"{spec['id']}.csv"
        if not path.exists():
            raise SystemExit(f"缺少 {path}，先运行导入脚本")
        up = upstreams.get(spec["upstream"], {})
        colors = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rgb = [int(row["r"]), int(row["g"]), int(row["b"])]
                lab = srgb8_to_lab(rgb)
                c = {"code": row["code"], "name": row["name"], "rgb": rgb,
                     "lab": [round(float(v), 4) for v in lab], "series": row["series"]}
                if int(row.get("transparent") or 0):
                    c["transparent"] = True
                colors.append(c)
        out.append({
            "id": spec["id"],
            "version": up.get("commit", spec.get("version", "")),
            "source": up.get("repo", spec.get("source", "")),
            "license": up.get("license", spec.get("license", "")),
            "brand": spec.get("brand", ""),
            "bead_mm": spec.get("bead_mm"),
            "verified": spec.get("verified", True),
            "colors": colors,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "build" / "palettes.json")
    ap.add_argument("--extra", type=Path, action="append", default=[],
                    help="额外的色卡目录（含 sources.json 和 data/）。只给 bench 内部用，产物不得进入 App")
    args = ap.parse_args()

    palettes = load_dir(HERE / "data", HERE / "sources.json")
    for extra in args.extra:
        palettes += load_dir(extra / "data", extra / "sources.json")
    ids = [p["id"] for p in palettes]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"色卡 id 重复：{ids}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"palettes": palettes}, ensure_ascii=False, separators=(",", ":")) + "\n",
                        encoding="utf-8")
    print(f"写入 {args.out}：{len(palettes)} 套色卡，{sum(len(p['colors']) for p in palettes)} 色")


if __name__ == "__main__":
    main()
