"""对照 04 §2.1 的覆盖要求统计 manifest，输出 Markdown 表格和警告。

统计单位是「图纸单元」：普通图纸 1 张算 1 个，分页图（T4）一组算 1 套（见 bench/common/dataset.py）。
T7（多图合集）和 Web 版本（分辨率对比用）单独列出，不计入覆盖要求。
"""
from __future__ import annotations

from collections import Counter

from bench.common import dataset as ds
from bench.common import formats

TARGET_TOTAL = 150            # 07 T0.3 DoD：至少 150 张（不含 T7）
FIRST_ROUND = 120             # 04 §2.1：首轮标注 120 张（dev 40 + test 80）
BUCKET_SHARE = {"S": 0.20, "M": 0.30, "L": 0.30, "XL": 0.20}
BUCKET_TOL = 0.05             # 占比偏离目标超过 5 个百分点就警告
MIN_PER_TYPE = {"T2": 5, "T3": 5, "T5": 5, "T6": 5, "T4": 5}   # T4 按套计
MIN_TEST_PER_TYPE = 3
MIN_TEST_HARD = 40
BRANDS = {"MARD": "MARD", "COCO": "COCO", "MANMAN": "漫漫", "PANPAN": "盼盼", "MIXIAOWO": "咪小窝"}
MIN_PER_MINOR_BRAND = 3


def brand_of(hint: str | None) -> str | None:
    if not hint:
        return None
    return hint.split("-")[0].split(":")[0].upper()


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.0f}%" if b else "-"


def compute(manifest: dict[str, dict], dev_ids: list[str], test_ids: list[str]) -> dict:
    units = ds.build_units(manifest)
    split_of: dict[str, str] = {}
    for name, ids in (("dev", dev_ids), ("test", test_ids)):
        for iid in ids:
            if iid in manifest:
                split_of[ds.unit_key(manifest[iid])] = name
    rows = []
    for key, pages in units.items():
        a = ds.unit_attrs(pages)
        a["key"] = key
        a["split"] = split_of.get(key)
        a["hard"] = ds.is_hard(a)
        rows.append(a)
    variants = [r for r in manifest.values() if not ds.is_benchmark(r)]
    return {"units": rows, "n_variants": len(variants)}


def render(stats: dict) -> tuple[str, list[str]]:
    rows = stats["units"]
    main = [r for r in rows if r["type"] != "T7"]
    t7 = [r for r in rows if r["type"] == "T7"]
    n = len(main)
    n_img = sum(r["n_images"] for r in main)
    by_split = {s: [r for r in main if r["split"] == s] for s in ("dev", "test")}
    unsplit = [r for r in main if r["split"] is None]
    out: list[str] = []
    warn: list[str] = []

    out.append("## 总览\n")
    out.append(f"- 图纸单元 **{n}** 个（{n_img} 张图；分页图一组算 1 套），另有 T7 {len(t7)} 个、"
               f"Web 版本 {stats['n_variants']} 张（只用于分辨率对比），均不计入覆盖要求")
    out.append(f"- dev {len(by_split['dev'])} / test {len(by_split['test'])} / 未划分 {len(unsplit)}")
    out.append(f"- 目标：采集 ≥ {TARGET_TOTAL}（07 T0.3），首轮标注 {FIRST_ROUND}（04 §2.1，dev 40 + test 80）\n")
    if n < TARGET_TOTAL:
        warn.append(f"样本数 {n} < {TARGET_TOTAL}（07 T0.3 DoD）")

    # 尺寸档
    out.append("## 尺寸档\n")
    out.append("| 档 | 全部 | 占比 | 目标 | dev | test |")
    out.append("|---|---|---|---|---|---|")
    bc = Counter(r["size_bucket"] for r in main)
    for b in formats.SIZE_BUCKETS:
        share = bc[b] / n if n else 0.0
        out.append(f"| {b} | {bc[b]} | {_pct(bc[b], n)} | {BUCKET_SHARE[b]:.0%} | "
                   f"{sum(1 for r in by_split['dev'] if r['size_bucket'] == b)} | "
                   f"{sum(1 for r in by_split['test'] if r['size_bucket'] == b)} |")
        if n and abs(share - BUCKET_SHARE[b]) > BUCKET_TOL:
            warn.append(f"尺寸档 {b} 占比 {share:.0%}，目标 {BUCKET_SHARE[b]:.0%}")
    if bc.get(None):
        warn.append(f"{bc[None]} 个单元没有 size_bucket")
    out.append("")

    # 类型
    out.append("## 类型\n")
    out.append("| 类型 | 全部 | dev | test | 要求 |")
    out.append("|---|---|---|---|---|")
    tc = Counter(r["type"] for r in main)
    for t in formats.TYPES[:-1]:
        req = "≥ 50%" if t == "T1" else (f"≥ {MIN_PER_TYPE[t]}{' 套' if t == 'T4' else ''}" if t in MIN_PER_TYPE else "")
        req += f"；test ≥ {MIN_TEST_PER_TYPE}"
        out.append(f"| {t} | {tc[t]} | {sum(1 for r in by_split['dev'] if r['type'] == t)} | "
                   f"{sum(1 for r in by_split['test'] if r['type'] == t)} | {req} |")
    out.append(f"| T7 | {len(t7)} | - | - | 不进入自动识别统计 |")
    out.append("")
    if n and tc["T1"] < 0.5 * n:
        warn.append(f"T1 占比 {tc['T1'] / n:.0%} < 50%")
    for t, m in MIN_PER_TYPE.items():
        if tc[t] < m:
            warn.append(f"{t} 只有 {tc[t]} {'套' if t == 'T4' else '个'}，要求 ≥ {m}")
    if by_split["test"]:
        for t in formats.TYPES[:-1]:
            k = sum(1 for r in by_split["test"] if r["type"] == t)
            if k < MIN_TEST_PER_TYPE:
                warn.append(f"test 集里 {t} 只有 {k} 个，要求每种类型 ≥ {MIN_TEST_PER_TYPE}")

    # 尺寸 × 类型
    out.append("## 尺寸档 × 类型（全部 / test）\n")
    types = formats.TYPES[:-1]
    out.append("| 档 | " + " | ".join(types) + " |")
    out.append("|---|" + "---|" * len(types))
    for b in formats.SIZE_BUCKETS:
        cells = []
        for t in types:
            a = sum(1 for r in main if r["size_bucket"] == b and r["type"] == t)
            k = sum(1 for r in by_split["test"] if r["size_bucket"] == b and r["type"] == t)
            cells.append(f"{a} / {k}" if a else "")
        out.append(f"| {b} | " + " | ".join(cells) + " |")
    out.append("")

    # Hard
    hard_all = sum(1 for r in main if r["hard"])
    hard_test = sum(1 for r in by_split["test"] if r["hard"])
    out.append("## Hard 集（04 §0.2，按人工预估标签）\n")
    out.append(f"- 全部 {hard_all}，dev {sum(1 for r in by_split['dev'] if r['hard'])}，test {hard_test}"
               f"（要求 test ≥ {MIN_TEST_HARD}）\n")
    if by_split["test"] and hard_test < MIN_TEST_HARD:
        warn.append(f"test 集里 Hard 只有 {hard_test} 个，要求 ≥ {MIN_TEST_HARD}")

    # 品牌
    out.append("## 品牌（palette_hint）\n")
    out.append("| 品牌 | 全部 | test |")
    out.append("|---|---|---|")
    brc = Counter(brand_of(r["palette_hint"]) for r in main)
    for b, name in BRANDS.items():
        out.append(f"| {name}（{b}） | {brc[b]} | {sum(1 for r in by_split['test'] if brand_of(r['palette_hint']) == b)} |")
    others = sorted((k for k in brc if k not in BRANDS and k is not None))
    for b in others:
        out.append(f"| {b} | {brc[b]} | {sum(1 for r in by_split['test'] if brand_of(r['palette_hint']) == b)} |")
    out.append(f"| 未填 | {brc[None]} | - |")
    out.append("")
    if n and brc["MARD"] < max((brc[b] for b in brc if b != "MARD"), default=0):
        warn.append("MARD 不是数量最多的品牌（04 §2.1 要求以 MARD 221/291 为主）")
    for b in ("COCO", "MANMAN", "PANPAN", "MIXIAOWO"):
        if brc[b] < MIN_PER_MINOR_BRAND:
            warn.append(f"品牌 {BRANDS[b]} 只有 {brc[b]} 个，要求 ≥ {MIN_PER_MINOR_BRAND}")

    # 难点标签
    out.append("## 难点标签\n")
    out.append("| 标签 | 全部 | test |")
    out.append("|---|---|---|")
    tagc = Counter(t for r in main for t in r["tags"])
    for t in formats.KNOWN_TAGS:
        out.append(f"| {t} | {tagc[t]} | {sum(1 for r in by_split['test'] if t in r['tags'])} |")
        if tagc[t] == 0:
            warn.append(f"没有带 {t} 标签的样本")
    for t in sorted(k for k in tagc if k not in formats.KNOWN_TAGS):
        out.append(f"| {t}（自定义） | {tagc[t]} | {sum(1 for r in by_split['test'] if t in r['tags'])} |")
    no_wm = sum(1 for r in main if "watermark" not in r["tags"])
    out.append(f"\n- 无水印 {no_wm}，有水印 {tagc['watermark']}\n")
    if n and no_wm == 0:
        warn.append("没有无水印的样本")

    if unsplit and (by_split["dev"] or by_split["test"]):
        warn.append(f"{len(unsplit)} 个单元还没有划分（运行 split）")
    out.append("## 警告\n")
    if warn:
        out.extend(f"- {w}" for w in warn)
    else:
        out.append("- 无")
    return "\n".join(out) + "\n", warn
