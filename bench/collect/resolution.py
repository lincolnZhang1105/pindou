"""T0.3 子项：同一张图「App 保存」和「Web 端」两个版本的分辨率对比（04 §1.1「待实测」）。

manifest 里 Web 端版本的记录用 `pair_of` 指向 App 保存版本（source_path=web）。这里只生成表格，
结论由人写进 04 §1.1。

每格像素的估算：
- App 版本有真值时，用真值里该页的 x_lines / y_lines 精确计算（图纸区域宽 / 列数）；Web 版本按宽度等比换算；
- 没有真值但登记时给了 grid（列x行）时，用「整图宽 / 列数」，这是上限（图纸区域比整图窄），标「≤」。
"""
from __future__ import annotations

import statistics
from pathlib import Path

from bench.common import dataset as ds
from bench.common import formats


def find_pairs(manifest: dict[str, dict]) -> tuple[list[tuple[dict, dict]], list[str]]:
    pairs, warnings, seen = [], [], set()
    for r in manifest.values():
        other_id = r.get("pair_of")
        if not other_id:
            continue
        other = manifest.get(other_id)
        if other is None:
            warnings.append(f"{r['image_id']}：pair_of 指向的 {other_id} 不存在")
            continue
        web, app = (r, other) if r.get("source_path") == "web" else (other, r)
        if web.get("source_path") != "web":
            warnings.append(f"{r['image_id']} ↔ {other_id}：两边都不是 Web 版本（source_path=web），跳过")
            continue
        k = (app["image_id"], web["image_id"])
        if k not in seen:
            seen.add(k)
            pairs.append((app, web))
    pairs.sort(key=lambda p: p[0]["image_id"])
    return pairs, warnings


def _cell_px(app: dict, root: Path | None) -> tuple[float, float, str] | None:
    """App 版本的每格像素 (宽, 高, 来源)。"""
    if root is not None:
        lp = ds.label_path(ds.unit_key(app), root)
        if lp.is_file():
            try:
                lab = formats.load_json(lp)
                idx = [i for i, im in enumerate(lab.get("images", [])) if im.get("file") == app["file"]]
                for p in lab.get("pages", []):
                    if idx and p.get("image_index") == idx[0] and p.get("cols") and p.get("rows"):
                        xl, yl = p["x_lines"], p["y_lines"]
                        return (xl[-1] - xl[0]) / p["cols"], (yl[-1] - yl[0]) / p["rows"], "真值"
            except (ValueError, KeyError, IndexError):
                pass
    g = app.get("grid")
    if g:
        return app["width"] / g["cols"], app["height"] / g["rows"], "上限"
    return None


def _size(rec: dict, root: Path | None) -> str:
    if root is None:
        return "-"
    p = ds.image_path(rec, root)
    return f"{p.stat().st_size / 1024:.0f} KB" if p.is_file() else "-"


def report(manifest: dict[str, dict], root: Path | None = None) -> tuple[str, list[str]]:
    pairs, warnings = find_pairs(manifest)
    out = ["# App 保存 vs Web 端分辨率对比\n",
           "| App 版本 | App：尺寸 | 格式 | 大小 | Web：尺寸 | 格式 | 大小 | 宽度比 App/Web | 每格像素 App | 每格像素 Web | 依据 |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    ratios = []
    for app, web in pairs:
        ratio = app["width"] / web["width"]
        ratios.append(ratio)
        cell = _cell_px(app, root)
        if cell:
            cw, ch, src = cell
            pre = "≤ " if src == "上限" else ""
            s = web["width"] / app["width"]
            app_cell = f"{pre}{cw:.1f}×{ch:.1f}"
            web_cell = f"{pre}{cw * s:.1f}×{ch * s:.1f}"
        else:
            app_cell = web_cell = src = "-"
        out.append(f"| {app['image_id']} | {app['width']}×{app['height']} | {app['format']} | {_size(app, root)} | "
                   f"{web['width']}×{web['height']} | {web['format']} | {_size(web, root)} | "
                   f"{ratio:.2f} | {app_cell} | {web_cell} | {src} |")
    out.append("")
    if ratios:
        bigger = sum(1 for r in ratios if r > 1.0001)
        out.append(f"- 共 {len(pairs)} 对；宽度比中位数 {statistics.median(ratios):.2f}"
                   f"（最小 {min(ratios):.2f}，最大 {max(ratios):.2f}）；App 版本更大的有 {bigger} 对，"
                   f"一样大的 {sum(1 for r in ratios if abs(r - 1) <= 0.0001)} 对")
    else:
        out.append("- 还没有配对样本：登记 Web 端版本时用 `--pair-of <App 版本 image_id>`")
    out.append("- 每格像素：依据为「真值」时按 x_lines/y_lines 精确计算；「上限」= 整图宽高 / 行列数")
    return "\n".join(out) + "\n", warnings
