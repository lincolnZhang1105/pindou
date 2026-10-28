"""合成样本生成：底图 → 配色 → 渲染 → 分页 / 拼图 → 退化 → 写图片、真值、附加标注。

输出目录布局（和私有数据目录一致，方便直接用 `PINDOU_DATA_DIR=<out>` 跑 eval / 标注工具）：

    <out>/images/<image_id>.<ext>          图片（分页图：<group>_<k>.<ext>）
    <out>/labels/<image_id>.json           真值（bench/README.md 格式，通过 validate_label）
    <out>/extra/<image_id>.extra.json      训练用附加标注（格式见 bench/synth/README.md）
    <out>/manifest.jsonl                   manifest（platform=synth）

确定性：每个样本用 SeedSequence([seed, index]) 派生自己的随机数，同样的 seed / index / 字体 / 依赖版本
得到逐字节相同的输出，与并行进程数无关。
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bench.common.formats import save_json, size_bucket_of, validate_label
from bench.synth import degrade as D
from bench.synth.basemaps import make_basemap, size_for_bucket
from bench.synth.colors import assign_colors
from bench.synth.fonts import FontBook
from bench.synth.geom import Ann, box_tf, mat_translate, rbox
from bench.synth.palettes import (BENCH_PATH, CLEAN_PATH, choose_scheme, clean_color_pool, load_clean_palettes,
                                  load_code_tables)
from bench.synth.render import Pattern, Rendered, render
from bench.synth.styles import STYLE_NAMES, TYPE_STYLES, sample_style

VERSION = "0.1.0"
TYPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
TYPE_WEIGHTS = {"T1": 0.5, "T2": 0.12, "T3": 0.08, "T4": 0.1, "T5": 0.08, "T6": 0.08, "T7": 0.04}
BUCKET_WEIGHTS = {"S": 0.25, "M": 0.3, "L": 0.25, "XL": 0.2}
STYLE_WEIGHTS = {"table_cyan": 0.22, "fine5": 0.34, "fine10": 0.22, "nocoord": 0.22, "blocks": 1.0, "pixel": 1.0}
RESIZE_WIDTHS = {1080: 0.55, 1040: 0.08, 1000: 0.08, 1200: 0.05, 1440: 0.06, 828: 0.1, 750: 0.08}
FORMATS = {"png": 0.25, "webp": 0.5, "jpg": 0.25}


@dataclass
class GenOptions:
    types: tuple[str, ...] | None = None
    styles: tuple[str, ...] | None = None
    buckets: tuple[str, ...] | None = None
    style_overrides: dict = field(default_factory=dict)
    # 强制退化参数（测试用）。键：watermark(bool)、resize(int 宽度 / 0 表示不缩放)、fmt、quality、blur(float)、
    # pre_jpeg(T5 截图的第一次 JPEG 质量，0 表示不做)、t4_mode(board / nine)
    degrade: dict = field(default_factory=dict)
    max_side: int | None = None          # 限制图纸边长（测试提速）
    allow_fallback_font: bool = False
    clean_path: Path = CLEAN_PATH
    bench_path: Path = BENCH_PATH


def _wchoice(rng, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=np.float64)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


@dataclass
class _Img:
    image: np.ndarray
    ann: Ann
    M: np.ndarray                     # 渲染坐标 → 本图坐标（拼图、截图前的坐标系）
    ops: list = field(default_factory=list)
    data: bytes = b""
    fmt: str = "png"


class Generator:
    def __init__(self, opts: GenOptions | None = None):
        self.opts = opts or GenOptions()
        self.fonts = FontBook(allow_fallback=self.opts.allow_fallback_font)
        self.clean = load_clean_palettes(self.opts.clean_path)
        self.tables = load_code_tables(self.opts.bench_path, self.opts.clean_path)
        self.pool = clean_color_pool(self.opts.clean_path)

    # ------------------------------------------------------------------ 抽样计划
    def _plan(self, rng) -> tuple[str, str, str]:
        o = self.opts
        styles = tuple(o.styles) if o.styles else STYLE_NAMES
        buckets = tuple(o.buckets) if o.buckets else tuple(BUCKET_WEIGHTS)
        types = []
        for t in (o.types or TYPES):
            if not set(TYPE_STYLES[t]) & set(styles):
                continue
            if t == "T4" and "XL" not in buckets:
                continue
            if t == "T7" and not {"S", "M"} & set(buckets):
                continue
            types.append(t)
        if not types:
            raise ValueError(f"--type / --style / --size-bucket 的组合没有可生成的样本：{o.types} {o.styles} {o.buckets}")
        t = _wchoice(rng, {k: TYPE_WEIGHTS[k] for k in types})
        cand = [s for s in TYPE_STYLES[t] if s in styles]
        style = _wchoice(rng, {s: STYLE_WEIGHTS[s] for s in cand})
        if t == "T4":
            bucket = _wchoice(rng, {"M": 0.2, "L": 0.4, "XL": 0.4})
        elif t == "T7":
            bucket = _wchoice(rng, {b: BUCKET_WEIGHTS[b] for b in buckets if b in ("S", "M")})
        else:
            bucket = _wchoice(rng, {b: BUCKET_WEIGHTS[b] for b in buckets})
        return t, style, bucket

    def _size(self, rng, bucket: str) -> tuple[int, int]:
        cols, rows = size_for_bucket(rng, bucket)
        if self.opts.max_side:
            k = self.opts.max_side / max(cols, rows)
            if k < 1:
                cols, rows = max(4, int(cols * k)), max(4, int(rows * k))
        return cols, rows

    def _pattern(self, rng, st, cols: int, rows: int) -> tuple[Pattern, object, str, bool]:
        bm = make_basemap(rng, cols, rows, kind=None if st.name != "pixel" else
                          str(rng.choice(["sprite", "multi", "scene", "sprite"])))
        native_only = (not st.cell_text) and st.legend == "none"
        for _ in range(20):
            pid, native = choose_scheme(rng, self.tables, native_only, short_codes=st.cell_text)
            if len(self.tables[pid].codes) >= len(bm.roles) + 2:
                break
        else:
            pid, native = "ARTKAL-S", True
        ents, empty = assign_colors(rng, bm, self.tables[pid], self.clean[pid] if native else None, self.pool,
                                    st.empty_rgb)
        st.empty_rgb = empty
        pat = Pattern(idx=bm.idx, codes=[e.code for e in ents], rgb=np.array([e.rgb for e in ents], dtype=np.uint8),
                      sources=[e.source for e in ents], empty_rgb=empty)
        pat.roles = [r.kind for r in bm.roles]
        return pat, bm, pid, native

    # ------------------------------------------------------------------ 分页（T4）
    def _paged(self, rng, pat: Pattern, st) -> tuple[list[_Img], dict]:
        mode = self.opts.degrade.get("t4_mode") or ("board" if rng.random() < 0.6 else "nine")
        cols, rows = pat.cols, pat.rows
        if mode == "board":
            sizes = [s for s in (16, 20, 26, 29, 32, 40, 52) if 2 <= math.ceil(cols / s) * math.ceil(rows / s) <= 16]
            bs = int(sizes[int(rng.integers(len(sizes)))]) if sizes else max(4, (max(cols, rows) + 1) // 2)
            where = str(rng.choice(["last", "separate", "first"]))
            wins = [(c0, r0, min(bs, cols - c0), min(bs, rows - r0)) for r0 in range(0, rows, bs)
                    for c0 in range(0, cols, bs)]
            out = []
            for k, w in enumerate(wins):
                show_leg = (where == "last" and k == len(wins) - 1) or (where == "first" and k == 0)
                r = render(pat, st, self.fonts, rng, window=w, show_legend=show_leg, show_title=(k == 0))
                out.append(_Img(r.image, r.ann, np.eye(3)))
            if where == "separate" and st.legend != "none":
                r = render(pat, st, self.fonts, rng, show_grid=False, show_title=False,
                           width_hint=out[0].image.shape[1])
                out.append(_Img(r.image, r.ann, np.eye(3)))
            return out, {"mode": "board", "board": bs, "legend": where if st.legend != "none" else "none"}
        # nine：整张渲染后沿格子边界切成 nx × ny 块，图例单独一张
        r = render(pat, st, self.fonts, rng)
        nx = int(rng.integers(2, 4)) if cols >= 8 else 1
        ny = int(rng.integers(2, 4)) if rows >= 8 else 1
        if nx * ny == 1:
            nx = 2
        page = r.ann.pages[0]
        xl, yl = page["x_lines"], page["y_lines"]
        cs = [int(round(v)) for v in np.linspace(0, cols, nx + 1)]
        rs = [int(round(v)) for v in np.linspace(0, rows, ny + 1)]
        tmax = max(st.thin_w, st.bold_w, st.border_w, 1)
        H, W = r.image.shape[:2]
        reg = {g["kind"]: g["box"] for g in r.ann.regions}
        grid_end = yl[-1] + tmax
        for k in ("coords_bottom",):
            if k in reg:
                grid_end = max(grid_end, reg[k][3])
        grid_end = int(min(H, math.ceil(grid_end + max(2, st.pitch * 0.3))))
        out = []
        for j in range(ny):
            for i in range(nx):
                x0 = 0 if i == 0 else int(math.floor(xl[cs[i]] - tmax / 2 - 1))
                x1 = W if i == nx - 1 else int(math.ceil(xl[cs[i + 1]] + tmax / 2 + 1))
                y0 = 0 if j == 0 else int(math.floor(yl[rs[j]] - tmax / 2 - 1))
                y1 = grid_end if j == ny - 1 else int(math.ceil(yl[rs[j + 1]] + tmax / 2 + 1))
                a = r.ann.cropped(x0, y0, x1, y1)
                a.legend = []
                a.texts = [it for it in a.texts if it["kind"] == "title"]
                a.regions = [g for g in a.regions if g["kind"].startswith("coords") or g["kind"] == "title"]
                a.pages = [{"image_index": 0, "x_lines": [x - x0 for x in xl[cs[i]:cs[i + 1] + 1]],
                            "y_lines": [y - y0 for y in yl[rs[j]:rs[j + 1] + 1]], "cols": cs[i + 1] - cs[i],
                            "rows": rs[j + 1] - rs[j], "col_offset": cs[i], "row_offset": rs[j], "H": np.eye(3)}]
                a.regions.append({"kind": "grid", "box": [a.pages[0]["x_lines"][0], a.pages[0]["y_lines"][0],
                                                          a.pages[0]["x_lines"][-1], a.pages[0]["y_lines"][-1]]})
                a.coords = [c for c in a.coords
                            if (c["axis"] == "col" and cs[i] <= c["index"] < cs[i + 1])
                            or (c["axis"] == "row" and rs[j] <= c["index"] < rs[j + 1])]
                out.append(_Img(r.image[y0:y1, x0:x1].copy(), a, mat_translate(-x0, -y0)))
        if r.ann.legend and grid_end < H - 4:
            a = r.ann.cropped(0, grid_end, W, H)
            a.pages, a.coords = [], []
            a.regions = [g for g in a.regions if g["kind"] == "legend"]
            out.append(_Img(r.image[grid_end:H].copy(), a, mat_translate(0, -grid_end)))
        return out, {"mode": "nine", "tiles": [nx, ny], "legend": "separate" if r.ann.legend else "none"}

    # ------------------------------------------------------------------ 拼图（T7）
    def _collage(self, rng, st, bucket: str, first: Pattern) -> tuple[_Img, list, list]:
        n = int(rng.choice([2, 3, 4, 4, 6, 9]))
        gc = {2: 2, 3: 3, 4: 2, 6: 3, 9: 3}[n]
        gr = math.ceil(n / gc)
        rends: list[Rendered] = []
        pats = [first]
        for k in range(n):
            if k > 0:
                cols, rows = self._size(rng, bucket)
                cols, rows = max(8, int(cols * rng.uniform(0.7, 1.1))), max(8, int(rows * rng.uniform(0.7, 1.1)))
                p2, _, _, _ = self._pattern(rng, st, cols, rows)
                pats.append(p2)
            rends.append(render(pats[k], st, self.fonts, rng, show_title=False))
        cw = max(r.image.shape[1] for r in rends)
        ch = max(r.image.shape[0] for r in rends)
        gap = int(cw * rng.uniform(0.02, 0.08))
        W, H = gc * cw + (gc + 1) * gap, gr * ch + (gr + 1) * gap
        canvas = np.empty((H, W, 3), dtype=np.uint8)
        canvas[:] = np.array(st.bg_rgb if rng.random() < 0.7 else (240, 240, 244), dtype=np.uint8)
        ann = Ann(width=W, height=H)
        panels = []
        for k, r in enumerate(rends):
            gy, gx = divmod(k, gc)
            h, w = r.image.shape[:2]
            ox = gap + gx * (cw + gap) + (cw - w) // 2
            oy = gap + gy * (ch + gap) + (ch - h) // 2
            canvas[oy:oy + h, ox:ox + w] = r.image
            a = r.ann.transformed(mat_translate(ox, oy), W, H)
            for name in ("legend", "coords", "texts", "regions"):
                for it in getattr(a, name):
                    it["panel"] = k
                getattr(ann, name).extend(getattr(a, name))
            if k == 0:
                ann.pages = a.pages
            panels.append({"panel": k, "box": [ox, oy, ox + w, oy + h], "cols": pats[k].cols, "rows": pats[k].rows})
        return _Img(canvas, ann, np.eye(3)), panels, rends

    # ------------------------------------------------------------------ 退化
    def _degrade_plan(self, rng, t: str, style: str) -> dict:
        f = self.opts.degrade
        plan = {
            "watermark": f.get("watermark", rng.random() < (0.15 if style == "pixel" else 0.35)),
            "resize": f.get("resize", (int(_wchoice(rng, RESIZE_WIDTHS)) if rng.random() < 0.75 else 0)),
            "blur": f.get("blur", float(rng.uniform(0.5, 1.2)) if rng.random() < 0.08 else 0.0),
            "fmt": f.get("fmt", _wchoice(rng, FORMATS)),
            "quality": int(f.get("quality", int(rng.integers(60, 91)))),
        }
        if t == "T5":
            plan["pre_jpeg"] = int(f.get("pre_jpeg", int(rng.integers(70, 93))))
            if "fmt" not in f and plan["fmt"] == "png":
                plan["fmt"] = "jpg"
        if t == "T7":  # 合集图在平台上一定会被压到 1080 左右
            plan["resize"] = f.get("resize", int(_wchoice(rng, {1080: 0.7, 1000: 0.15, 1440: 0.15})))
        if t == "T6":
            plan["fmt"] = f.get("fmt", "jpg")
            plan["quality"] = int(f.get("quality", int(rng.integers(70, 93))))
            plan["resize"] = f.get("resize", 0)
        return plan

    def _apply(self, rng, im: _Img, t: str, plan: dict) -> _Img:
        img, M, ops = im.image, np.eye(3), []
        if plan["watermark"]:
            img, info = D.watermark(img, rng, self.fonts)
            ops.append({"op": "watermark", **info})
        extra_texts = []
        if t == "T5":
            img, Ms, info, extra_texts = D.screenshot(img, rng, self.fonts)
            M = Ms @ M
            ops.append({"op": "screenshot", **info})
        if t == "T6":
            img, Mp, info = D.photo(img, rng)
            M = Mp @ M
            ops.append({"op": "photo", **info})
        W = img.shape[1]
        target = int(plan["resize"] or 0)
        if W > 4000 and not target:
            target = 2048
        if target and W > target:
            img, Ms = D.resize(img, target)
            M = Ms @ M
            ops.append({"op": "resize", "width": target})
        if plan["blur"]:
            img = D.gaussian_blur(img, plan["blur"])
            ops.append({"op": "blur", "sigma": round(plan["blur"], 3)})
        if plan.get("pre_jpeg"):
            img = D.decode(D.encode(img, "jpg", plan["pre_jpeg"]))
            ops.append({"op": "jpeg", "quality": plan["pre_jpeg"]})
        fmt = plan["fmt"]
        data = D.encode(img, fmt, plan["quality"])
        ops.append({"op": "encode", "format": fmt, "quality": None if fmt == "png" else plan["quality"]})
        ann = im.ann.transformed(M, img.shape[1], img.shape[0])
        ann.texts.extend(extra_texts)
        return _Img(img, ann, M @ im.M, ops, data, fmt)

    # ------------------------------------------------------------------ 主流程
    def generate(self, seed: int, index: int, out_dir: Path) -> list[dict]:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
        out_dir = Path(out_dir)
        t, style_name, bucket = self._plan(rng)
        cols, rows = self._size(rng, bucket)
        st = sample_style(rng, style_name, cols, rows, self.fonts, self.opts.style_overrides)
        pat, bm, pid, native = self._pattern(rng, st, cols, rows)
        image_id = f"synth_s{seed}_{index:06d}"

        panels = None
        layout = {}
        if t == "T4":
            imgs, layout = self._paged(rng, pat, st)
        elif t == "T7":
            im, panels, _ = self._collage(rng, st, bucket, pat)
            imgs = [im]
        else:
            r = render(pat, st, self.fonts, rng)
            imgs = [_Img(r.image, r.ann, np.eye(3))]
        plan = self._degrade_plan(rng, t, style_name)
        outs = [self._apply(rng, im, t, plan) for im in imgs]

        # 文件
        multi = len(outs) > 1
        files = []
        for k, o in enumerate(outs):
            name = f"{image_id}_{k}.{o.fmt}" if multi else f"{image_id}.{o.fmt}"
            path = out_dir / "images" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(o.data)
            files.append((name, hashlib.sha256(o.data).hexdigest()))

        # 真值
        tags = set(bm.tags)
        if plan["watermark"]:
            tags.add("watermark")
        pages = []
        for k, o in enumerate(outs):
            for p in o.ann.pages:
                q = dict(p)
                q["image_index"] = k
                pages.append(o.ann.page_label(q))
        min_cell = min(min((pg["x_lines"][-1] - pg["x_lines"][0]) / pg["cols"],
                           (pg["y_lines"][-1] - pg["y_lines"][0]) / pg["rows"]) for pg in pages)
        if min_cell < 8 or plan["blur"] >= 1.0 or any(op.get("blur_sigma", 0) >= 1.2 for o in outs for op in o.ops):
            tags.add("blurry")
        has_legend = any(o.ann.legend for o in outs) and t != "T7" or (t == "T7" and any(
            it.get("panel") == 0 for it in outs[0].ann.legend))
        legend = []
        if has_legend:
            legend = [{"code": pat.codes[k], "count": int(pat.counts[k])} for k in pat.legend_order(st.legend_order)]
        cells = [[pat.codes[v] if v >= 0 else None for v in row] for row in pat.idx.tolist()]
        label = {
            "image_id": image_id,
            "source": {"platform": "synth", "note_id": None, "author": "bench.synth", "saved_at": None,
                       "source_path": "photo" if t == "T6" else "own_export",
                       "generator": f"bench.synth {VERSION}", "seed": int(seed), "index": int(index)},
            "type": t,
            "size_bucket": size_bucket_of(pat.cols, pat.rows, paged=(t == "T4")),
            "palette_hint": pid,
            "images": [{"file": n, "sha256": h} for n, h in files],
            "grid": {"cols": pat.cols, "rows": pat.rows},
            "pages": pages,
            "legend": legend,
            "cells": cells,
            "tags": sorted(tags),
        }
        errs = validate_label(label)
        if errs:
            raise AssertionError(f"{image_id} 真值校验失败：{errs}")
        save_json(out_dir / "labels" / f"{image_id}.json", label, indent=None)

        # 附加标注
        lum_white = [bool(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] < st.lum_thr) for c in pat.rgb.tolist()]
        has_text = ["".join("1" if (v >= 0 and st.cell_text) else "0" for v in row) for row in pat.idx.tolist()]
        tcolor = ["".join("." if (v < 0 or not st.cell_text) else ("w" if lum_white[v] else "k") for v in row)
                  for row in pat.idx.tolist()]
        text_boxes = {}
        if st.cell_text:
            # 文字框只依赖色号、字体和格距，任意一页重新取都一样；从第一次渲染里取不到时重新量
            from bench.synth.render import _fit_font, _glyph
            used = [k for k in range(len(pat.codes)) if pat.counts[k] > 0]
            for k in used:
                f = (_fit_font(self.fonts, st.text_face, st.text_weight, st.text_size,
                               [pat.codes[j] for j in used], 0.9 * st.pitch) if st.text_fit == "longest" else
                     _fit_font(self.fonts, st.text_face, st.text_weight, st.text_size, [pat.codes[k]], 0.9 * st.pitch))
                _, box = _glyph(pat.codes[k], f, st.pitch, st.text_dy)
                if box is not None:
                    text_boxes[pat.codes[k]] = [round(v, 4) for v in box]
        extra = {
            "format": "pindou-synth-extra", "version": 1, "image_id": image_id, "generator": f"bench.synth {VERSION}",
            "seed": int(seed), "index": int(index), "type": t, "size_bucket": label["size_bucket"],
            "style": st.describe(), "layout": layout,
            "basemap": {"kind": bm.kind},
            "palette": {"code_table": pid, "native": bool(native)},
            "codebook": [{"code": pat.codes[k], "rgb": [int(v) for v in pat.rgb[k]], "source": pat.sources[k],
                          "role": pat.roles[k], "count": int(pat.counts[k]), "text_color": "w" if lum_white[k] else "k"}
                         for k in range(len(pat.codes))],
            "empty_rgb": [int(v) for v in pat.empty_rgb],
            "cells": {"has_text": has_text, "text_color": tcolor},
            "cell_text_boxes": text_boxes,
            "images": [_image_extra(n, o) for (n, _), o in zip(files, outs)],
        }
        if panels is not None:
            extra["panels"] = [{**pn, "box": rbox(box_tf(outs[0].M, pn["box"]))} for pn in panels]
        save_json(out_dir / "extra" / f"{image_id}.extra.json", extra, indent=None)

        # manifest
        rows_out = []
        for k, ((n, h), o) in enumerate(zip(files, outs)):
            iid = f"{image_id}_{k}" if multi else image_id
            rows_out.append({
                "image_id": iid, "file": n, "sha256": h, "width": int(o.image.shape[1]),
                "height": int(o.image.shape[0]), "format": o.fmt, "platform": "synth", "note_id": None,
                "author": "bench.synth", "url": None, "saved_at": None, "source_path": label["source"]["source_path"],
                "type": t, "size_bucket": label["size_bucket"], "palette_hint": pid, "tags": label["tags"],
                "group": image_id if multi else None, "notes": f"style={style_name}"})
        return rows_out


def _image_extra(name: str, o: _Img) -> dict:
    def clean(items):
        out = []
        for it in items:
            d = {}
            for k, v in it.items():
                if k.endswith("box") and v is not None:
                    d[k] = [round(float(x), 2) for x in v]
                else:
                    d[k] = v
            out.append(d)
        return out
    return {"file": name, "width": int(o.image.shape[1]), "height": int(o.image.shape[0]),
            "transform": [[float(round(v, 9)) for v in row] for row in (o.M / o.M[2, 2])],
            "degradations": o.ops, "legend": clean(o.ann.legend), "coords": clean(o.ann.coords),
            "texts": clean(o.ann.texts), "regions": clean(o.ann.regions)}


# --------------------------------------------------------------------------- 多进程入口

_GEN: Generator | None = None


def _init(opts: GenOptions) -> None:
    global _GEN
    _GEN = Generator(opts)


def _work(args) -> list[dict]:
    seed, index, out = args
    return _GEN.generate(seed, index, Path(out))
