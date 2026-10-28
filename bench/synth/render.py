"""把（底图 + 色号 + 风格）渲染成「主流 App 导出的图纸」，同时产出所有几何标注。

渲染坐标系见 `geom.py`。格子是整数格距 p，第 i 条竖线画在 x = gx + i·p 处（线宽 t 时覆盖
[x − ⌊t/2⌋, x − ⌊t/2⌋ + t)），真值记录线的中心。格内文字按色号预渲染成 p×p 的 alpha 小图，
用 numpy 一次性贴到所有格子上（208×208 的图纸也只要几十毫秒）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from bench.synth.fonts import FontBook
from bench.synth.geom import Ann
from bench.synth.styles import Style


@dataclass
class Pattern:
    """一张图纸的内容：idx (rows, cols)，-1 为空格；第 k 色的色号、色值、色值来源。"""
    idx: np.ndarray
    codes: list[str]
    rgb: np.ndarray
    sources: list[str]
    empty_rgb: tuple
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.counts = np.bincount(self.idx[self.idx >= 0].ravel(), minlength=len(self.codes))

    @property
    def rows(self) -> int:
        return int(self.idx.shape[0])

    @property
    def cols(self) -> int:
        return int(self.idx.shape[1])

    def legend_order(self, order: str) -> list[int]:
        ks = [k for k in range(len(self.codes)) if self.counts[k] > 0]
        if order == "first_seen":
            flat = self.idx.ravel()
            flat = flat[flat >= 0]
            _, first = np.unique(flat, return_index=True)
            pos = {int(k): int(i) for k, i in zip(np.unique(flat), first)}
            return sorted(ks, key=lambda k: pos[k])
        if order == "code_str":
            return sorted(ks, key=lambda k: self.codes[k])
        if order == "count_desc":
            return sorted(ks, key=lambda k: (-int(self.counts[k]), self.codes[k]))
        return sorted(ks, key=lambda k: _natkey(self.codes[k]))


def _natkey(code: str):
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.findall(r"\d+|\D+", code)]


def luminance(rgb) -> float:
    r, g, b = (float(v) for v in rgb)
    return 0.299 * r + 0.587 * g + 0.114 * b


@dataclass
class Rendered:
    image: np.ndarray                 # (H, W, 3) uint8
    ann: Ann
    text_boxes: dict = field(default_factory=dict)   # 色号 → 文字在格内的框（以格距为单位的比例）
    has_text: np.ndarray | None = None               # 本页 (rows, cols) bool
    text_white: np.ndarray | None = None             # 本页 (rows, cols) bool：白字


# --------------------------------------------------------------------------- 小工具

def _fit_font(fonts: FontBook, face, weight: str, size: int, texts: list[str], max_w: float, min_size: int = 4):
    s = int(size)
    while s > min_size:
        f = fonts.get(face, s, weight)
        if max((f.getlength(t) for t in texts), default=0) <= max_w:
            return f
        s -= 1
    return fonts.get(face, min_size, weight)


def _text_h(font) -> float:
    b = font.getbbox("0123456789")
    return float(b[3] - b[1])


def _glyph(code: str, font, p: int, dy: float) -> tuple[np.ndarray, list[float] | None]:
    im = Image.new("L", (p, p), 0)
    d = ImageDraw.Draw(im)
    d.text((p / 2.0, p / 2.0 + dy), code, font=font, fill=255, anchor="mm")
    a = np.asarray(im, dtype=np.uint8)
    ys, xs = np.nonzero(a > 40)
    box = None if len(xs) == 0 else [xs.min() / p, ys.min() / p, (xs.max() + 1) / p, (ys.max() + 1) / p]
    return a, box


# --------------------------------------------------------------------------- 图例

class _Legend:
    """图例布局：先 measure 出高度，再 draw。条目框记录在 items 里（渲染坐标）。"""

    def __init__(self, pat: Pattern, st: Style, fonts: FontBook, width: int, rng_vals: dict):
        self.pat, self.st, self.fonts, self.W = pat, st, fonts, width
        self.order = pat.legend_order(st.legend_order)
        self.n = len(self.order)
        self.v = rng_vals
        self.items: list[dict] = []
        self.texts: list[dict] = []
        lay = st.legend
        avail = max(120, width)
        if lay == "table":
            nc = max(1, min(2 if (st.legend_cols == 1 and self.n > 10) else st.legend_cols, self.n))
            gapu = self.v["table_gap"]
            self.u = max(14.0, min(avail / (nc * 6 + (nc - 1) * gapu + 2 * 0.8), avail / 13.0))
            self.nc, self.gapu = nc, gapu
            self.nrows = math.ceil(self.n / nc)
            self.h = self.nrows * self.u + self.u * 0.6
        elif lay in ("chips_in", "chips_under"):
            u = max(18.0, avail * st.legend_unit_frac)
            gx = u * self.v["chip_gap"]
            self.u, self.gx = u, gx
            self.per = max(1, int((avail + gx) // (u + gx)))
            self.nrows = math.ceil(self.n / self.per)
            cf = fonts.get(st.legend_count_face, int(u * 0.42), st.legend_weight)
            self.cf = cf
            th = _text_h(cf)
            if lay == "chips_in":
                self.row_h = u + th * 1.9 + u * 0.35
            else:
                self.codef = fonts.get(st.legend_face, int(u * 0.38), st.legend_weight)
                self.row_h = u + th * 1.6 + _text_h(self.codef) * 1.7 + u * 0.35
            self.h = self.nrows * self.row_h
        elif lay == "inline":
            u = max(14.0, avail * st.legend_unit_frac * 0.75)
            self.u = u
            self.nc = max(1, min(st.legend_cols, self.n))
            self.colw = avail / self.nc
            self.nrows = math.ceil(self.n / self.nc)
            self.row_h = u * 1.45
            self.h = self.nrows * self.row_h
        else:
            self.h = 0
        self.total_cn = False
        if st.total_line and self.h:
            cjk = [f for f in fonts.candidates("legend") if f.cjk]
            self.total_cn = bool(self.v["total_cn"] and cjk)
            face = cjk[0] if self.total_cn else st.legend_face
            self.tf = fonts.get(face, int(max(12, avail * 0.028)), "regular")
            self.h += _text_h(self.tf) * 2.2

    def _slot(self, i: int, nrows: int, ncols: int) -> tuple[int, int]:
        if self.st.legend_colmajor:
            return i % nrows, i // nrows
        return i // ncols, i % ncols

    def _tcolor(self, rgb) -> tuple:
        return self.st.text_light if luminance(rgb) < self.st.lum_thr else self.st.text_dark

    def draw(self, d: ImageDraw.ImageDraw, x0: float, y0: float) -> None:
        st, pat, fonts = self.st, self.pat, self.fonts
        lay = st.legend
        line_rgb = st.thin_rgb if st.thin_w else (60, 60, 60)
        if lay == "table":
            u = self.u
            codef = _fit_font(fonts, st.legend_face, st.legend_weight, int(u * 0.5),
                              [pat.codes[k] for k in self.order], 4.6 * u)
            cntf = _fit_font(fonts, st.legend_count_face, st.legend_weight, int(u * 0.5),
                             [st.count_fmt.format(n=int(pat.counts[k])) for k in self.order], 0.8 * u)
            bw = max(1, int(round(u / 40)))
            for i, k in enumerate(self.order):
                r, c = self._slot(i, self.nrows, self.nc)
                x = x0 + 0.8 * u + c * (6 + self.gapu) * u
                y = y0 + r * u
                rgb = tuple(int(v) for v in pat.rgb[k])
                sw = [x, y, x + 5 * u, y + u]
                d.rectangle([round(sw[0]), round(sw[1]), round(sw[2]) - 1, round(sw[3]) - 1], fill=rgb)
                cb = [x + 5 * u, y, x + 6 * u, y + u]
                d.rectangle([round(cb[0]), round(cb[1]), round(cb[2]) - 1, round(cb[3]) - 1], fill=(255, 255, 255),
                            outline=line_rgb, width=bw)
                d.rectangle([round(sw[0]), round(sw[1]), round(cb[2]) - 1, round(cb[3]) - 1], outline=line_rgb, width=bw)
                code = pat.codes[k]
                cnt = st.count_fmt.format(n=int(pat.counts[k]))
                pc = ((sw[0] + sw[2]) / 2, (sw[1] + sw[3]) / 2)
                d.text(pc, code, font=codef, fill=self._tcolor(rgb), anchor="mm")
                pn = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
                d.text(pn, cnt, font=cntf, fill=(0, 0, 0), anchor="mm")
                self.items.append({"code": code, "count": int(pat.counts[k]), "swatch_box": sw,
                                   "code_box": list(d.textbbox(pc, code, font=codef, anchor="mm")), "code_text": code,
                                   "count_box": list(d.textbbox(pn, cnt, font=cntf, anchor="mm")), "count_text": cnt,
                                   "code_in_swatch": True})
            yb = y0 + self.nrows * u + u * 0.6
        elif lay in ("chips_in", "chips_under"):
            u = self.u
            for i, k in enumerate(self.order):
                r, c = i // self.per, i % self.per
                x = x0 + c * (u + self.gx)
                y = y0 + r * self.row_h
                rgb = tuple(int(v) for v in pat.rgb[k])
                sw = [x, y, x + u, y + u]
                rad = st.chip_radius * u
                outline = (205, 205, 205) if st.chip_border else None
                if rad >= u * 0.49:
                    d.ellipse([round(x), round(y), round(x + u) - 1, round(y + u) - 1], fill=rgb, outline=outline)
                else:
                    d.rounded_rectangle([round(x), round(y), round(x + u) - 1, round(y + u) - 1], radius=int(rad),
                                        fill=rgb, outline=outline)
                code = pat.codes[k]
                cnt = st.count_fmt.format(n=int(pat.counts[k]))
                if lay == "chips_in":
                    codef = _fit_font(fonts, st.legend_face, st.legend_weight, int(u * 0.4), [code], 0.86 * u)
                    pc = (x + u / 2, y + u / 2)
                    d.text(pc, code, font=codef, fill=self._tcolor(rgb), anchor="mm")
                    pn = (x + u / 2, y + u + u * 0.2)
                    cntf = _fit_font(fonts, st.legend_count_face, st.legend_weight, self.cf.size, [cnt], u + self.gx * 0.9)
                    d.text(pn, cnt, font=cntf, fill=st.text_dark, anchor="mt")
                else:
                    codef = _fit_font(fonts, st.legend_face, st.legend_weight, self.codef.size, [code], u + self.gx * 0.9)
                    pc = (x + u / 2, y + u + u * 0.15)
                    d.text(pc, code, font=codef, fill=st.text_dark, anchor="mt")
                    pn = (x + u / 2, pc[1] + _text_h(self.codef) * 1.5)
                    cntf = _fit_font(fonts, st.legend_count_face, st.legend_weight, self.cf.size, [cnt], u + self.gx * 0.9)
                    d.text(pn, cnt, font=cntf, fill=(110, 110, 110), anchor="mt")
                self.items.append({"code": code, "count": int(pat.counts[k]), "swatch_box": sw,
                                   "code_box": list(d.textbbox(pc, code, font=codef, anchor="mm" if lay == "chips_in" else "mt")),
                                   "code_text": code,
                                   "count_box": list(d.textbbox(pn, cnt, font=cntf, anchor="mt")), "count_text": cnt,
                                   "code_in_swatch": lay == "chips_in"})
            yb = y0 + self.nrows * self.row_h
        elif lay == "inline":
            u = self.u
            for i, k in enumerate(self.order):
                r, c = self._slot(i, self.nrows, self.nc)
                x = x0 + c * self.colw
                y = y0 + r * self.row_h
                rgb = tuple(int(v) for v in pat.rgb[k])
                s = u * 0.9
                sw = [x, y + (u - s) / 2, x + s, y + (u - s) / 2 + s]
                outline = (190, 190, 190) if (st.chip_border or luminance(rgb) > 240) else None
                if st.chip_radius >= 0.49:
                    d.ellipse([round(sw[0]), round(sw[1]), round(sw[2]) - 1, round(sw[3]) - 1], fill=rgb, outline=outline)
                else:
                    d.rounded_rectangle([round(sw[0]), round(sw[1]), round(sw[2]) - 1, round(sw[3]) - 1],
                                        radius=int(st.chip_radius * s), fill=rgb, outline=outline)
                code = pat.codes[k]
                cnt = st.count_fmt.format(n=int(pat.counts[k]))
                avail = self.colw - s - u * 0.5
                codef = _fit_font(fonts, st.legend_face, st.legend_weight, int(u * 0.62), [code], avail * 0.5)
                cntf = _fit_font(fonts, st.legend_count_face, st.legend_weight, codef.size, [cnt], avail * 0.45)
                pc = (sw[2] + u * 0.3, y + u / 2)
                d.text(pc, code, font=codef, fill=st.text_dark, anchor="lm")
                cb = d.textbbox(pc, code, font=codef, anchor="lm")
                pn = (cb[2] + u * 0.35, y + u / 2)
                d.text(pn, cnt, font=cntf, fill=(90, 90, 90), anchor="lm")
                self.items.append({"code": code, "count": int(pat.counts[k]), "swatch_box": sw,
                                   "code_box": list(cb), "code_text": code,
                                   "count_box": list(d.textbbox(pn, cnt, font=cntf, anchor="lm")), "count_text": cnt,
                                   "code_in_swatch": False})
            yb = y0 + self.nrows * self.row_h
        else:
            return
        if st.total_line:
            total = int(pat.counts.sum())
            t = f"共 {total} 颗  {self.n} 色" if self.total_cn else f"Total: {total}"
            pt = (x0, yb + _text_h(self.tf) * 0.6)
            d.text(pt, t, font=self.tf, fill=(60, 60, 60), anchor="lt")
            self.texts.append({"kind": "total", "text": t, "value": total,
                               "box": list(d.textbbox(pt, t, font=self.tf, anchor="lt"))})


# --------------------------------------------------------------------------- 主渲染

def render(pat: Pattern, st: Style, fonts: FontBook, rng: np.random.Generator,
           window: tuple[int, int, int, int] | None = None, show_grid: bool = True, show_legend: bool = True,
           show_title: bool = True, width_hint: int | None = None) -> Rendered:
    """渲染一页。window = (col0, row0, cols, rows)：分页时只画整张图纸的一部分（坐标和粗线按绝对位置）。
    show_grid=False 时只画标题和图例（分页图单独的图例页），宽度用 width_hint。"""
    c0, r0, cw, rh = window or (0, 0, pat.cols, pat.rows)
    sub = pat.idx[r0:r0 + rh, c0:c0 + cw]
    p = st.pitch
    m = st.margin
    lv = {"table_gap": float(rng.uniform(1.0, 3.0)), "chip_gap": float(rng.uniform(0.25, 0.6)),
          "total_cn": bool(rng.random() < 0.7), "title_center": bool(rng.random() < 0.6),
          "legend_gap": float(rng.uniform(0.4, 1.5))}

    # ---------------------------------------------------------------- 坐标带
    bands = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    sides = set(st.coord_sides) if (st.coords != "none" and show_grid) else set()
    cf = None
    if st.coords == "bar" and sides:
        nums = [str(i + st.coord_start) for i in range(max(pat.cols, pat.rows))]
        cf = _fit_font(fonts, st.coord_face, st.coord_weight, st.coord_size, nums, 0.9 * p)
        for s in sides:
            bands[s] = p
    elif st.coords == "small" and sides:
        cf = fonts.get(st.coord_face, st.coord_size, st.coord_weight)
        th = _text_h(cf)
        mw = max(cf.getlength(str(c0 + cw - 1 + st.coord_start)), cf.getlength(str(r0 + rh - 1 + st.coord_start)))
        for s in sides:
            bands[s] = int(math.ceil((th if s in ("top", "bottom") else mw) + st.coord_gap + 2))
    pad = max(st.thin_w, st.bold_w if st.bold_period else 0, st.border_w) + 1

    # ---------------------------------------------------------------- 标题
    title_h = 0
    tf = None
    if st.title and show_title:
        est_w = cw * p + 2 * m + bands["left"] + bands["right"]
        tf = _fit_font(fonts, st.title_face, st.title_weight, int(max(14, est_w * 0.045)), [st.title], est_w * 0.9)
        title_h = int(_text_h(tf) * 2.0)

    gx = m + bands["left"] + pad if show_grid else m
    gy = m + title_h + bands["top"] + pad if show_grid else m + title_h
    gw, gh = (cw * p, rh * p) if show_grid else (0, 0)
    W = gx + gw + (pad + bands["right"] + m if show_grid else m)
    if not show_grid:
        W = max(W, int(width_hint or 1000))
    W = max(W, 2 * m + 120)

    # ---------------------------------------------------------------- 图例尺寸
    leg = None
    leg_h = 0
    if show_legend and st.legend != "none":
        leg = _Legend(pat, st, fonts, W - 2 * m, lv)
        leg_h = int(math.ceil(leg.h))
    grid_bottom = gy + gh + (pad + bands["bottom"] if show_grid else 0)
    sep = int(max(8, p * lv["legend_gap"])) if leg_h else 0
    H = grid_bottom + sep + leg_h + m

    canvas = np.empty((H, W, 3), dtype=np.uint8)
    canvas[:] = np.array(st.bg_rgb, dtype=np.uint8)
    ann = Ann(width=W, height=H)
    has_text = text_white = None
    text_boxes: dict = {}

    # 图例区域的浅色背景网格（0_2 那种）
    if leg is not None and st.legend_bg_grid:
        step = int(round(leg.u)) if st.legend == "table" else p
        gcol = np.array(_mix(st.bg_rgb, (205, 205, 205), 0.6), dtype=np.uint8)
        ys0 = grid_bottom
        xs = np.arange(m, W - m + 1, step)
        ys = np.arange(ys0, H - m + 1, step)
        for x in xs:
            canvas[ys0:H - m, x:x + 1] = gcol
        for y in ys:
            canvas[y:y + 1, m:W - m] = gcol

    if show_grid:
        colors = np.vstack([pat.rgb.astype(np.uint8), np.array(pat.empty_rgb, dtype=np.uint8)[None]])
        k_empty = len(pat.codes)
        sidx = np.where(sub >= 0, sub, k_empty)
        block = np.repeat(np.repeat(colors[sidx], p, axis=0), p, axis=1)
        if st.cell_shape == "bead":
            yy, xx = np.mgrid[0:p, 0:p]
            rr = (yy + 0.5 - p / 2) ** 2 + (xx + 0.5 - p / 2) ** 2
            circ = rr <= (p * 0.45) ** 2
            hole = rr <= (p * 0.13) ** 2
            mask = np.tile(circ & ~hole, (rh, cw))
            bgc = np.array(st.bg_rgb, dtype=np.uint8)
            block = np.where(mask[..., None], block, bgc)
        # 格内文字
        has_text = (sub >= 0) & st.cell_text
        lum = np.array([luminance(c) for c in pat.rgb]) if len(pat.codes) else np.zeros(0)
        white_k = lum < st.lum_thr
        text_white = has_text & np.where(sub >= 0, white_k[np.maximum(sub, 0)], False)
        if st.cell_text and has_text.any():
            used = sorted(int(k) for k in np.unique(sub[sub >= 0]))
            all_codes = [pat.codes[k] for k in range(len(pat.codes)) if pat.counts[k] > 0]
            if st.text_fit == "longest":
                font = _fit_font(fonts, st.text_face, st.text_weight, st.text_size, all_codes, 0.9 * p)
                fonts_k = {k: font for k in used}
            else:
                fonts_k = {k: _fit_font(fonts, st.text_face, st.text_weight, st.text_size, [pat.codes[k]], 0.9 * p)
                           for k in used}
            patches = np.zeros((k_empty + 1, p, p), dtype=np.uint8)
            for k in used:
                a, box = _glyph(pat.codes[k], fonts_k[k], p, st.text_dy)
                patches[k] = a
                if box is not None:
                    text_boxes[pat.codes[k]] = [round(v, 4) for v in box]
            A = patches[sidx].transpose(0, 2, 1, 3).reshape(rh * p, cw * p)
            tcol = np.where(white_k[:, None], np.array(st.text_light)[None], np.array(st.text_dark)[None])
            tcol = np.vstack([tcol, np.zeros((1, 3))]).astype(np.uint16)
            T = np.repeat(np.repeat(tcol[sidx], p, axis=0), p, axis=1)
            a16 = A.astype(np.uint16)[..., None]
            block = ((block.astype(np.uint16) * (255 - a16) + T * a16 + 127) // 255).astype(np.uint8)
        canvas[gy:gy + gh, gx:gx + gw] = block

        # 坐标栏底色
        if st.coords == "bar":
            bar = np.array(st.bar_rgb, dtype=np.uint8)
            if "top" in sides:
                canvas[gy - p:gy, gx:gx + gw] = bar
            if "bottom" in sides:
                canvas[gy + gh:gy + gh + p, gx:gx + gw] = bar
            if "left" in sides:
                canvas[gy:gy + gh, gx - p:gx] = bar
            if "right" in sides:
                canvas[gy:gy + gh, gx + gw:gx + gw + p] = bar

        # 网格线
        x_lines, y_lines = _draw_lines(canvas, st, gx, gy, cw, rh, c0, r0, bands if st.coords == "bar" else None)
        ann.pages.append({"image_index": 0, "x_lines": x_lines, "y_lines": y_lines, "cols": cw, "rows": rh,
                          "col_offset": c0, "row_offset": r0, "H": np.eye(3)})
        ann.regions.append({"kind": "grid", "box": [x_lines[0], y_lines[0], x_lines[-1], y_lines[-1]]})

    img = Image.fromarray(canvas)
    d = ImageDraw.Draw(img)

    # ---------------------------------------------------------------- 坐标文字
    if show_grid and sides:
        for side in ("top", "bottom", "left", "right"):
            if side not in sides:
                continue
            n = cw if side in ("top", "bottom") else rh
            off = c0 if side in ("top", "bottom") else r0
            boxes = []
            for i in range(n):
                val = off + i + st.coord_start
                if st.coord_every > 1 and not (val % st.coord_every == 0 or val == 1):
                    continue
                t = str(val)
                if side in ("top", "bottom"):
                    x = gx + (i + 0.5) * p
                    if st.coords == "bar":
                        y = gy - p / 2 if side == "top" else gy + gh + p / 2
                        anchor = "mm"
                    else:
                        y = gy - pad - st.coord_gap if side == "top" else gy + gh + pad + st.coord_gap
                        anchor = "ms" if side == "top" else "mt"
                else:
                    y = gy + (i + 0.5) * p
                    if st.coords == "bar":
                        x = gx - p / 2 if side == "left" else gx + gw + p / 2
                        anchor = "mm"
                    else:
                        x = gx - pad - st.coord_gap if side == "left" else gx + gw + pad + st.coord_gap
                        anchor = "rm" if side == "left" else "lm"
                d.text((x, y), t, font=cf, fill=st.coord_rgb, anchor=anchor)
                b = list(d.textbbox((x, y), t, font=cf, anchor=anchor))
                boxes.append(b)
                ann.coords.append({"axis": "col" if side in ("top", "bottom") else "row", "index": off + i,
                                   "text": t, "side": side, "box": b})
            if boxes:
                bb = np.array(boxes)
                ann.regions.append({"kind": f"coords_{side}",
                                    "box": [float(bb[:, 0].min()), float(bb[:, 1].min()), float(bb[:, 2].max()),
                                            float(bb[:, 3].max())]})

    # ---------------------------------------------------------------- 标题
    if tf is not None:
        if lv["title_center"]:
            pt, anchor = (W / 2, m + title_h * 0.45), "mm"
        else:
            pt, anchor = (m, m + title_h * 0.45), "lm"
        d.text(pt, st.title, font=tf, fill=(40, 40, 40), anchor=anchor)
        b = list(d.textbbox(pt, st.title, font=tf, anchor=anchor))
        ann.texts.append({"kind": "title", "text": st.title, "box": b})
        ann.regions.append({"kind": "title", "box": b})

    # ---------------------------------------------------------------- 图例
    if leg is not None:
        y_leg = grid_bottom + sep
        if st.legend_sep and show_grid:
            ysep = grid_bottom + sep // 2
            d.line([(m, ysep), (W - m, ysep)], fill=(225, 225, 225), width=max(1, p // 12))
        leg.draw(d, m, y_leg)
        ann.legend.extend(leg.items)
        ann.texts.extend(leg.texts)
        if leg.items:
            bb = np.array([it["swatch_box"] + it["count_box"] for it in leg.items], dtype=np.float64)
            ann.regions.append({"kind": "legend", "box": [float(min(bb[:, 0].min(), bb[:, 4].min())),
                                                          float(min(bb[:, 1].min(), bb[:, 5].min())),
                                                          float(max(bb[:, 2].max(), bb[:, 6].max())),
                                                          float(max(bb[:, 3].max(), bb[:, 7].max()))]})

    return Rendered(image=np.asarray(img, dtype=np.uint8).copy(), ann=ann, text_boxes=text_boxes,
                    has_text=has_text, text_white=text_white)


def _mix(a, b, t):
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def _draw_lines(canvas: np.ndarray, st: Style, gx: int, gy: int, cw: int, rh: int, c0: int, r0: int,
                bar_bands: dict | None) -> tuple[list[float], list[float]]:
    """画网格线，返回每条线的中心坐标。先画细线，再画粗线，最后画外框（后画的覆盖先画的）。"""
    p = st.pitch
    H, W = canvas.shape[:2]

    def plan(n, off):
        out = []
        for i in range(n + 1):
            if i in (0, n) and st.border_w:
                out.append((st.border_w, st.border_rgb, 2))
            elif st.bold_period and (off + i) % st.bold_period == 0 and st.bold_w:
                out.append((st.bold_w, st.bold_rgb, 1))
            elif st.thin_w:
                out.append((st.thin_w, st.thin_rgb, 0))
            else:
                out.append((0, None, 0))
        return out

    px, py = plan(cw, c0), plan(rh, r0)
    # 线的延伸范围：彩色坐标栏样式下穿过坐标栏
    ext = bar_bands or {"top": 0, "bottom": 0, "left": 0, "right": 0}
    tmax = max([t for t, _, _ in px + py] + [0])
    ya, yb = gy - ext["top"] - tmax // 2, gy + rh * p + ext["bottom"] + (tmax - tmax // 2)
    xa, xb = gx - ext["left"] - tmax // 2, gx + cw * p + ext["right"] + (tmax - tmax // 2)
    ya, xa = max(0, ya), max(0, xa)
    x_lines, y_lines = [], []
    for i, (t, _, _) in enumerate(px):
        x = gx + i * p
        x_lines.append(float(x - t // 2 + t / 2.0) if t else float(x))
    for j, (t, _, _) in enumerate(py):
        y = gy + j * p
        y_lines.append(float(y - t // 2 + t / 2.0) if t else float(y))
    for level in (0, 1, 2):
        for i, (t, rgb, lv) in enumerate(px):
            if t and lv == level:
                x = gx + i * p - t // 2
                canvas[ya:yb, x:x + t] = rgb
        for j, (t, rgb, lv) in enumerate(py):
            if t and lv == level:
                y = gy + j * p - t // 2
                canvas[y:y + t, xa:xb] = rgb
    # 彩色坐标栏：坐标栏外沿也画线（0_2 的表格效果）
    if bar_bands:
        t, rgb = st.thin_w or 1, st.thin_rgb
        for s, v in bar_bands.items():
            if not v:
                continue
            if s == "top":
                canvas[gy - p - t // 2:gy - p - t // 2 + t, xa:xb] = rgb
            elif s == "bottom":
                canvas[gy + rh * p + p - t // 2:gy + rh * p + p - t // 2 + t, xa:xb] = rgb
            elif s == "left":
                canvas[ya:yb, gx - p - t // 2:gx - p - t // 2 + t] = rgb
            elif s == "right":
                canvas[ya:yb, gx + cw * p + p - t // 2:gx + cw * p + p - t // 2 + t] = rgb
    return x_lines, y_lines
