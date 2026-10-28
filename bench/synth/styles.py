"""渲染风格：模仿主流 App 的导出图（04 §2.3「渲染」），所有参数都随机化。

| 风格 | 模仿对象 | 要点 |
|---|---|---|
| `table_cyan` | data/samples/0_2 | 大格子、深色细线、四周彩色坐标栏（同格距）、粗体坐标数字、两列表格式图例 |
| `fine5` | data/samples/3_0 | 浅灰细网格、每 5 格一条彩色粗线、四周小号坐标、横排圆角色块图例（色块里写色号，下方 x颗数） |
| `fine10` | 同上的变体 | 每 10 格粗线，坐标可能只标 5/10 的倍数，图例布局随机 |
| `nocoord` | data/samples/0_0 | 没有坐标；粗线可有可无；图例可有可无 |
| `blocks` | T2 | 格内没有文字；方格或圆形豆子；图例可有可无（没有图例时只用 native 色卡） |
| `pixel` | T3 | 没有网格、没有图例、没有文字的像素画 |

每个风格都会随机：字体、字号、线色、粗线颜色和间隔、文字黑白阈值、坐标样式、图例布局和排序、标题。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench.synth.fonts import FontBook, FontFace

STYLE_NAMES = ("table_cyan", "fine5", "fine10", "nocoord", "blocks", "pixel")
TEXT_STYLES = ("table_cyan", "fine5", "fine10", "nocoord")
TYPE_STYLES = {
    "T1": TEXT_STYLES,
    "T2": ("blocks",),
    "T3": ("pixel",),
    "T4": TEXT_STYLES + ("blocks",),
    "T5": TEXT_STYLES + ("blocks",),
    "T6": TEXT_STYLES + ("blocks",),
    "T7": TEXT_STYLES + ("blocks",),
}
LEGEND_LAYOUTS = ("table", "chips_in", "chips_under", "inline")
COUNT_FORMATS = {"x{n}": 0.34, "×{n}": 0.14, "{n}": 0.26, "{n}颗": 0.14, "({n})": 0.12}

_BAR_COLORS = [(168, 232, 224), (176, 222, 246), (200, 238, 200), (250, 238, 186), (250, 214, 226), (226, 226, 226),
               (214, 206, 246)]
_BOLD_COLORS = [(180, 168, 240), (150, 140, 230), (120, 120, 120), (40, 40, 40), (232, 120, 120), (110, 170, 240),
                (90, 190, 150), (250, 160, 80), (200, 120, 200)]

_ADJ = ["小", "可爱", "奶呼呼", "复古", "像素", "软萌", "迷你", "圆滚滚", "治愈", "胖胖", "彩色", "简单"]
_NOUN = ["猫咪", "小熊", "兔子", "蘑菇", "草莓", "星星", "小鸭", "恐龙", "花朵", "小狗", "企鹅", "西瓜", "云朵",
         "小鱼", "仙人掌", "糖果", "爱心", "火箭", "房子", "小鸡", "青蛙", "苹果"]
_SUFFIX = ["拼豆图纸", "图纸", "像素图", "拼豆", "豆豆图纸", "图案"]


@dataclass
class Style:
    name: str
    cell_text: bool = True
    cell_shape: str = "square"            # square / bead
    pitch: int = 20
    bg_rgb: tuple = (255, 255, 255)
    empty_rgb: tuple = (255, 255, 255)
    # 网格线
    thin_w: int = 1
    thin_rgb: tuple = (200, 200, 200)
    bold_period: int = 5
    bold_w: int = 2
    bold_rgb: tuple = (180, 168, 240)
    border_w: int = 1
    border_rgb: tuple = (200, 200, 200)
    # 格内文字
    text_face: FontFace | None = None
    text_weight: str = "regular"
    text_size: int = 8
    text_fit: str = "longest"             # longest：统一字号；each：长色号单独缩小
    text_dark: tuple = (0, 0, 0)
    text_light: tuple = (255, 255, 255)
    lum_thr: float = 140.0
    text_dy: float = 0.0
    # 坐标
    coords: str = "small"                 # none / small / bar
    coord_sides: tuple = ("top", "bottom", "left", "right")
    coord_every: int = 1
    coord_start: int = 1
    coord_face: FontFace | None = None
    coord_weight: str = "regular"
    coord_size: int = 8
    coord_rgb: tuple = (120, 120, 120)
    coord_gap: int = 4
    bar_rgb: tuple = (168, 232, 224)
    # 图例
    legend: str = "chips_in"              # none / table / chips_in / chips_under / inline
    legend_cols: int = 2
    legend_face: FontFace | None = None
    legend_weight: str = "regular"
    legend_count_face: FontFace | None = None
    count_fmt: str = "x{n}"
    legend_order: str = "code_str"        # first_seen / code_str / code_nat / count_desc
    legend_colmajor: bool = True
    chip_radius: float = 0.2
    chip_border: bool = False
    legend_bg_grid: bool = False
    legend_sep: bool = True
    legend_unit_frac: float = 0.055       # 图例单元相对画布宽度
    total_line: bool = False
    # 标题
    title: str = ""
    title_face: FontFace | None = None
    title_weight: str = "regular"
    margin: int = 20
    extra: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """写进附加标注的风格参数（字体只写 id，便于统计字体分布）。"""
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, FontFace):
                out[k] = v.id
            elif v is None or isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, tuple):
                out[k] = list(v)
        return out


def _choice(rng, items, p=None):
    items = list(items)
    if p is None:
        return items[int(rng.integers(len(items)))]
    p = np.asarray(p, dtype=np.float64)
    return items[int(rng.choice(len(items), p=p / p.sum()))]


def _jitter(rng, rgb, amt=12):
    return tuple(int(np.clip(c + rng.integers(-amt, amt + 1), 0, 255)) for c in rgb)


def _gray(rng, lo, hi):
    g = int(rng.integers(lo, hi + 1))
    return (g, g, g)


def random_title(rng, cols: int, rows: int) -> str:
    s = _choice(rng, _ADJ) + _choice(rng, _NOUN)
    if rng.random() < 0.7:
        s += " " + _choice(rng, _SUFFIX)
    if rng.random() < 0.5:
        s += f" {cols}×{rows}"
    return s


def sample_style(rng: np.random.Generator, name: str, cols: int, rows: int, fonts: FontBook,
                 overrides: dict | None = None) -> Style:
    """按风格名抽一组渲染参数。overrides 用于测试（直接覆盖同名字段）。"""
    st = Style(name=name)
    long_side = max(cols, rows)
    # ---------------------------------------------------------------- 格距（渲染分辨率）
    if name == "pixel":
        target = float(np.exp(rng.uniform(np.log(500), np.log(2000))))
        st.pitch = int(max(2, round(target / max(cols, rows))))
    elif name == "table_cyan":
        target = float(np.exp(rng.uniform(np.log(1000), np.log(2600))))
        st.pitch = int(np.clip(round(target / (cols + 2.4)), 12, 72))
    else:
        target = float(np.exp(rng.uniform(np.log(900), np.log(3200))))
        st.pitch = int(np.clip(round(target / (cols + 3)), 10 if name != "blocks" else 6, 56))
    p = st.pitch
    st.margin = int(max(6, p * rng.uniform(0.3, 1.2)))
    st.bg_rgb = (255, 255, 255) if rng.random() < 0.8 else _jitter(rng, (250, 248, 244), 6)
    st.empty_rgb = st.bg_rgb if rng.random() < 0.85 else (255, 255, 255)

    # ---------------------------------------------------------------- 网格线
    if name == "table_cyan":
        st.thin_w = 1 if p < 30 else int(rng.integers(1, 3))
        st.thin_rgb = _gray(rng, 20, 90)
        st.bold_period = int(_choice(rng, [0, 5, 10], [0.7, 0.2, 0.1]))
        st.bold_w = st.thin_w + int(rng.integers(1, 3))
        st.bold_rgb = _choice(rng, [(0, 0, 0), (60, 60, 60), _choice(rng, _BOLD_COLORS)])
        st.border_w, st.border_rgb = st.thin_w, st.thin_rgb
    elif name in ("fine5", "fine10", "nocoord", "blocks"):
        st.thin_w = 1 if p < 36 else int(rng.integers(1, 3))
        st.thin_rgb = _gray(rng, 175, 225) if rng.random() < 0.8 else _gray(rng, 90, 170)
        st.bold_period = {"fine5": 5, "fine10": 10}.get(name, int(_choice(rng, [0, 5, 10], [0.4, 0.4, 0.2])))
        st.bold_w = st.thin_w + int(rng.integers(1, 3))
        st.bold_rgb = _jitter(rng, _choice(rng, _BOLD_COLORS), 15)
        if rng.random() < 0.5:
            st.border_w, st.border_rgb = st.bold_w, st.bold_rgb
        else:
            st.border_w, st.border_rgb = st.thin_w, st.thin_rgb
    if name == "blocks":
        st.cell_text = False
        st.cell_shape = "bead" if rng.random() < 0.25 else "square"
        if st.cell_shape == "bead" and rng.random() < 0.6:
            st.thin_w, st.bold_period, st.border_w = 0, 0, 0
    if name == "pixel":
        st.cell_text = False
        st.thin_w, st.bold_period, st.border_w = 0, 0, 0
        st.margin = int(rng.integers(0, max(1, p * 4)))
        if rng.random() < 0.3:
            st.bg_rgb = st.empty_rgb = _jitter(rng, _choice(rng, [(250, 250, 250), (235, 240, 250), (255, 248, 235)]), 8)

    # ---------------------------------------------------------------- 格内文字
    st.text_face = fonts.pick(rng, "cell")
    st.text_weight = fonts.pick_weight(rng, st.text_face, ("regular", "medium", "semibold", "bold"))
    st.text_size = int(max(4, round(p * rng.uniform(0.34, 0.54))))
    st.text_fit = _choice(rng, ["longest", "each"], [0.6, 0.4])
    st.text_dark = (0, 0, 0) if rng.random() < 0.6 else _gray(rng, 25, 80)
    st.text_light = (255, 255, 255) if rng.random() < 0.85 else _gray(rng, 230, 250)
    st.lum_thr = float(rng.uniform(105, 165))
    st.text_dy = float(rng.uniform(-0.04, 0.06)) * p

    # ---------------------------------------------------------------- 坐标
    if name == "table_cyan":
        st.coords = "bar"
        st.bar_rgb = _jitter(rng, _choice(rng, _BAR_COLORS, [0.5, 0.1, 0.1, 0.08, 0.08, 0.08, 0.06]), 8)
        st.coord_sides = ("top", "bottom", "left", "right") if rng.random() < 0.7 else ("top", "left")
        st.coord_rgb = (0, 0, 0) if rng.random() < 0.8 else _gray(rng, 30, 80)
        st.coord_size = int(max(6, round(p * rng.uniform(0.4, 0.56))))
    elif name in ("fine5", "fine10") or (name == "blocks" and rng.random() < 0.5):
        st.coords = "small"
        r = rng.random()
        st.coord_sides = (("top", "bottom", "left", "right") if r < 0.6 else ("top", "left") if r < 0.9
                          else ("bottom", "right"))
        st.coord_rgb = _gray(rng, 90, 170) if rng.random() < 0.7 else _gray(rng, 0, 60)
        st.coord_size = int(max(6, round(p * rng.uniform(0.38, 0.62))))
        st.coord_gap = int(max(2, p * rng.uniform(0.15, 0.6)))
    else:
        st.coords = "none"
    st.coord_every = 1 if name != "fine10" else int(_choice(rng, [1, 5, 10], [0.5, 0.25, 0.25]))
    if long_side > 120 and st.coords == "small" and rng.random() < 0.3:
        st.coord_every = 5
    st.coord_start = 1 if rng.random() < 0.93 else 0
    st.coord_face = fonts.pick(rng, "coord")
    st.coord_weight = ("bold" if name == "table_cyan" and rng.random() < 0.8
                       else fonts.pick_weight(rng, st.coord_face, ("regular", "medium")))
    if st.coord_face is not None and st.coord_weight not in st.coord_face.weights:
        st.coord_weight = st.coord_face.weights[-1]

    # ---------------------------------------------------------------- 图例
    if name == "pixel":
        st.legend = "none"
    elif name == "table_cyan":
        st.legend = _choice(rng, LEGEND_LAYOUTS, [0.7, 0.1, 0.1, 0.1])
    elif name == "fine5":
        st.legend = _choice(rng, LEGEND_LAYOUTS, [0.1, 0.6, 0.15, 0.15])
    elif name == "blocks":
        st.legend = _choice(rng, ("none",) + LEGEND_LAYOUTS, [0.4, 0.15, 0.15, 0.15, 0.15])
    elif name == "nocoord":
        st.legend = _choice(rng, ("none",) + LEGEND_LAYOUTS, [0.25, 0.2, 0.25, 0.15, 0.15])
    else:
        st.legend = _choice(rng, LEGEND_LAYOUTS, [0.2, 0.4, 0.2, 0.2])
    st.legend_cols = int(_choice(rng, [1, 2, 3], [0.15, 0.6, 0.25])) if st.legend == "table" else int(rng.integers(2, 6))
    st.legend_face = fonts.pick(rng, "legend")
    st.legend_weight = fonts.pick_weight(rng, st.legend_face, ("regular", "medium"))
    st.count_fmt = (_choice(rng, ["{n}", "x{n}"], [0.8, 0.2]) if st.legend == "table"
                    else _choice(rng, list(COUNT_FORMATS), list(COUNT_FORMATS.values())))
    st.legend_count_face = st.legend_face
    if "颗" in st.count_fmt:
        cjk = [f for f in fonts.candidates("legend") if f.cjk]
        st.legend_count_face = cjk[int(rng.integers(len(cjk)))] if cjk else st.legend_face
        if not cjk and not fonts.fallback:
            st.count_fmt = "x{n}"
        if fonts.fallback:
            st.count_fmt = "x{n}"
    st.legend_order = _choice(rng, ["first_seen", "code_str", "code_nat", "count_desc"], [0.3, 0.3, 0.25, 0.15])
    st.legend_colmajor = rng.random() < 0.6
    st.chip_radius = float(_choice(rng, [0.0, rng.uniform(0.08, 0.3), 0.5], [0.25, 0.6, 0.15]))
    st.chip_border = rng.random() < 0.3
    st.legend_bg_grid = name == "table_cyan" and rng.random() < 0.7
    st.legend_sep = rng.random() < 0.5
    st.legend_unit_frac = float(rng.uniform(0.04, 0.07))
    st.total_line = st.legend != "none" and rng.random() < 0.3

    # ---------------------------------------------------------------- 标题
    if rng.random() < (0.3 if name != "pixel" else 0.1):
        st.title = random_title(rng, cols, rows)
        cands = [f for f in fonts.candidates("title")]
        st.title_face = cands[int(rng.integers(len(cands)))] if cands else None
        st.title_weight = fonts.pick_weight(rng, st.title_face, ("regular", "medium", "bold"))
        if fonts.fallback:
            st.title = f"PATTERN {cols}x{rows}"
    if fonts.fallback:
        st.count_fmt = st.count_fmt.replace("颗", "").replace("×", "x")

    for k, v in (overrides or {}).items():
        setattr(st, k, v)
    return st
