"""底图生成：全部程序生成（自有数据，05 §7 登记为「程序生成」），不读取任何外部图片。

输出 `BaseMap`：
- `idx`：(rows, cols) 的 int16，-1 表示空格，k ≥ 0 指向 `roles[k]`；
- `roles`：每种颜色的「语义角色」（轮廓、填充、渐变、白豆、相近色……），由 `colors.assign_colors` 映射到具体色值；
- `tags`：本图触发的难点标签（04 §2.1：similar_colors / single_pixel_detail / gradient_skin / white_vs_empty）。

构图种类：character（卡通角色：头、身体、耳朵、眼睛、腮红、粗黑轮廓，最接近真实图纸）、sprite（对称精灵）、
multi（多个小物件，类似 0_0 的合集排版）、shapes（带轮廓的几何形状）、
scene（铺满背景的场景）、blocks（色块拼接）、noise（随机噪声纹理）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

KINDS = {"character": 0.42, "sprite": 0.16, "multi": 0.18, "shapes": 0.1, "scene": 0.06, "blocks": 0.04, "noise": 0.04}
MAX_ROLES = 56

_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float64) / 16.0 - 0.47


@dataclass
class Role:
    kind: str                 # outline / eye / highlight / white / similar / ramp / bg / fill / noise
    group: int = -1           # ramp、similar 的组号
    order: int = 0            # ramp 组内顺序（0 最暗）
    ramp: str = ""            # ramp 类型：skin / hue
    size: int = 0             # ramp 组的大小


@dataclass
class BaseMap:
    idx: np.ndarray
    roles: list[Role]
    tags: set[str] = field(default_factory=set)
    kind: str = ""

    @property
    def rows(self) -> int:
        return int(self.idx.shape[0])

    @property
    def cols(self) -> int:
        return int(self.idx.shape[1])


class _B:
    """构图时的小工具：分配角色、记录标签。"""

    def __init__(self, rows: int, cols: int, rng: np.random.Generator):
        self.rows, self.cols, self.rng = rows, cols, rng
        self.idx = np.full((rows, cols), -1, dtype=np.int16)
        self.roles: list[Role] = []
        self.tags: set[str] = set()
        self._groups = 0
        self.shared_outline: int | None = None

    def role(self, kind: str, **kw) -> int:
        if len(self.roles) >= MAX_ROLES:
            # 颜色太多时复用已有的同类角色（真实图纸通常 10–40 色）
            same = [i for i, r in enumerate(self.roles) if r.kind == kind] or list(range(len(self.roles)))
            return same[int(self.rng.integers(len(same)))]
        self.roles.append(Role(kind=kind, **kw))
        return len(self.roles) - 1

    def fill_role(self) -> int:
        """新填充色；颜色多了以后越来越倾向复用。"""
        fills = [i for i, r in enumerate(self.roles) if r.kind == "fill"]
        if fills and self.rng.random() < min(0.85, len(self.roles) / 60.0):
            return fills[int(self.rng.integers(len(fills)))]
        return self.role("fill")

    def outline_role(self) -> int:
        if self.shared_outline is None or self.rng.random() < 0.15:
            r = self.role("outline")
            if self.shared_outline is None:
                self.shared_outline = r
            return r
        return self.shared_outline

    def group(self) -> int:
        self._groups += 1
        return self._groups - 1

    def ramp_roles(self, n: int, ramp: str) -> list[int]:
        g = self.group()
        return [self.role("ramp", group=g, order=i, ramp=ramp, size=n) for i in range(n)]

    def similar_roles(self, n: int) -> list[int]:
        g = self.group()
        return [self.role("similar", group=g, order=i, size=n) for i in range(n)]

    def finish(self, kind: str) -> BaseMap:
        """去掉没用到的角色、重新编号。"""
        used = np.unique(self.idx[self.idx >= 0])
        remap = np.full(len(self.roles) + 1, -1, dtype=np.int16)
        remap[used] = np.arange(len(used), dtype=np.int16)
        idx = np.where(self.idx >= 0, remap[np.maximum(self.idx, 0)], -1).astype(np.int16)
        roles = [self.roles[i] for i in used]
        # 组内 order 重排成连续值
        groups: dict[int, list[Role]] = {}
        for r in roles:
            if r.group >= 0:
                groups.setdefault(r.group, []).append(r)
        for g, rs in groups.items():
            rs.sort(key=lambda r: r.order)
            for i, r in enumerate(rs):
                r.order, r.size = i, len(rs)
        tags = set(self.tags)
        kinds = {r.kind for r in roles}
        if "white" in kinds and (idx < 0).any():
            tags.add("white_vs_empty")
        else:
            tags.discard("white_vs_empty")
        if not any(r.kind == "ramp" and r.ramp == "skin" for r in roles):
            tags.discard("gradient_skin")
        if not any(len(v) >= 2 for g, v in groups.items() if v[0].kind == "similar"):
            tags.discard("similar_colors")
        return BaseMap(idx=idx, roles=roles, tags=tags, kind=kind)


# --------------------------------------------------------------------------- 形状

def _local(r0: int, c0: int, h: int, w: int):
    """框内每个格子中心的归一化坐标 u（左右）、v（上下），范围约 [-1, 1]。"""
    v = (np.arange(h) + 0.5) / h * 2 - 1
    u = (np.arange(w) + 0.5) / w * 2 - 1
    return np.meshgrid(u, v)


def _blob_mask(rng, h: int, w: int, symmetric: bool) -> np.ndarray:
    u, v = _local(0, 0, h, w)
    for _ in range(6):
        n = int(rng.integers(3, 8))
        cx = rng.uniform(0.0 if symmetric else -0.6, 0.6, n)
        cy = rng.uniform(-0.6, 0.6, n)
        rad = rng.uniform(0.25, 0.6, n)
        f = np.zeros_like(u)
        for i in range(n):
            f += rad[i] ** 2 / ((u - cx[i]) ** 2 + (v - cy[i]) ** 2 + 1e-3)
            if symmetric:
                f += rad[i] ** 2 / ((u + cx[i]) ** 2 + (v - cy[i]) ** 2 + 1e-3)
        m = f >= (2.2 if symmetric else 1.2)
        m &= (np.abs(u) < 0.98) & (np.abs(v) < 0.98)
        lab, nl = ndimage.label(m)
        if nl:
            sizes = ndimage.sum(m, lab, range(1, nl + 1))
            m = lab == (1 + int(np.argmax(sizes)))
            if m.sum() >= max(6, 0.15 * h * w):
                if symmetric:
                    m = m | m[:, ::-1]
                return m
    return u ** 2 + v ** 2 <= 0.85


def _shape_mask(rng, h: int, w: int, shape: str) -> np.ndarray:
    u, v = _local(0, 0, h, w)
    if shape == "ellipse":
        return u ** 2 + v ** 2 <= 0.92
    if shape == "rrect":
        p = rng.uniform(3, 8)
        return np.abs(u / 0.95) ** p + np.abs(v / 0.95) ** p <= 1
    if shape == "diamond":
        return np.abs(u) + np.abs(v) <= 0.97
    if shape == "heart":
        x, y = u * 1.25, -v * 1.25 + 0.15
        return (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3 <= 0
    if shape == "star":
        k = int(rng.choice([5, 5, 6, 4]))
        r = np.hypot(u, v)
        th = np.arctan2(u, -v)
        rr = 0.62 + 0.33 * np.cos(k * th)
        return r <= rr
    if shape == "triangle":
        return (v <= 0.95) & (v >= -0.95 + 1.9 * np.abs(u) * 1.0 - 0.0) & (np.abs(u) <= 0.97)
    return _blob_mask(rng, h, w, True)


def _outline(mask: np.ndarray, thick: int = 1) -> np.ndarray:
    inner = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1), iterations=thick,
                                   border_value=0)
    return mask & ~inner


def _partition(rng, region: np.ndarray, n: int, symmetric: bool) -> np.ndarray:
    """把 region 分成 n 块（Voronoi，对称时按 |u| 计算），返回 (h, w) 的块号，region 外为 -1。"""
    h, w = region.shape
    u, v = _local(0, 0, h, w)
    if symmetric:
        u = np.abs(u)
    sx = rng.uniform(0 if symmetric else -1, 1, n)
    sy = rng.uniform(-1, 1, n)
    if rng.random() < 0.4:  # 横向分带（头 / 身 / 腿）
        sx = np.zeros(n)
        sy = np.sort(rng.uniform(-1, 1, n))
    d = (u[..., None] - sx) ** 2 + (v[..., None] - sy) ** 2
    lab = np.argmin(d, axis=-1).astype(np.int16)
    return np.where(region, lab, -1)


def _ramp_levels(rng, region: np.ndarray, n: int, dither: bool) -> np.ndarray:
    h, w = region.shape
    u, v = _local(0, 0, h, w)
    ang = rng.uniform(-0.6, 0.6)
    t = (v * np.cos(ang) + u * np.sin(ang))
    vals = t[region]
    if vals.size == 0:
        return np.zeros((h, w), dtype=np.int16)
    lo, hi = float(vals.min()), float(vals.max())
    t = (t - lo) / max(hi - lo, 1e-6)
    if rng.random() < 0.5:
        t = 1 - t
    if dither:
        by = np.tile(_BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        t = t + by / n
    lev = np.clip(np.floor(t * n), 0, n - 1).astype(np.int16)
    return lev


def _shade(rng, view: np.ndarray, reg: np.ndarray, roles: list[int]) -> None:
    """区域内按光照方向分出阴影带（roles[0]，最暗）、本色和高光带（roles[-1]）。"""
    n = len(roles)
    dist = ndimage.distance_transform_cdt(np.pad(reg, 1), metric="taxicab")[1:-1, 1:-1]
    h, w = reg.shape
    u, v = _local(0, 0, h, w)
    ang = rng.uniform(0, 2 * np.pi)
    side = u * np.cos(ang) + v * np.sin(ang)
    width = int(rng.integers(1, 3)) if min(h, w) < 40 else int(rng.integers(2, 5))
    base = roles[n // 2] if n == 3 else roles[-1]
    view[reg] = base
    view[reg & (dist <= width) & (side > 0)] = roles[0]
    if n == 3:
        view[reg & (dist <= max(1, width - 1)) & (side < -0.2)] = roles[2]


# --------------------------------------------------------------------------- 物件

def _paint_object(b: _B, r0: int, c0: int, h: int, w: int, kind: str) -> None:
    """在 (r0, c0, h, w) 框里画一个物件：轮廓 + 分区填充 + 细节。"""
    rng = b.rng
    symmetric = rng.random() < 0.8
    if kind == "sprite":
        mask = _blob_mask(rng, h, w, symmetric)
    else:
        mask = _shape_mask(rng, h, w, str(rng.choice(["ellipse", "rrect", "diamond", "heart", "star", "triangle"])))
        symmetric = True
    if mask.sum() < 3:
        return
    view = b.idx[r0:r0 + h, c0:c0 + w]
    small = min(h, w) < 9
    has_outline = rng.random() < (0.9 if not small else 0.6)
    outline = _outline(mask, 2 if (min(h, w) > 70 and rng.random() < 0.4) else 1) if has_outline else np.zeros_like(mask)
    interior = mask & ~outline
    if has_outline:
        view[outline] = b.outline_role()
    if interior.sum() == 0:
        if not has_outline:
            view[mask] = b.fill_role()
        return
    n_parts = int(rng.integers(1, 3)) if small else int(rng.integers(2, 5 if min(h, w) < 20 else 8))
    parts = _partition(rng, interior, n_parts, symmetric)
    for p in range(n_parts):
        reg = parts == p
        if not reg.any():
            continue
        x = rng.random()
        if x < 0.18 and reg.sum() >= 6:                       # 肤色 / 色相渐变
            n = int(rng.integers(3, 7))
            skin = rng.random() < 0.6
            roles = b.ramp_roles(n, "skin" if skin else "hue")
            lev = _ramp_levels(rng, reg, n, dither=rng.random() < 0.5)
            view[reg] = np.array(roles, dtype=np.int16)[lev[reg]]
            if skin:
                b.tags.add("gradient_skin")
        elif x < 0.32:                                        # 白豆（轮廓里的白色）
            view[reg] = b.role("white")
            if rng.random() < 0.5 and reg.sum() >= 8:             # 白色区域里加一点浅灰阴影：相近色
                sims = b.similar_roles(int(rng.integers(1, 3)))
                shade = reg & ndimage.binary_dilation(outline, iterations=1) & (np.arange(w)[None, :] >= w // 2)
                for k, s in enumerate(sims):
                    sh = shade if k == 0 else (reg & ~shade & (rng.random(reg.shape) < 0.15))
                    view[sh] = s
                b.tags.add("similar_colors")
        elif x < 0.44 and reg.sum() >= 4:                    # 相近色组（白、米白、浅灰）
            sims = b.similar_roles(int(rng.integers(2, 4)))
            sub = _partition(rng, reg, len(sims), symmetric)
            for k, s in enumerate(sims):
                view[sub == k] = s
            b.tags.add("similar_colors")
        elif x < 0.54:                                        # 噪声纹理（毛发、草地）
            base = b.fill_role()
            others = [b.role("noise") for _ in range(int(rng.integers(1, 3)))]
            view[reg] = base
            noise = rng.random(reg.shape)
            pr = rng.uniform(0.08, 0.3)
            for k, o in enumerate(others):
                view[reg & (noise >= k * pr) & (noise < (k + 1) * pr)] = o
        elif x < 0.78 and reg.sum() >= 10:                   # 像素画常见的明暗：阴影 / 高光带
            n = int(rng.integers(2, 4))
            roles = b.ramp_roles(n, "hue")
            _shade(rng, view, reg, roles)
        elif x < 0.86 and reg.sum() >= 12:                   # 斑点
            view[reg] = b.fill_role()
            spot = b.fill_role()
            sm = _blob_mask(rng, reg.shape[0], reg.shape[1], symmetric) if rng.random() < 0.5 else                 (rng.random(reg.shape) < rng.uniform(0.05, 0.15))
            view[reg & sm & ~ndimage.binary_dilation(~reg, iterations=1)] = spot
        else:
            view[reg] = b.fill_role()
    # 分区之间的内轮廓
    if has_outline and n_parts > 1 and rng.random() < 0.35:
        edge = np.zeros_like(mask)
        edge[:, 1:] |= (parts[:, 1:] != parts[:, :-1]) & (parts[:, 1:] >= 0) & (parts[:, :-1] >= 0)
        edge[1:, :] |= (parts[1:, :] != parts[:-1, :]) & (parts[1:, :] >= 0) & (parts[:-1, :] >= 0)
        view[edge & interior] = b.outline_role()
    # 眼睛和高光（单像素细节）
    if min(h, w) >= 10 and rng.random() < 0.75:
        ey = int(round((rng.uniform(-0.45, 0.05) + 1) / 2 * h))
        ex = int(round(rng.uniform(0.15, 0.45) / 2 * w))
        cx = w // 2
        eye = b.role("eye") if rng.random() < 0.5 else b.outline_role()
        hl = b.role("highlight") if rng.random() < 0.6 else b.role("white")
        big = min(h, w) >= 28 and rng.random() < 0.6
        for sx in (-1, 1):
            x = cx + sx * ex - (1 if sx < 0 and w % 2 == 0 else 0)
            y = ey
            if 0 < y < h - 1 and 0 < x < w - 1 and interior[y, x]:
                if big:
                    view[y:y + 2, x:x + 2] = eye
                    view[y, x] = hl
                else:
                    view[y, x] = eye
                b.tags.add("single_pixel_detail")
    if min(h, w) >= 8 and rng.random() < 0.35:                   # 零星高光点
        hl = b.role("highlight")
        ys, xs = np.nonzero(interior)
        for k in rng.choice(len(ys), size=min(len(ys), int(rng.integers(1, 4))), replace=False):
            view[ys[k], xs[k]] = hl
        b.tags.add("single_pixel_detail")


def _ellipse(h: int, w: int, cy: float, cx: float, ry: float, rx: float, p: float = 2.0) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64) + 0.5
    return np.abs((yy - cy) / max(ry, 0.5)) ** p + np.abs((xx - cx) / max(rx, 0.5)) ** p <= 1.0


def _paint_character(b: _B, r0: int, c0: int, h: int, w: int) -> None:
    """卡通角色：头 + 身体 + 耳朵 / 手 / 脚，粗黑轮廓，眼睛、高光、腮红、嘴巴，可选头发和白肚皮。"""
    rng = b.rng
    view = b.idx[r0:r0 + h, c0:c0 + w]
    cx = w / 2.0
    head_h = h * rng.uniform(0.42, 0.62)
    head_w = w * rng.uniform(0.62, 0.95)
    hy = head_h / 2 + h * rng.uniform(0.04, 0.1)
    head = _ellipse(h, w, hy, cx, head_h / 2, head_w / 2, rng.uniform(2.0, 3.2))
    parts = {}
    if rng.random() < 0.5:                                         # 耳朵
        er = head_w * rng.uniform(0.1, 0.2)
        ex = head_w * rng.uniform(0.25, 0.4)
        ey = hy - head_h * rng.uniform(0.35, 0.5)
        k = rng.uniform(1.0, 1.8)
        ears = _ellipse(h, w, ey, cx - ex, er * k, er) | _ellipse(h, w, ey, cx + ex, er * k, er)
        parts["ears"] = ears & ~head
    body_top = hy + head_h * rng.uniform(0.3, 0.45)
    body_h = h - body_top - h * rng.uniform(0.02, 0.08)
    body_w = head_w * rng.uniform(0.5, 0.9)
    parts["body"] = _ellipse(h, w, body_top + body_h / 2, cx, body_h / 2, body_w / 2, rng.uniform(2.2, 5)) & ~head
    if rng.random() < 0.6:                                         # 手
        ay = body_top + body_h * rng.uniform(0.2, 0.4)
        ar = (body_h * rng.uniform(0.15, 0.25), body_w * rng.uniform(0.1, 0.16))
        arms = _ellipse(h, w, ay, cx - body_w / 2, ar[0], ar[1]) | _ellipse(h, w, ay, cx + body_w / 2, ar[0], ar[1])
        parts["arms"] = arms & ~head & ~parts["body"]
    if rng.random() < 0.6:                                         # 脚
        fy = body_top + body_h * 0.95
        fr = (max(1.0, body_h * 0.12), body_w * rng.uniform(0.14, 0.22))
        feet = _ellipse(h, w, fy, cx - body_w * 0.25, fr[0], fr[1]) | _ellipse(h, w, fy, cx + body_w * 0.25, fr[0], fr[1])
        parts["feet"] = feet & ~head & ~parts["body"]
    sil = head.copy()
    for m in parts.values():
        sil |= m
    if sil.sum() < 6:
        return
    thick = 2 if (min(h, w) > 90 and rng.random() < 0.5) else 1
    outline = _outline(sil, thick)
    ol = b.outline_role()
    # 头：肤色明暗 / 白色（白色动物）/ 普通填充
    hr = rng.random()
    inner_head = head & ~outline
    if hr < 0.5 and inner_head.sum() >= 4:
        roles = b.ramp_roles(int(rng.integers(2, 4)), "skin")
        _shade(rng, view, inner_head, roles)
        b.tags.add("gradient_skin")
        head_role = roles[-1]
    elif hr < 0.75:
        head_role = b.role("white")
        view[inner_head] = head_role
        if rng.random() < 0.5:
            sims = b.similar_roles(1)
            shade = inner_head & ndimage.binary_dilation(outline, iterations=1)
            shade &= np.arange(w)[None, :] > cx + head_w * 0.1
            view[shade] = sims[0]
            b.tags.add("similar_colors")
    else:
        head_role = b.fill_role()
        view[inner_head] = head_role
    # 身体、手、脚、耳朵
    inner_body = parts["body"] & ~outline
    if rng.random() < 0.5 and inner_body.sum() >= 10:
        _shade(rng, view, inner_body, b.ramp_roles(int(rng.integers(2, 4)), "hue"))
    else:
        view[inner_body] = b.fill_role()
    if rng.random() < 0.4 and inner_body.sum() >= 12:              # 白肚皮
        bel = _ellipse(h, w, body_top + body_h * 0.55, cx, body_h * 0.28, body_w * 0.25) & inner_body
        view[bel] = b.role("white")
    if rng.random() < 0.3 and inner_body.sum() >= 12:              # 条纹
        per = int(rng.integers(2, 5))
        stripe = ((np.arange(h)[:, None] // per) % 2 == 0) & np.ones((1, w), bool)
        view[inner_body & stripe] = b.fill_role()
    for name in ("arms", "ears", "feet"):
        if name in parts:
            m = parts[name] & ~outline
            view[m] = head_role if (name in ("arms", "ears") and rng.random() < 0.6) else b.fill_role()
            if name == "ears" and rng.random() < 0.5:
                view[ndimage.binary_erosion(m, iterations=1)] = b.fill_role()
    # 头发
    if rng.random() < 0.45:
        line = hy - head_h * rng.uniform(0.05, 0.3)
        wave = line + np.sin(np.arange(w) / max(1, w) * np.pi * rng.uniform(2, 6)) * head_h * 0.06
        hair = head & (np.arange(h)[:, None] < wave[None, :]) & ~outline
        view[hair] = ol if rng.random() < 0.5 else b.fill_role()
    view[outline] = ol
    if rng.random() < 0.7:                                         # 头和身体之间的内轮廓
        view[ndimage.binary_dilation(head, iterations=1) & parts["body"]] = ol
    # 五官
    e = max(1, int(round(min(h, w) / rng.uniform(16, 28))))
    ey = int(round(hy + head_h * rng.uniform(-0.05, 0.15)))
    ex = head_w * rng.uniform(0.17, 0.3)
    eye_role = ol if rng.random() < 0.7 else b.role("eye")
    hl = b.role("white") if rng.random() < 0.6 else b.role("highlight")
    for sx in (-1, 1):
        x = int(round(cx + sx * ex - e / 2))
        if 0 <= ey < h - e and 0 <= x < w - e:
            view[ey:ey + e, x:x + e] = eye_role
            if e >= 2:
                view[ey, x + (0 if sx < 0 else e - 1)] = hl
            b.tags.add("single_pixel_detail")
    if rng.random() < 0.7:                                         # 腮红
        cr = b.fill_role()
        cy_ = ey + e + max(1, e // 2)
        ch_, cw_ = max(1, e // 2 + 1), max(1, e + int(rng.integers(0, 2)))
        for sx in (-1, 1):
            x = int(round(cx + sx * (ex + e * 0.8) - cw_ / 2))
            if 0 <= cy_ < h - ch_ and 0 <= x < w - cw_:
                sl = (slice(cy_, cy_ + ch_), slice(x, x + cw_))
                view[sl][inner_head[sl]] = cr
    my = int(round(ey + e + head_h * rng.uniform(0.08, 0.16)))      # 嘴
    mw = max(1, int(round(head_w * rng.uniform(0.04, 0.12))))
    if 0 <= my < h:
        x0 = int(round(cx - mw / 2))
        view[my, max(0, x0):min(w, x0 + mw)] = ol if rng.random() < 0.6 else b.fill_role()


def _compose(b: _B, kind: str) -> None:
    rng, rows, cols = b.rng, b.rows, b.cols
    if kind == "character":
        fh, fw = rng.uniform(0.75, 0.98), rng.uniform(0.6, 0.95)
        h, w = max(8, int(rows * fh)), max(8, int(cols * fw))
        _paint_character(b, (rows - h) // 2, int((cols - w) * rng.uniform(0.2, 0.8)), h, w)
        if rng.random() < 0.4 and min(rows, cols) >= 24:          # 旁边的小物件（落在空白处才画）
            hh = max(5, int(min(rows, cols) * rng.uniform(0.12, 0.25)))
            r0, c0 = int(rng.integers(0, rows - hh)), int(rng.integers(0, cols - hh))
            region = b.idx[r0:r0 + hh, c0:c0 + hh]
            if (region < 0).all():
                sub = _B(hh, hh, rng)
                sub.roles, sub.shared_outline, sub._groups = b.roles, b.shared_outline, b._groups
                _paint_object(sub, 0, 0, hh, hh, "shape")
                b._groups = sub._groups
                m = sub.idx >= 0
                region[m] = sub.idx[m]
                b.tags |= sub.tags
    elif kind == "sprite":
        fh, fw = rng.uniform(0.7, 0.98), rng.uniform(0.7, 0.98)
        h, w = max(4, int(rows * fh)), max(4, int(cols * fw))
        _paint_object(b, (rows - h) // 2 + int(rng.integers(0, rows - h + 1)) // 3, (cols - w) // 2, h, w, "sprite")
        if rng.random() < 0.35 and min(rows, cols) >= 24:          # 旁边加一个小物件（爱心、星星）
            hh = max(5, int(min(rows, cols) * rng.uniform(0.12, 0.25)))
            r0, c0 = int(rng.integers(0, rows - hh)), int(rng.integers(0, max(1, cols // 4)))
            sub = _B(hh, hh, rng)
            sub.roles, sub.shared_outline, sub._groups = b.roles, b.shared_outline, b._groups
            _paint_object(sub, 0, 0, hh, hh, "shape")
            b._groups = sub._groups
            m = sub.idx >= 0
            b.idx[r0:r0 + hh, c0:c0 + hh][m] = sub.idx[m]
            b.tags |= sub.tags
    elif kind in ("multi", "shapes"):
        gr = int(rng.integers(1, 5)) if kind == "multi" else int(rng.integers(1, 3))
        gc = int(rng.integers(2, 5)) if kind == "multi" else int(rng.integers(1, 3))
        sh, sw = rows / gr, cols / gc
        for i in range(gr):
            for j in range(gc):
                if rng.random() < 0.12:
                    continue
                h = max(4, int(sh * rng.uniform(0.55, 0.92)))
                w = max(4, int(sw * rng.uniform(0.55, 0.92)))
                r0 = int(i * sh + rng.uniform(0, max(0, sh - h)))
                c0 = int(j * sw + rng.uniform(0, max(0, sw - w)))
                h, w = min(h, rows - r0), min(w, cols - c0)
                if h >= 4 and w >= 4:
                    x = rng.random()
                    if kind == "multi" and x < 0.4 and min(h, w) >= 10:
                        _paint_character(b, r0, c0, h, w)
                    else:
                        _paint_object(b, r0, c0, h, w, "sprite" if (kind == "multi" and x < 0.75) else "shape")
    elif kind == "scene":
        if rng.random() < 0.5:
            n = int(rng.integers(3, 7))
            roles = b.ramp_roles(n, "hue")
            lev = _ramp_levels(rng, np.ones((rows, cols), bool), n, dither=rng.random() < 0.5)
            b.idx[:, :] = np.array(roles, dtype=np.int16)[lev]
        else:
            b.idx[:, :] = b.role("bg")
        gh = int(rows * rng.uniform(0.1, 0.3))
        if gh:
            b.idx[rows - gh:, :] = b.fill_role()
            if rng.random() < 0.5:
                noise = rng.random((gh, cols)) < 0.2
                b.idx[rows - gh:, :][noise] = b.role("noise")
        h, w = max(4, int(rows * rng.uniform(0.45, 0.8))), max(4, int(cols * rng.uniform(0.35, 0.7)))
        _paint_object(b, rows - h - max(0, gh // 2), int(rng.integers(0, cols - w + 1)), h, w, "sprite")
    elif kind == "blocks":
        n = int(rng.integers(4, 22))
        if rng.random() < 0.5:                                     # Voronoi 色块
            ys, xs = np.mgrid[0:rows, 0:cols]
            sy, sx = rng.uniform(0, rows, n), rng.uniform(0, cols, n)
            d = (ys[..., None] - sy) ** 2 + (xs[..., None] - sx) ** 2
            lab = np.argmin(d, axis=-1)
        else:                                                      # 递归切矩形（拼布）
            lab = np.zeros((rows, cols), dtype=np.int64)
            boxes = [(0, 0, rows, cols)]
            k = 1
            while k < n and boxes:
                i = int(rng.integers(len(boxes)))
                r0, c0, r1, c1 = boxes.pop(i)
                if (r1 - r0) * (c1 - c0) < 4:
                    continue
                if (r1 - r0) > (c1 - c0) or ((r1 - r0) == (c1 - c0) and rng.random() < 0.5):
                    m = int(rng.integers(r0 + 1, r1)) if r1 - r0 > 1 else r0
                    lab[m:r1, c0:c1] = k
                    boxes += [(r0, c0, m, c1), (m, c0, r1, c1)]
                else:
                    m = int(rng.integers(c0 + 1, c1)) if c1 - c0 > 1 else c0
                    lab[r0:r1, m:c1] = k
                    boxes += [(r0, c0, r1, m), (r0, m, r1, c1)]
                k += 1
        roles = [b.fill_role() for _ in range(int(lab.max()) + 1)]
        if rng.random() < 0.3:
            sims = b.similar_roles(int(rng.integers(2, 4)))
            for k, s in enumerate(sims):
                roles[k % len(roles)] = s
            b.tags.add("similar_colors")
        b.idx[:, :] = np.array(roles, dtype=np.int16)[lab]
        if rng.random() < 0.4:
            edge = np.zeros((rows, cols), bool)
            edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
            edge[1:, :] |= lab[1:, :] != lab[:-1, :]
            b.idx[edge] = b.outline_role()
        if rng.random() < 0.4:                                     # 只保留一个大形状内的部分
            m = _shape_mask(rng, rows, cols, str(rng.choice(["ellipse", "heart", "rrect", "star"])))
            b.idx[~m] = -1
            if rng.random() < 0.7:
                b.idx[_outline(m) & m] = b.outline_role()
    else:  # noise
        h, w = max(4, int(rows * rng.uniform(0.6, 1.0))), max(4, int(cols * rng.uniform(0.6, 1.0)))
        r0, c0 = (rows - h) // 2, (cols - w) // 2
        m = _shape_mask(rng, h, w, str(rng.choice(["ellipse", "rrect", "blob"]))) if rng.random() < 0.6 else np.ones((h, w), bool)
        k = int(rng.integers(2, 7))
        roles = [b.fill_role() if i == 0 else b.role("noise") for i in range(k)]
        p = rng.dirichlet(np.ones(k) * 0.8)
        pick = rng.choice(k, size=(h, w), p=p)
        view = b.idx[r0:r0 + h, c0:c0 + w]
        view[m] = np.array(roles, dtype=np.int16)[pick[m]]
        b.tags.add("single_pixel_detail")


def make_basemap(rng: np.random.Generator, cols: int, rows: int, kind: str | None = None) -> BaseMap:
    if kind is None:
        keys = sorted(KINDS)
        p = np.array([KINDS[k] for k in keys])
        kind = keys[int(rng.choice(len(keys), p=p / p.sum()))]
    for _ in range(8):
        b = _B(rows, cols, rng)
        _compose(b, kind)
        bm = b.finish(kind)
        nz = int((bm.idx >= 0).sum())
        min_roles = 2 if (kind == "noise" or max(rows, cols) < 24) else 4
        if len(bm.roles) >= min_roles and nz >= max(4, 0.08 * rows * cols):
            return bm
    # 兜底：两色棋盘
    b = _B(rows, cols, rng)
    a, c = b.fill_role(), b.role("fill")
    yy, xx = np.mgrid[0:rows, 0:cols]
    b.idx[:, :] = np.where((yy // 2 + xx // 2) % 2 == 0, a, c)
    return b.finish("fallback")


def size_for_bucket(rng: np.random.Generator, bucket: str) -> tuple[int, int]:
    """按尺寸档抽 (cols, rows)。长边决定档位（size_bucket_of），短边按 0.6–1.0 的比例。"""
    lo, hi = {"S": (16, 40), "M": (41, 80), "L": (81, 130), "XL": (131, 208)}[bucket]
    long_side = int(rng.integers(lo, hi + 1))
    short = int(round(long_side * rng.uniform(0.6, 1.0)))
    short = max(min(short, long_side), 10)
    if rng.random() < 0.3:
        short = long_side
    return (long_side, short) if rng.random() < 0.5 else (short, long_side)
