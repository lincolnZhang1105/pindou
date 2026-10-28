"""几何变换与标注同步。

坐标约定（真值、附加标注统一使用）：**连续像素坐标**，像素 (i, j) 覆盖 [i, i+1) × [j, j+1)，
所以像素中心是 (i + 0.5, j + 0.5)。在这个约定下，把图片从宽 W 缩放到 W' 时坐标严格按 x' = x · W'/W 变换。

- `x_lines[i]`：第 i 条竖网格线的**中心**横坐标（没有网格线的像素画就是格子边界）；
- 框 `[x0, y0, x1, y1]`：左上角和右下角（x1、y1 不含）。

每张输出图维护一个 3×3 矩阵 M（渲染坐标 → 输出图坐标）。缩放、平移、裁切是轴对齐变换，
`x_lines` 可以精确更新；透视（T6 拍照）不是轴对齐的，网格线不再是竖直/水平的，处理方式见 `Ann.transformed`。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


def mat_scale(sx: float, sy: float) -> np.ndarray:
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def mat_translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def apply_pts(M: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    h = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    return h[:, :2] / h[:, 2:3]


def is_axis_aligned(M: np.ndarray, eps: float = 1e-9) -> bool:
    return abs(M[0, 1]) < eps and abs(M[1, 0]) < eps and abs(M[2, 0]) < eps and abs(M[2, 1]) < eps


def box_tf(M: np.ndarray, box) -> list[float]:
    x0, y0, x1, y1 = box
    p = apply_pts(M, [[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    return [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]


def cv2_matrix(M: np.ndarray) -> np.ndarray:
    """连续坐标下的 M → OpenCV（以像素中心为整数坐标）用的矩阵：T(-0.5) · M · T(+0.5)。"""
    return mat_translate(-0.5, -0.5) @ M @ mat_translate(0.5, 0.5)


def r3(v: float) -> float:
    return float(round(float(v), 3))


def rbox(b) -> list[float]:
    return [r3(v) for v in b]


@dataclass
class Ann:
    """一张图片上的全部几何标注（坐标系由所属图片决定）。

    pages：网格页，字段同真值的 pages（x_lines / y_lines / cols / rows / col_offset / row_offset），
           另有 `H`：从 x_lines/y_lines 所在的「平面坐标」到本图坐标的 3×3 矩阵（轴对齐时恒为单位阵）。
    legend / coords / texts / regions：附加标注的条目（含 box 字段）。
    """
    width: int
    height: int
    pages: list[dict] = field(default_factory=list)
    legend: list[dict] = field(default_factory=list)
    coords: list[dict] = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)
    regions: list[dict] = field(default_factory=list)

    def transformed(self, M: np.ndarray, width: int, height: int) -> "Ann":
        out = Ann(width=width, height=height)
        aligned = is_axis_aligned(M)
        for p in self.pages:
            q = copy.deepcopy(p)
            if aligned and is_axis_aligned(p["H"]):
                sx, tx, sy, ty = M[0, 0] / M[2, 2], M[0, 2] / M[2, 2], M[1, 1] / M[2, 2], M[1, 2] / M[2, 2]
                q["x_lines"] = [sx * x + tx for x in p["x_lines"]]
                q["y_lines"] = [sy * y + ty for y in p["y_lines"]]
            else:
                q["H"] = M @ p["H"]
            out.pages.append(q)
        for name in ("legend", "coords", "texts", "regions"):
            items = []
            for it in getattr(self, name):
                it2 = dict(it)
                for k in list(it2):
                    if k.endswith("box") and it2[k] is not None:
                        it2[k] = box_tf(M, it2[k])
                items.append(it2)
            setattr(out, name, items)
        return out

    def cropped(self, x0: int, y0: int, x1: int, y1: int, keep_min_frac: float = 0.6) -> "Ann":
        """裁切到 [x0, x1) × [y0, y1)：平移坐标，丢掉大部分落在裁切范围外的条目。"""
        out = self.transformed(mat_translate(-x0, -y0), x1 - x0, y1 - y0)
        W, H = x1 - x0, y1 - y0
        for name in ("legend", "coords", "texts", "regions"):
            kept = []
            for it in getattr(out, name):
                b = it.get("box") or it.get("swatch_box")
                if b is None or _inside_frac(b, W, H) >= keep_min_frac:
                    kept.append(it)
            setattr(out, name, kept)
        return out

    def page_label(self, p: dict) -> dict:
        """输出真值格式的 page（坐标保留 3 位小数）。

        轴对齐：x_lines / y_lines 就是本图像素坐标，bbox 是外框线中心围成的矩形。
        非轴对齐（T6）：
          - `x_lines[i]` / `y_lines[j]` 取该条线**中点**在本图里的横 / 纵坐标（近似值，供格距、模糊度等粗略统计）；
          - 精确几何：`flat_x_lines` / `flat_y_lines`（平面坐标）+ `homography`（3×3，行优先，平面坐标 → 本图像素）；
          - bbox：网格四角变换后的外接矩形。
        """
        H = p["H"]
        out = {"image_index": p["image_index"]}
        xs, ys = p["x_lines"], p["y_lines"]
        if is_axis_aligned(H):
            xl = [r3(x) for x in xs]
            yl = [r3(y) for y in ys]
            out["bbox"] = [xl[0], yl[0], xl[-1], yl[-1]]
            out["x_lines"], out["y_lines"] = xl, yl
        else:
            ymid, xmid = (ys[0] + ys[-1]) / 2, (xs[0] + xs[-1]) / 2
            xl = [r3(v) for v in apply_pts(H, [[x, ymid] for x in xs])[:, 0]]
            yl = [r3(v) for v in apply_pts(H, [[xmid, y] for y in ys])[:, 1]]
            corners = apply_pts(H, [[xs[0], ys[0]], [xs[-1], ys[0]], [xs[0], ys[-1]], [xs[-1], ys[-1]]])
            out["bbox"] = rbox([corners[:, 0].min(), corners[:, 1].min(), corners[:, 0].max(), corners[:, 1].max()])
            out["x_lines"], out["y_lines"] = xl, yl
            Hn = H / H[2, 2]
            out["homography"] = [[float(round(v, 9)) for v in row] for row in Hn]
            out["flat_x_lines"] = [r3(x) for x in xs]
            out["flat_y_lines"] = [r3(y) for y in ys]
        for k in ("col_offset", "row_offset", "cols", "rows"):
            out[k] = p[k]
        return out


def _inside_frac(b, W: float, H: float) -> float:
    x0, y0, x1, y1 = b
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    ix = max(0.0, min(x1, W) - max(x0, 0.0))
    iy = max(0.0, min(y1, H) - max(y0, 0.0))
    return ix * iy / area
