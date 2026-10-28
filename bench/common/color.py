"""颜色工具（Python 侧），公式与 core/src/color 完全一致，便于对拍。

- sRGB → Lab：IEC 61966-2-1 矩阵 + D65 白点（xy = 0.3127, 0.3290），与 colour-science 默认一致
- CIEDE2000：Sharma 2005 公式，支持 numpy 向量化
"""
from __future__ import annotations

import numpy as np

_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.3127 / 0.3290, 1.0, (1.0 - 0.3127 - 0.3290) / 0.3290])
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def srgb8_to_lab(rgb) -> np.ndarray:
    """rgb: (..., 3) 的 0–255 整数或浮点。返回 (..., 3) 的 Lab。"""
    lin = srgb_to_linear(np.asarray(rgb, dtype=np.float64) / 255.0)
    xyz = lin @ _M.T
    t = xyz / _WHITE
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_srgb8(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    f = np.stack([fx, fy, fz], axis=-1)
    t = np.where(f ** 3 > _EPS, f ** 3, (116.0 * f - 16.0) / _KAPPA)
    xyz = t * _WHITE
    lin = xyz @ np.linalg.inv(_M).T
    srgb = linear_to_srgb(np.clip(lin, 0, None))
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def delta_e_2000(lab1, lab2) -> np.ndarray:
    """CIEDE2000（kL = kC = kH = 1），支持广播。"""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.where((b1 == 0) & (a1p == 0), 0.0, np.degrees(np.arctan2(b1, a1p)) % 360.0)
    h2p = np.where((b2 == 0) & (a2p == 0), 0.0, np.degrees(np.arctan2(b2, a2p)) % 360.0)
    dLp = L2 - L1
    dCp = C2p - C1p
    prod = C1p * C2p
    dh = h2p - h1p
    dh = np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh))
    dh = np.where(prod == 0, 0.0, dh)
    dHp = 2.0 * np.sqrt(prod) * np.sin(np.radians(dh) / 2.0)
    Lbar = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) <= 180.0, hsum / 2.0,
                    np.where(hsum < 360.0, (hsum + 360.0) / 2.0, (hsum - 360.0) / 2.0))
    hbar = np.where(prod == 0, hsum, hbar)
    T = (1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    dtheta = 30.0 * np.exp(-(((hbar - 275.0) / 25.0) ** 2))
    Cb7 = Cbarp ** 7
    RC = 2.0 * np.sqrt(Cb7 / (Cb7 + 25.0 ** 7))
    SL = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T
    RT = -np.sin(np.radians(2 * dtheta)) * RC
    tL, tC, tH = dLp / SL, dCp / SC, dHp / SH
    return np.sqrt(tL ** 2 + tC ** 2 + tH ** 2 + RT * tC * tH)
