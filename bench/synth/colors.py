"""把底图的「角色」映射成具体色号和色值。

色值只有三种来源（写进附加标注的 codebook[].source，测试会检查）：
- `palette:<干净色卡 id>:<色号>`：取自 `palettes/build/palettes.json`；
- `random`：在 Lab 空间随机生成；
- `empty_bg`：白豆和空格同色（white_vs_empty），等于渲染时的空格底色。

色号文字可以来自任何 `CodeTable`（包括只有文字的 bench 内部色卡）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.common.color import delta_e_2000, lab_to_srgb8, srgb8_to_lab
from bench.synth.basemaps import BaseMap, Role
from bench.synth.palettes import CleanPalette, CodeTable, ColorPool


@dataclass
class Entry:
    code: str
    rgb: tuple[int, int, int]
    source: str


def _lch(lab: np.ndarray):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360.0


def _pred(role: Role, lab: np.ndarray, hue0: float | None) -> np.ndarray:
    L, C, h = _lch(lab)
    k = role.kind
    if k == "outline":
        return L <= 30
    if k == "eye":
        return L <= 35
    if k == "highlight":
        return (L >= 85) & (C <= 25)
    if k == "white":
        return (L >= 93) & (C <= 5)
    if k == "similar":
        return (L >= 78) & (L <= 97) & (C <= 14)
    if k == "ramp":
        if role.ramp == "skin":
            return (h >= 15) & (h <= 75) & (C >= 8) & (C <= 50) & (L >= 40) & (L <= 93)
        dh = np.abs(((h - (hue0 or 0)) + 180) % 360 - 180)
        return (dh <= 28) & (C >= 12) & (L >= 22) & (L <= 92)
    if k == "bg":
        return L >= 60
    return np.ones(L.shape, dtype=bool)


def _random_lab(rng, role: Role, hue0: float | None, n: int = 64) -> np.ndarray:
    """按角色在 Lab 里随机采样 n 个候选（转成 sRGB 再转回，保证在色域内）。"""
    k = role.kind
    if k in ("outline", "eye"):
        L = rng.uniform(3, 28, n); C = rng.uniform(0, 25, n)
    elif k == "highlight":
        L = rng.uniform(86, 99, n); C = rng.uniform(0, 20, n)
    elif k == "white":
        L = rng.uniform(95, 100, n); C = rng.uniform(0, 3, n)
    elif k == "similar":
        L = rng.uniform(80, 96, n); C = rng.uniform(0, 12, n)
    elif k == "ramp" and role.ramp == "skin":
        L = rng.uniform(45, 92, n); C = rng.uniform(10, 45, n)
        hue0 = 50.0 if hue0 is None else hue0
    elif k == "ramp":
        L = rng.uniform(25, 90, n); C = rng.uniform(15, 60, n)
    elif k == "bg":
        L = rng.uniform(65, 97, n); C = rng.uniform(0, 40, n)
    else:
        L = rng.uniform(15, 95, n); C = rng.uniform(0, 75, n)
    if hue0 is not None and k == "ramp":
        h = np.radians(hue0 + rng.uniform(-15, 15, n))
    else:
        h = rng.uniform(0, 2 * np.pi, n)
    lab = np.stack([L, C * np.cos(h), C * np.sin(h)], axis=-1)
    rgb = lab_to_srgb8(lab)
    return rgb


def _min_de(lab: np.ndarray, chosen: list[np.ndarray]) -> np.ndarray:
    if not chosen:
        return np.full(lab.shape[0], 1e9)
    return delta_e_2000(lab[:, None, :], np.asarray(chosen)[None, :, :]).min(axis=1)


def assign_colors(rng: np.random.Generator, bm: BaseMap, table: CodeTable, native: CleanPalette | None,
                  pool: ColorPool, empty_rgb: tuple[int, int, int], p_random: float = 0.2
                  ) -> tuple[list[Entry], tuple[int, int, int]]:
    """返回 (每个角色对应的 Entry, 实际使用的空格底色)。

    native 不为 None 时，色号和色值都取自这套干净色卡（保持色号↔色值一致）；否则色号从 table 抽取，
    色值从干净色值池或随机生成中取。
    """
    roles = bm.roles
    n = len(roles)
    empty_lab = srgb8_to_lab(np.array(empty_rgb, dtype=np.float64))
    if native is not None:
        cand_rgb, cand_lab, cand_src = native.rgb, native.lab, [native.source_of(i) for i in range(len(native.codes))]
        use_random = False
    else:
        cand_rgb, cand_lab, cand_src = pool.rgb, pool.lab, list(pool.sources)
        use_random = rng.random() < p_random
    used = np.zeros(len(cand_rgb), dtype=bool)
    out_rgb: list[tuple[int, int, int] | None] = [None] * n
    out_src: list[str] = [""] * n
    out_cand: list[int] = [-1] * n
    chosen_lab: list[np.ndarray] = []
    hue_of_group: dict[int, float] = {}
    group_base: dict[int, np.ndarray] = {}

    prio = {"white": 0, "similar": 1, "ramp": 2, "outline": 3, "eye": 4, "highlight": 5, "bg": 6, "fill": 7, "noise": 8}
    order = sorted(range(n), key=lambda i: (prio.get(roles[i].kind, 9), roles[i].group, roles[i].order, i))

    # white_vs_empty：白豆和空格同色
    white_rgb, white_taken, white_cand, white_src = None, False, -1, ""
    if any(r.kind == "white" for r in roles):
        if native is not None:
            L, C, _ = _lch(native.lab)
            score = np.where(C <= 6, L, L - 50)
            wi = int(np.argmax(score))
            white_rgb = tuple(int(v) for v in native.rgb[wi])
            if float(delta_e_2000(native.lab[wi], empty_lab)) < 4.0:
                # 色卡里的白色接近纯白：让空格底色等于它，两者完全一样（3_0 的 H2 就是这样）
                empty_rgb = white_rgb
                empty_lab = srgb8_to_lab(np.array(empty_rgb, dtype=np.float64))
            white_src, white_cand = native.source_of(wi), wi
        else:
            white_rgb, white_src, white_cand = tuple(int(v) for v in empty_rgb), "empty_bg", -1

    for i in order:
        r = roles[i]
        if r.kind == "white":
            if white_taken:
                pass  # 同一色只能给一个角色：第二个白色角色按普通流程选
            else:
                white_taken = True
                out_rgb[i], out_src[i], out_cand[i] = white_rgb, white_src, white_cand
                if white_cand >= 0:
                    used[white_cand] = True
                chosen_lab.append(empty_lab if white_cand < 0 else native.lab[white_cand])
                continue
        hue0 = None
        if r.kind == "ramp":
            if r.group not in hue_of_group:
                hue_of_group[r.group] = float(rng.uniform(15, 70) if r.ramp == "skin" else rng.uniform(0, 360))
            hue0 = hue_of_group[r.group]
        # 和已选颜色的最小色差：相近色组内允许很接近，其他至少 5
        min_de = 5.0
        if r.kind in ("similar", "ramp"):
            min_de = 1.2 if r.kind == "similar" else 2.5
        pick = -1
        if not use_random:
            ok = _pred(r, cand_lab, hue0) & ~used
            if ok.any():
                de = _min_de(cand_lab, chosen_lab)
                de_empty = delta_e_2000(cand_lab, empty_lab)
                ok2 = ok & (de >= min_de) & (de_empty >= 3.0)
                if r.kind == "similar" and r.group in group_base:
                    ok2 &= delta_e_2000(cand_lab, group_base[r.group]) <= 7.0
                if ok2.any():
                    idx = np.flatnonzero(ok2)
                    pick = int(idx[int(rng.integers(len(idx)))])
            if pick < 0 and native is not None:
                # native 方案不能随机造色：放宽条件，从这套色卡里挑离已选颜色最远的
                de = _min_de(cand_lab, chosen_lab)
                de = np.where(used, -1, de)
                pick = int(np.argmax(de))
        if pick >= 0:
            used[pick] = True
            out_rgb[i] = tuple(int(v) for v in cand_rgb[pick])
            out_src[i] = cand_src[pick]
            out_cand[i] = pick
            lab = cand_lab[pick]
        else:
            best, best_lab, best_score = None, None, -1e9
            for _ in range(6):
                rgb = _random_lab(rng, r, hue0)
                lab_c = srgb8_to_lab(rgb)
                ok = _pred(r, lab_c, hue0)
                de = _min_de(lab_c, chosen_lab)
                de_empty = delta_e_2000(lab_c, empty_lab)
                score = np.minimum(de, 40) + 20 * ok + 10 * (de_empty >= 3.0)
                if r.kind == "similar" and r.group in group_base:
                    score -= 3 * np.maximum(0, delta_e_2000(lab_c, group_base[r.group]) - 7.0)
                j = int(np.argmax(score))
                if score[j] > best_score:
                    best, best_lab, best_score = rgb[j], lab_c[j], float(score[j])
                if ok[j] and de[j] >= min_de and de_empty[j] >= 3.0:
                    break
            out_rgb[i] = tuple(int(v) for v in best)
            out_src[i] = "random"
            lab = best_lab
        chosen_lab.append(np.asarray(lab, dtype=np.float64))
        if r.kind == "similar" and r.group not in group_base:
            group_base[r.group] = np.asarray(lab, dtype=np.float64)

    # 渐变组：按亮度排序后重新分配，保证 order 0 最暗
    groups: dict[int, list[int]] = {}
    for i, r in enumerate(roles):
        if r.kind == "ramp":
            groups.setdefault(r.group, []).append(i)
    for g, members in groups.items():
        members.sort(key=lambda i: roles[i].order)
        trip = sorted(((float(srgb8_to_lab(np.array(out_rgb[i], dtype=np.float64))[0]), out_rgb[i], out_src[i], out_cand[i])
                       for i in members), key=lambda t: (t[0], t[1]))
        for i, t in zip(members, trip):
            out_rgb[i], out_src[i], out_cand[i] = t[1], t[2], t[3]

    # 色号
    if native is not None:
        codes = [native.codes[c] if c >= 0 else "" for c in out_cand]
        # 理论上 native 不会走到随机分支；保险起见，缺色号的从未用过的色号里补
        spare = [c for k, c in enumerate(native.codes) if not used[k]]
        for i in range(n):
            if not codes[i]:
                codes[i] = spare.pop(0)
    else:
        codes = _sample_codes(rng, table, n)
    return [Entry(code=codes[i], rgb=out_rgb[i], source=out_src[i]) for i in range(n)], tuple(int(v) for v in empty_rgb)


def _sample_codes(rng, table: CodeTable, n: int) -> list[str]:
    codes = list(table.codes)
    if n > len(codes):
        raise ValueError(f"{table.id} 只有 {len(codes)} 个色号，不够 {n} 色")
    if rng.random() < 0.5:
        sel = rng.choice(len(codes), size=n, replace=False)
        return [codes[int(k)] for k in sel]
    # 按系列聚集抽取（真实图纸的色号往往集中在几个系列）
    series: dict[str, list[str]] = {}
    for c in codes:
        s = "".join(ch for ch in c if ch.isalpha()) or c[:1]
        series.setdefault(s, []).append(c)
    keys = sorted(series)
    picked: list[str] = []
    seen: set[str] = set()
    while len(picked) < n:
        s = keys[int(rng.integers(len(keys)))]
        pool = [c for c in series[s] if c not in seen]
        if not pool:
            keys.remove(s)
            if not keys:
                break
            continue
        take = min(len(pool), int(rng.integers(1, 6)), n - len(picked))
        for k in rng.choice(len(pool), size=take, replace=False):
            picked.append(pool[int(k)])
            seen.add(pool[int(k)])
    if len(picked) < n:
        rest = [c for c in codes if c not in seen]
        for k in rng.choice(len(rest), size=n - len(picked), replace=False):
            picked.append(rest[int(k)])
    return picked
