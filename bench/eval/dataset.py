"""评测单元：把 split 里的 image_id 解析成「一张整图纸 + 它的真值」，并计算 Hard/Clean 划分所需的属性。

- 一个评测单元 = 一个真值文件（labels/<image_id>.json）。分页图（T4）的多页共用一个真值，按整张图纸评测。
- split 文件里分页图写每一页的 image_id；评测器按「直接同名 → manifest 的 group → 扫描所有真值的 images」
  的顺序找到它所属的真值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from bench.common import dataset as ds
from bench.common import formats as F
from bench.eval.codes import CodeMapper
from bench.eval.config import Config


class DatasetError(RuntimeError):
    pass


@dataclass
class Unit:
    image_id: str
    label: dict
    label_sha: str
    page_ids: tuple[str, ...]
    type: str
    size_bucket: str
    paged: bool
    human_tags: tuple[str, ...]
    tags: tuple[str, ...]            # 生效的标签：人工标签去掉 blurry，再按规则补上 blurry
    min_cell_px: float
    lap_var: float
    blurry: bool
    hard: bool
    hard_reasons: tuple[str, ...]
    in_main: bool                    # 主统计池（不含 T6、T7）
    in_flag_scope: bool              # 第 3、6 条口径（不含 T7）
    mapper: CodeMapper
    label_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.in_main and not self.hard


# --------------------------------------------------------------------------- 属性计算

def min_cell_px_of(label: dict) -> float:
    """最小格边长：每页、每个方向的平均格距（(最后一条线 − 第一条线) / 格数），取最小值。"""
    best = math.inf
    for p in label["pages"]:
        for lines in (p["x_lines"], p["y_lines"]):
            n = len(lines) - 1
            if n >= 1:
                best = min(best, abs(float(lines[-1]) - float(lines[0])) / n)
    if best is math.inf:
        raise DatasetError(f"{label.get('image_id')}: 真值没有网格线，无法计算格边长")
    return best


def load_gray(path: Path) -> np.ndarray:
    """按 EXIF 方向校正后转灰度（与 core 的输入一致：方向已校正）；带透明通道的先叠在白底上。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("L"), dtype=np.uint8)


def laplacian_variance(gray: np.ndarray, ksize: int = 1) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    return float(lap.var())


def find_image(images_dir: Path, file_name: str, manifest: dict[str, dict]) -> Path:
    p = images_dir / file_name
    if p.exists():
        return p
    stem = Path(file_name).stem
    m = manifest.get(stem)
    if m and m.get("file") and (images_dir / m["file"]).exists():
        return images_dir / m["file"]
    cands = sorted(images_dir.glob(f"{stem}.*"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise DatasetError(f"图片 {stem} 有多个候选文件：{[c.name for c in cands]}")
    raise DatasetError(f"找不到图片 {file_name}（在 {images_dir} 下）。blurry 判定需要原图")


def find_image_by_id(images_dir: Path, image_id: str, manifest: dict[str, dict]) -> Path:
    m = manifest.get(image_id) or {}
    if m.get("file") and (images_dir / m["file"]).exists():
        return images_dir / m["file"]
    cands = sorted(images_dir.glob(f"{image_id}.*"))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise DatasetError(f"图片 {image_id} 有多个候选文件：{[c.name for c in cands]}")
    raise DatasetError(f"找不到图片 {image_id}（在 {images_dir} 下）")


def readable_image(path: Path, images_dir: Path) -> Path:
    """HEIC 等 PIL 读不了的格式，改读 `bench.collect convert` 无损转出的 PNG（约定见 bench/common/dataset.py）。"""
    if path.suffix.lower().lstrip(".") not in ds.DECODE_ONLY_FORMATS:
        return path
    dec = ds.decoded_path(path.stem, images_dir.parent)
    if not dec.is_file():
        raise DatasetError(f"{path.name} 是 {path.suffix} 格式，需要先运行 `python -m bench.collect convert` 生成 {dec}")
    return dec


def lap_var_of(label: dict, images_dir: Path, manifest: dict[str, dict], ksize: int) -> float:
    """图纸区域（真值网格线围成的矩形）的拉普拉斯方差；分页图取各页最小值。"""
    best = math.inf
    cache: dict[Path, np.ndarray] = {}
    for p in label["pages"]:
        img = label["images"][p["image_index"]]
        path = readable_image(find_image(images_dir, img["file"], manifest), images_dir)
        if path not in cache:
            cache[path] = load_gray(path)
        gray = cache[path]
        h, w = gray.shape
        xs, ys = [float(v) for v in p["x_lines"]], [float(v) for v in p["y_lines"]]
        x0, x1 = max(0, math.floor(min(xs))), min(w, math.ceil(max(xs)))
        y0, y1 = max(0, math.floor(min(ys))), min(h, math.ceil(max(ys)))
        if x1 - x0 < 3 or y1 - y0 < 3:
            raise DatasetError(f"{label['image_id']}: 图纸区域太小或超出图片范围（{x0}:{x1}, {y0}:{y1}）")
        best = min(best, laplacian_variance(gray[y0:y1, x0:x1], ksize))
    return best


def make_unit(label: dict, cfg: Config, min_cell_px: float, lap_var: float, label_sha: str = "",
              label_path: Path | None = None) -> Unit:
    """由真值和两个图像属性得到评测单元（测试里可以直接传入 min_cell_px / lap_var）。"""
    t = label["type"]
    rows, cols = label["grid"]["rows"], label["grid"]["cols"]
    paged = t == "T4" or len(label["images"]) > 1 or len(label["pages"]) > 1
    bucket = F.size_bucket_of(cols, rows, paged=paged)
    human = tuple(label.get("tags") or ())
    blurry = (min_cell_px < cfg.min_cell_px) or (lap_var < cfg.theta)
    tags = set(human) - {"blurry"}
    if blurry:
        tags.add("blurry")
    reasons: list[str] = []
    if bucket in cfg.hard_size_buckets:
        reasons.append(f"size:{bucket}")
    for tg in cfg.hard_tags:
        if tg in tags:
            reasons.append(f"tag:{tg}")
    if t in cfg.hard_types:
        reasons.append(f"type:{t}")
    warnings: list[str] = []
    if label.get("size_bucket") and label["size_bucket"] != bucket:
        warnings.append(f"真值里的 size_bucket={label['size_bucket']}，按网格计算为 {bucket}（以计算结果为准）")
    if ("blurry" in human) != blurry:
        warnings.append(f"人工 blurry 标签={'有' if 'blurry' in human else '无'}，规则判定={'是' if blurry else '否'}"
                        f"（最小格边长 {min_cell_px:.2f} px，拉普拉斯方差 {lap_var:.1f}；以规则为准）")
    if not label.get("palette_hint"):
        warnings.append("真值缺 palette_hint：比较色号时不区分品牌")
    return Unit(
        image_id=label["image_id"], label=label, label_sha=label_sha,
        page_ids=tuple(Path(im["file"]).stem for im in label["images"]),
        type=t, size_bucket=bucket, paged=paged, human_tags=human, tags=tuple(sorted(tags)),
        min_cell_px=float(min_cell_px), lap_var=float(lap_var), blurry=blurry,
        hard=bool(reasons), hard_reasons=tuple(reasons),
        in_main=t not in cfg.main_excluded_types,
        in_flag_scope=t not in cfg.flag_legend_excluded_types,
        mapper=CodeMapper(label.get("palette_hint")),
        label_path=label_path, warnings=warnings,
    )


# --------------------------------------------------------------------------- split → 单元

def _label_index(labels_dir: Path) -> dict[str, Path]:
    """扫描全部真值：页 image_id（images[].file 去扩展名）和真值自身的 image_id → 真值文件。"""
    idx: dict[str, Path] = {}
    for p in sorted(labels_dir.glob("*.json")):
        try:
            lab = F.load_json(p)
        except Exception:
            continue
        idx.setdefault(lab.get("image_id", p.stem), p)
        for im in lab.get("images") or []:
            if isinstance(im, dict) and im.get("file"):
                idx.setdefault(Path(im["file"]).stem, p)
    return idx


def resolve_label_paths(ids: list[str], labels_dir: Path, manifest: dict[str, dict],
                        allow_missing: bool = False) -> tuple[list[Path], list[str]]:
    """split 里的 image_id → 去重后的真值文件列表（按文件名排序）。"""
    found: dict[Path, None] = {}
    missing: list[str] = []
    index: dict[str, Path] | None = None
    for i in ids:
        p = labels_dir / f"{i}.json"
        if not p.exists():
            g = (manifest.get(i) or {}).get("group")
            p = labels_dir / f"{g}.json" if g else p
        if not p.exists():
            if index is None:
                index = _label_index(labels_dir)
            p = index.get(i)  # type: ignore[assignment]
        if p is None or not p.exists():
            missing.append(i)
            continue
        found[p] = None
    warns: list[str] = []
    if missing:
        msg = f"{len(missing)} 个 image_id 找不到真值：{missing[:10]}{' …' if len(missing) > 10 else ''}"
        if not allow_missing:
            raise DatasetError(msg + "（标注未完成时可在 dev 集上加 --allow-missing-labels 跳过）")
        warns.append(msg + "（已跳过）")
    return sorted(found, key=lambda q: q.name), warns


def load_units(split: str, cfg: Config, labels_dir: Path, images_dir: Path, manifest: dict[str, dict],
               allow_missing: bool = False) -> tuple[list[Unit], list[str]]:
    """读取 split 的全部评测单元。真值结构校验不通过、跨 split 泄漏、缺原图都直接报错。"""
    if split == "synth":
        # 合成集（bench/synth 生成的目录）：评测 manifest 里的全部样本，不涉及 dev/test 划分
        ids = sorted(manifest)
        other: list[str] = []
    else:
        ids = F.read_split(split)
        other = F.read_split("test" if split == "dev" else "dev")
    if not ids:
        raise DatasetError(f"split {split} 为空（{F.DATASETS / (split + '.txt')}）")
    overlap = sorted(set(ids) & set(other))
    if overlap:
        raise DatasetError(f"dev 和 test 有重叠的 image_id：{overlap[:10]}")
    paths, warns = resolve_label_paths(ids, labels_dir, manifest, allow_missing)
    id_set, other_set = set(ids), set(other)
    units: list[Unit] = []
    errors: list[str] = []
    seen: dict[str, Path] = {}
    for p in paths:
        lab = F.load_json(p)
        errs = F.validate_label(lab)
        if errs:
            errors.append(f"{p.name}: " + "；".join(errs))
            continue
        uid = lab["image_id"]
        if uid in seen:
            errors.append(f"{p.name} 与 {seen[uid].name} 的 image_id 都是 {uid}")
            continue
        seen[uid] = p
        pages = [Path(im["file"]).stem for im in lab["images"]]
        leak = [q for q in pages if q in other_set]
        if leak:
            errors.append(f"{uid}: 分页 {leak} 在另一个 split 里（数据泄漏）")
            continue
        unit = make_unit(lab, cfg, min_cell_px_of(lab), lap_var_of(lab, images_dir, manifest, cfg.laplacian_ksize),
                         label_sha=F.sha256_file(p), label_path=p)
        if p.stem != uid:
            unit.warnings.append(f"真值文件名 {p.name} 与 image_id {uid} 不一致")
        not_listed = [q for q in pages if q not in id_set]
        if not_listed and uid not in id_set:
            unit.warnings.append(f"分页 {not_listed} 没有列在 {split}.txt 里")
        m = manifest.get(uid) or manifest.get(pages[0])
        if m and m.get("type") and m["type"] != unit.type:
            unit.warnings.append(f"manifest 里 type={m['type']}，真值为 {unit.type}（以真值为准）")
        units.append(unit)
    if errors:
        raise DatasetError("真值有问题，拒绝评测：\n  " + "\n  ".join(errors))
    units.sort(key=lambda u: u.image_id)
    return units, warns
