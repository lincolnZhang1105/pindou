"""HEIC / HEIF / AVIF → PNG（无损），供 beadcore-cli 和标注工具使用（CLI 不引入 libheif，05 §4.1）。

- `images/` 里的原图保持不动；PNG 写到 `$PINDOU_DATA_DIR/cache/decoded/<image_id>.png`；
- 解码时应用 HEIF 的旋转 / 镜像（irot / imir）和 EXIF 方向，PNG 里是「方向已校正」的像素，
  与 core 的约定一致（03 §3.1：方向在平台层校正）；真值坐标也以这张 PNG 为准；
- 像素不缩放、不重采样；PNG 本身无损。

HEIC 解码依赖可选的 pillow-heif（不在 requirements.txt 里）：
源码 BSD-3-Clause，但**二进制 wheel 捆绑了 libheif / libde265（LGPL-3.0）和 x265（GPL-2.0），整体按 GPLv2 发布**。
只在 bench 本地使用，绝不能进入 app/、core/、palettes/、backend/。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from bench.common import dataset as ds
from bench.common import formats
from bench.collect.imageinfo import probe


def _open_heif_capable(src: Path) -> Image.Image:
    try:
        return Image.open(src)
    except Exception:  # noqa: BLE001 —— PIL 不认识时再尝试 pillow-heif
        pass
    try:
        import pillow_heif  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "解码 HEIC 需要 pillow-heif：.venv/Scripts/pip install pillow-heif"
            "（二进制 wheel 是 GPLv2，只在 bench 本地用，见 bench/collect/README.md）") from e
    pillow_heif.register_heif_opener()
    return Image.open(src)


def decode_to_png(src: Path, dst: Path) -> tuple[int, int]:
    """解码 src，校正方向后无损写成 PNG。返回 (宽, 高)。"""
    with _open_heif_capable(src) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
        im = im.convert("RGBA" if has_alpha else "RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        im.save(tmp, format="PNG")
        tmp.replace(dst)
        return im.size


def convert(ids: list[str] | None = None, root: Path | None = None, *, force: bool = False):
    """转换 manifest 里所有（或 ids 指定的）HEIC/HEIF/AVIF 图片。返回 (已转换, 已跳过, 错误)。"""
    root = Path(root) if root is not None else formats.data_dir()
    manifest = formats.load_manifest(formats.MANIFEST)
    if ids:
        missing = [i for i in ids if i not in manifest]
        recs = [manifest[i] for i in ids if i in manifest]
    else:
        missing = []
        recs = list(manifest.values())
    errors = [f"{i}：不在 manifest 里" for i in missing]
    done, skipped = [], []
    for rec in recs:
        if rec.get("format") not in ds.DECODE_ONLY_FORMATS:
            if ids:
                skipped.append(f"{rec['image_id']}：{rec.get('format')} 不需要转换")
            continue
        src = ds.image_path(rec, root)
        dst = ds.decoded_path(rec["image_id"], root)
        if dst.is_file() and not force:
            skipped.append(f"{rec['image_id']}：已存在 {dst.name}")
            continue
        if not src.is_file():
            errors.append(f"{rec['image_id']}：原图不存在 {src}")
            continue
        try:
            w, h = decode_to_png(src, dst)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{rec['image_id']}：解码失败：{e}")
            continue
        info = probe(dst)
        if (info["width"], info["height"]) != (w, h):
            errors.append(f"{rec['image_id']}：写出的 PNG 尺寸不对")
        note = ""
        if (w, h) != (rec.get("width"), rec.get("height")):
            note = f"（manifest 记录 {rec.get('width')}×{rec.get('height')}，解码后 {w}×{h}：方向校正）"
        done.append(f"{rec['image_id']} → {dst}  {w}×{h}{note}")
    return done, skipped, errors
