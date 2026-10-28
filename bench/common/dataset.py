"""数据集层面的共用约定（collect / label / eval 共用）：私有数据目录下的路径、
「标注单元」（分页图按 group 合成一个单元）、只用于分辨率对比的样本过滤。

这里只放约定，不放业务逻辑；数据格式的读写和校验在 `bench/common/formats.py`。

**标注单元**：一张普通图纸就是一个单元，key = image_id；分页图（T4）同一个 group 的所有页合成一个单元，
key = group id，页序按 image_id 末尾的序号。真值文件是 `labels/<key>.json`，草稿是
`labels/drafts/<key>.json`。划分文件（dev.txt / test.txt）里仍然逐页写 image_id。
"""
from __future__ import annotations

import re
from pathlib import Path

from bench.common import formats

# beadcore-cli 和浏览器都不能直接读的格式：保留原图，另外由 `python -m bench.collect convert` 无损转成 PNG
DECODE_ONLY_FORMATS = ("heic", "heif", "avif")
# 只用于「App 保存 vs Web 端」分辨率对比的版本：不进入 dev/test 划分、不计入覆盖统计、不需要标注
VARIANT_SOURCE_PATHS = ("web",)
# 合法的 image_id / group id / run 名：只允许字母、数字、下划线、连字符（同时用作文件名和 URL 片段）
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_TRAIL_NUM = re.compile(r"(\d+)$")


def valid_id(s: str | None) -> bool:
    return bool(s) and ID_RE.match(s) is not None


# --------------------------------------------------------------------------- 私有数据目录下的路径

def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else formats.data_dir()


def images_dir(root: Path | None = None) -> Path:
    return _root(root) / "images"


def inbox_dir(root: Path | None = None) -> Path:
    return _root(root) / "inbox"


def labels_dir(root: Path | None = None) -> Path:
    return _root(root) / "labels"


def drafts_dir(root: Path | None = None) -> Path:
    return _root(root) / "labels" / "drafts"


def preds_dir(root: Path | None = None) -> Path:
    return _root(root) / "preds"


def decoded_dir(root: Path | None = None) -> Path:
    return _root(root) / "cache" / "decoded"


def image_path(rec: dict, root: Path | None = None) -> Path:
    """原图（原字节）路径。"""
    return images_dir(root) / rec["file"]


def decoded_path(image_id: str, root: Path | None = None) -> Path:
    """HEIC 等格式无损转出的 PNG 路径（`bench.collect convert` 生成）。"""
    return decoded_dir(root) / f"{image_id}.png"


def readable_image_path(rec: dict, root: Path | None = None) -> Path:
    """喂给 beadcore-cli / 标注工具的图片路径：HEIC 等格式用转出的 PNG，其余用原图。"""
    if rec.get("format") in DECODE_ONLY_FORMATS:
        return decoded_path(rec["image_id"], root)
    return image_path(rec, root)


def label_path(key: str, root: Path | None = None) -> Path:
    return labels_dir(root) / f"{key}.json"


def draft_path(key: str, root: Path | None = None) -> Path:
    return drafts_dir(root) / f"{key}.json"


# --------------------------------------------------------------------------- 样本与标注单元

def is_benchmark(rec: dict) -> bool:
    """是否是 benchmark 样本（排除只用于分辨率对比的 Web 版本）。"""
    return rec.get("source_path") not in VARIANT_SOURCE_PATHS


def page_number(image_id: str) -> int:
    """分页图的页码：image_id 末尾的序号（没有序号时为 0）。"""
    m = _TRAIL_NUM.search(image_id)
    return int(m.group(1)) if m else 0


def unit_key(rec: dict) -> str:
    """标注单元的 key：有 group 的用 group id，否则用 image_id。"""
    return rec.get("group") or rec["image_id"]


def build_units(manifest: dict[str, dict], ids=None) -> dict[str, list[dict]]:
    """把 manifest（或其中 ids 指定的部分）按标注单元分组。只包含 benchmark 样本。
    返回 {unit_key: [按页码排序的 manifest 记录]}，单元按 key 排序。"""
    recs = manifest.values() if ids is None else (manifest[i] for i in ids if i in manifest)
    units: dict[str, list[dict]] = {}
    for rec in recs:
        if not is_benchmark(rec):
            continue
        units.setdefault(unit_key(rec), []).append(rec)
    for pages in units.values():
        pages.sort(key=lambda r: (page_number(r["image_id"]), r["image_id"]))
    return dict(sorted(units.items()))


def unit_attrs(pages: list[dict]) -> dict:
    """标注单元的汇总属性（统计和分层划分用）：type、size_bucket、tags、palette_hint。
    分页图（多页或 type=T4）的 size_bucket 一律 XL（04 §2.1）。"""
    types = sorted({p.get("type") for p in pages if p.get("type")})
    typ = "T4" if "T4" in types else (types[0] if types else None)
    paged = typ == "T4" or len(pages) > 1
    bucket = formats.size_bucket_of(1, 1, paged=True) if paged else pages[0].get("size_bucket")
    tags = sorted({t for p in pages for t in (p.get("tags") or [])})
    hint = next((p.get("palette_hint") for p in pages if p.get("palette_hint")), None)
    return {"type": typ, "types": types, "size_bucket": bucket, "tags": tags, "palette_hint": hint,
            "paged": paged, "n_images": len(pages)}


def is_hard(attrs: dict) -> bool:
    """04 §0.2 Hard 集（采集阶段按人工预估的标签判断；blurry 的阈值 θ 由 eval 在 dev 集上标定）。"""
    return (attrs.get("size_bucket") in ("L", "XL") or attrs.get("type") == "T4"
            or bool({"watermark", "blurry"} & set(attrs.get("tags") or [])))
