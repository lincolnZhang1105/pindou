"""manifest 的登记（add）和核对（verify）。

**只处理人工保存下来的图片，不写任何抓取、下载代码**（AGENTS 规则 5、04 §2.1）。
登记时原字节复制到 `$PINDOU_DATA_DIR/images/`，绝不重新编码、绝不缩放。
"""
from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

from bench.common import dataset as ds
from bench.common import formats
from bench.collect.imageinfo import EXT_OF, IMAGE_SUFFIXES, ImageInfoError, probe

# manifest 字段顺序（bench/README.md），后面是可选字段
FIELDS = ("image_id", "file", "sha256", "width", "height", "format", "platform", "note_id", "author", "url",
          "saved_at", "source_path", "type", "size_bucket", "palette_hint", "tags", "group", "notes")
OPTIONAL_FIELDS = ("pair_of", "grid", "exif_orientation")
# 可以由命令行参数或 sidecar CSV 提供的元数据
META_FIELDS = ("image_id", "platform", "note_id", "author", "url", "saved_at", "source_path", "type",
               "size_bucket", "palette_hint", "tags", "group", "notes", "pair_of", "grid")
SOURCE_PATHS = formats.SOURCE_PATHS + ds.VARIANT_SOURCE_PATHS
KNOWN_PLATFORMS = ("xhs", "douyin", "own", "photo", "other")
_GRID_RE = re.compile(r"^\s*(\d+)\s*[x×X*,]\s*(\d+)\s*$")
_TAG_SPLIT = re.compile(r"[;,|\s]+")
_SUFFIX_ALIASES = {".jpg": (".jpg", ".jpeg"), ".tif": (".tif", ".tiff")}


class CollectError(Exception):
    """登记 / 核对失败。args[0] 是错误列表。"""

    def __init__(self, errors: list[str]):
        super().__init__(errors)
        self.errors = errors

    def __str__(self) -> str:
        return "\n".join(self.errors)


def manifest_path() -> Path:
    # 运行时取 formats.MANIFEST（测试会 monkeypatch 它）
    return formats.MANIFEST


def load_rows() -> list[dict]:
    return formats.read_jsonl(manifest_path())


def write_rows(rows: list[dict]) -> None:
    formats.write_jsonl(manifest_path(), rows)


def parse_grid(s) -> dict | None:
    """'58x53' → {"cols": 58, "rows": 53}（列在前）。"""
    if s in (None, ""):
        return None
    if isinstance(s, dict):
        return {"cols": int(s["cols"]), "rows": int(s["rows"])}
    m = _GRID_RE.match(str(s))
    if not m:
        raise ValueError(f"grid 格式应为 列x行，例如 58x53：{s}")
    cols, rows = int(m.group(1)), int(m.group(2))
    if cols <= 0 or rows <= 0:
        raise ValueError(f"grid 必须是正整数：{s}")
    return {"cols": cols, "rows": rows}


def parse_tags(v) -> list[str]:
    if v in (None, ""):
        return []
    if isinstance(v, (list, tuple)):
        items = [str(t) for t in v]
    else:
        items = _TAG_SPLIT.split(str(v))
    out: list[str] = []
    for t in items:
        t = t.strip()
        if t and t not in out:
            out.append(t)
    return out


def _clean(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v or None
    return v


def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def list_inbox(root: Path | None = None) -> list[Path]:
    """inbox/ 下的图片（不递归；按文件名自然排序）。"""
    d = ds.inbox_dir(root)
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=_natural_key)


def read_sidecar(csv_path: Path) -> list[tuple[Path, dict]]:
    """批量登记用的 sidecar CSV：必须有 `file` 列（相对 CSV 所在目录，或绝对路径），其余列名同 META_FIELDS。
    空单元格表示「用命令行参数的默认值」。"""
    csv_path = Path(csv_path)
    out = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "file" not in reader.fieldnames:
            raise CollectError([f"{csv_path}：缺少 file 列"])
        unknown = [c for c in reader.fieldnames if c not in META_FIELDS and c != "file"]
        if unknown:
            raise CollectError([f"{csv_path}：未知列 {unknown}（可用列：file, {', '.join(META_FIELDS)}）"])
        for row in reader:
            if not (row.get("file") or "").strip():
                continue
            p = Path(row["file"].strip())
            if not p.is_absolute():
                p = csv_path.parent / p
            out.append((p, {k: _clean(v) for k, v in row.items() if k != "file" and _clean(v) is not None}))
    return out


def _saved_at(v, src: Path, warnings: list[str]) -> str | None:
    if v is None:
        return None
    if v == "mtime":
        return datetime.fromtimestamp(src.stat().st_mtime).astimezone().isoformat(timespec="seconds")
    try:
        datetime.fromisoformat(v)
    except ValueError:
        warnings.append(f"{src.name}：saved_at 不是 ISO 8601 时间：{v}")
    return v


def _id_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s)[:8]


def add_images(entries: list[tuple[Path, dict]], root: Path | None = None, *, move: bool = False,
               dry_run: bool = False) -> tuple[list[dict], list[str]]:
    """登记一批图片。entries 是 [(源文件, 元数据)]。全部校验通过才复制和写 manifest（要么全成，要么全不做）。
    返回 (新记录, 警告)。失败时抛 CollectError。"""
    root = Path(root) if root is not None else formats.data_dir()
    rows = load_rows()
    by_id = {r["image_id"]: r for r in rows}
    by_sha = {r["sha256"]: r["image_id"] for r in rows}
    groups = {r["group"] for r in rows if r.get("group")}
    errors: list[str] = []
    warnings: list[str] = []
    new: list[dict] = []
    seq: dict[str, int] = {}

    def next_seq(prefix: str) -> int:
        if prefix not in seq:
            pat = re.compile(re.escape(prefix) + r"_(\d+)$")
            nums = [int(m.group(1)) for i in by_id if (m := pat.match(i))]
            seq[prefix] = max(nums) + 1 if nums else 0
        n = seq[prefix]
        seq[prefix] = n + 1
        return n

    for src, meta in entries:
        src = Path(src)
        tag = src.name
        if not src.is_file():
            errors.append(f"{src}：文件不存在")
            continue
        try:
            info = probe(src)
        except ImageInfoError as e:
            errors.append(str(e))
            continue
        sha = formats.sha256_file(src)
        if sha in by_sha:
            errors.append(f"{tag}：sha256 与已登记的 {by_sha[sha]} 重复，拒绝登记")
            continue
        meta = {k: _clean(v) for k, v in meta.items()}
        errs: list[str] = []
        pair_of = meta.get("pair_of")
        target = by_id.get(pair_of) if pair_of else None
        if pair_of and target is None:
            errs.append(f"{tag}：pair_of 指向的 {pair_of} 不在 manifest 里（先登记 App 保存的版本）")
        inherit = target if target is not None else {}
        source_path = meta.get("source_path") or ("web" if pair_of else None)
        platform = (meta.get("platform") or inherit.get("platform") or "").lower() or None
        typ = (meta.get("type") or (inherit.get("type") if source_path == "web" else None) or "").upper() or None
        if source_path not in SOURCE_PATHS:
            errs.append(f"{tag}：source_path 必须是 {'/'.join(SOURCE_PATHS)} 之一（现在是 {source_path}）")
        if source_path == "web" and not pair_of:
            errs.append(f"{tag}：Web 端版本必须用 pair_of 指明对应的 App 保存版本")
        if not platform or not re.match(r"^[a-z0-9]+$", platform):
            errs.append(f"{tag}：platform 必填，只能是小写字母和数字（如 {', '.join(KNOWN_PLATFORMS)}）")
        elif platform not in KNOWN_PLATFORMS:
            warnings.append(f"{tag}：platform={platform} 不在常用列表 {KNOWN_PLATFORMS} 里")
        if typ not in formats.TYPES:
            errs.append(f"{tag}：type 必须是 {'/'.join(formats.TYPES)} 之一（现在是 {typ}）")
        try:
            grid = parse_grid(meta.get("grid"))
        except ValueError as e:
            errs.append(f"{tag}：{e}")
            grid = None
        bucket = (meta.get("size_bucket") or "").upper() or None
        if grid is not None:
            gb = formats.size_bucket_of(grid["cols"], grid["rows"], paged=typ == "T4")
            if bucket and bucket != gb:
                warnings.append(f"{tag}：size_bucket={bucket} 与 grid 推算的 {gb} 不一致，以 grid 为准")
            bucket = gb
        if typ == "T4" and bucket != "XL":
            if bucket:
                warnings.append(f"{tag}：分页图（T4）一律计入 XL 档，size_bucket 从 {bucket} 改为 XL")
            bucket = "XL"
        if bucket is None and source_path == "web":
            bucket = inherit.get("size_bucket")
        if bucket is not None and bucket not in formats.SIZE_BUCKETS:
            errs.append(f"{tag}：size_bucket 必须是 {'/'.join(formats.SIZE_BUCKETS)} 之一")
        if bucket is None and typ != "T7" and source_path != "web":
            errs.append(f"{tag}：缺少 size_bucket（或用 grid 给出行列数自动推算）")
        tags = parse_tags(meta.get("tags"))
        for t in tags:
            if t not in formats.KNOWN_TAGS:
                warnings.append(f"{tag}：标签 {t} 不在 04 §2.1 的难点标签列表 {formats.KNOWN_TAGS} 里")
        note_id = meta.get("note_id")
        group = meta.get("group")
        if typ == "T4" and not group and platform:
            group = f"{platform}_{_id_part(note_id or sha)}"
        if group is not None and not ds.valid_id(group):
            errs.append(f"{tag}：group 只能包含字母、数字、下划线、连字符：{group}")
        image_id = meta.get("image_id")
        if image_id is None and not errs:
            if source_path == "web":
                image_id = f"{pair_of}_web"
                k = 2
                while image_id in by_id:
                    image_id = f"{pair_of}_web{k}"
                    k += 1
            else:
                prefix = f"{platform}_{_id_part(note_id or sha)}"
                image_id = f"{prefix}_{next_seq(prefix)}"
        if image_id is not None:
            if not ds.valid_id(image_id):
                errs.append(f"{tag}：image_id 只能包含字母、数字、下划线、连字符：{image_id}")
            elif image_id in by_id:
                errs.append(f"{tag}：image_id {image_id} 已存在")
            elif image_id in groups or image_id == group:
                errs.append(f"{tag}：image_id {image_id} 与某个 group id 重名（真值文件名会冲突）")
        if group is not None and group in by_id:
            errs.append(f"{tag}：group {group} 与已有的 image_id 重名（真值文件名会冲突）")
        if errs:
            errors.extend(errs)
            continue
        ext = EXT_OF[info["format"]]
        if src.suffix.lower() not in _SUFFIX_ALIASES.get(ext, (ext,)):
            warnings.append(f"{tag}：扩展名 {src.suffix} 与实际格式 {info['format']} 不符，登记为 {image_id}{ext}"
                            "（只改文件名，不转码）")
        rec = {
            "image_id": image_id, "file": f"{image_id}{ext}", "sha256": sha,
            "width": info["width"], "height": info["height"], "format": info["format"],
            "platform": platform, "note_id": note_id, "author": meta.get("author"), "url": meta.get("url"),
            "saved_at": _saved_at(meta.get("saved_at"), src, warnings), "source_path": source_path,
            "type": typ, "size_bucket": bucket, "palette_hint": (meta.get("palette_hint") or "").upper() or None,
            "tags": tags, "group": group, "notes": meta.get("notes") or "",
        }
        if pair_of:
            rec["pair_of"] = pair_of
        if grid is not None:
            rec["grid"] = grid
        if info.get("exif_orientation"):
            rec["exif_orientation"] = info["exif_orientation"]
        new.append(rec)
        by_id[image_id] = rec
        by_sha[sha] = image_id
        if group:
            groups.add(group)
        rec["_src"] = src

    if errors:
        raise CollectError(errors)
    images = ds.images_dir(root)
    for rec in new:
        if (images / rec["file"]).exists():
            raise CollectError([f"{images / rec['file']} 已存在（manifest 里没有它）：先人工确认这个文件"])
    if dry_run:
        for rec in new:
            rec.pop("_src")
        return new, warnings

    images.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    try:
        for rec in new:
            dest = images / rec["file"]
            shutil.copyfile(rec["_src"], dest)  # 原字节复制，不经过任何解码
            copied.append(dest)
            if formats.sha256_file(dest) != rec["sha256"]:
                raise CollectError([f"复制后 sha256 不一致：{dest}"])
        write_rows(rows + [{k: v for k, v in r.items() if k != "_src"} for r in new])
    except BaseException:
        for p in copied:
            p.unlink(missing_ok=True)
        raise
    for rec in new:
        src = rec.pop("_src")
        if move:
            done = src.parent / "_registered"
            done.mkdir(exist_ok=True)
            shutil.move(str(src), str(done / src.name))
    return new, warnings


# --------------------------------------------------------------------------- verify

def verify(root: Path | None = None, *, check_header: bool = True) -> tuple[list[str], list[str]]:
    """核对 manifest 与 images/：文件存在、sha256 一致、文件头尺寸一致；划分和 test.lock 一致。
    返回 (错误, 警告)。"""
    from bench.collect.lock import check_lock, lock_path  # 避免循环 import

    root = Path(root) if root is not None else formats.data_dir()
    rows = load_rows()
    errors: list[str] = []
    warnings: list[str] = []
    seen_id: set[str] = set()
    seen_sha: dict[str, str] = {}
    for r in rows:
        iid = r.get("image_id")
        for k in FIELDS[:6]:
            if r.get(k) in (None, ""):
                errors.append(f"{iid}：缺少字段 {k}")
        if iid in seen_id:
            errors.append(f"image_id 重复：{iid}")
        seen_id.add(iid)
        if r.get("sha256") in seen_sha:
            errors.append(f"{iid}：sha256 与 {seen_sha[r['sha256']]} 重复")
        seen_sha[r.get("sha256")] = iid
    by_id = {r["image_id"]: r for r in rows if r.get("image_id")}
    for r in rows:
        iid = r.get("image_id")
        if r.get("pair_of") and r["pair_of"] not in by_id:
            errors.append(f"{iid}：pair_of 指向的 {r['pair_of']} 不存在")
        if r.get("group") and r["group"] in by_id:
            errors.append(f"{iid}：group {r['group']} 与某个 image_id 重名")
        if not r.get("file"):
            continue
        p = ds.images_dir(root) / r["file"]
        if not p.is_file():
            errors.append(f"{iid}：图片不存在 {p}")
            continue
        if formats.sha256_file(p) != r.get("sha256"):
            errors.append(f"{iid}：sha256 不一致（图片被改动过？）{p}")
            continue
        if check_header:
            try:
                info = probe(p)
                if (info["width"], info["height"], info["format"]) != (r.get("width"), r.get("height"), r.get("format")):
                    errors.append(f"{iid}：文件头 {info['width']}×{info['height']} {info['format']} 与 manifest 不一致")
            except ImageInfoError as e:
                errors.append(f"{iid}：{e}")
        if r.get("format") in ds.DECODE_ONLY_FORMATS and not ds.decoded_path(iid, root).is_file():
            warnings.append(f"{iid}：{r['format']} 还没有转成 PNG（运行 python -m bench.collect convert）")
    files = {r.get("file") for r in rows}
    if ds.images_dir(root).is_dir():
        for p in sorted(ds.images_dir(root).iterdir()):
            if p.is_file() and p.name not in files:
                warnings.append(f"images/ 里有 manifest 之外的文件：{p.name}")
    dev, test = formats.read_split("dev"), formats.read_split("test")
    for name, ids in (("dev", dev), ("test", test)):
        for i in ids:
            if i not in by_id:
                errors.append(f"{name}.txt 里的 {i} 不在 manifest 里")
    both = set(dev) & set(test)
    if both:
        errors.append(f"dev 和 test 有重叠：{sorted(both)[:10]}")
    if lock_path().exists():
        errors.extend(check_lock(check_files=False))
    return errors, warnings
