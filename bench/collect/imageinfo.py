"""只读文件头，拿到图片的格式、宽高和 EXIF 方向。**不解码像素、不改动文件。**

- PNG / JPEG / WebP / GIF / BMP 等：用 PIL 的 `Image.open`（惰性，只读文件头）。
- HEIC / HEIF / AVIF：PIL 默认读不了，这里用标准库解析 ISOBMFF 的 `meta` box，
  取主图（pitm）关联的 `ispe`（尺寸）和 `irot`（旋转），不需要 pillow-heif。
"""
from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# PIL 的 format 名 → manifest 里的 format；以及登记时使用的扩展名
_PIL_FORMATS = {"PNG": "png", "JPEG": "jpeg", "MPO": "jpeg", "WEBP": "webp", "GIF": "gif", "BMP": "bmp",
                "TIFF": "tiff", "HEIF": "heic", "AVIF": "avif"}
EXT_OF = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "gif": ".gif", "bmp": ".bmp", "tiff": ".tif",
          "heic": ".heic", "heif": ".heif", "avif": ".avif"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}

_HEIC_BRANDS = {b"heic", b"heix", b"heim", b"heis", b"hevc", b"hevx"}
_AVIF_BRANDS = {b"avif", b"avis"}
_HEIF_BRANDS = _HEIC_BRANDS | _AVIF_BRANDS | {b"mif1", b"msf1"}


class ImageInfoError(Exception):
    pass


def probe(path: Path) -> dict:
    """返回 {"format", "width", "height", "exif_orientation"}。读不出来时抛 ImageInfoError。

    宽高是文件里存储的像素尺寸（JPEG 不按 EXIF 方向旋转；HEIF 已计入 irot 旋转，与解码结果一致）。
    """
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(64)
    if heif_brand(head) is not None:
        return heif_info(path)
    try:
        with Image.open(path) as im:
            fmt = _PIL_FORMATS.get((im.format or "").upper())
            if fmt is None:
                raise ImageInfoError(f"不支持的图片格式 {im.format}：{path}")
            w, h = im.size
            orient = None
            try:
                orient = im.getexif().get(0x0112)
            except Exception:  # noqa: BLE001 —— EXIF 损坏不影响登记
                orient = None
    except UnidentifiedImageError as e:
        raise ImageInfoError(f"无法识别的图片文件：{path}") from e
    return {"format": fmt, "width": int(w), "height": int(h),
            "exif_orientation": int(orient) if orient not in (None, 1) else None}


# --------------------------------------------------------------------------- HEIF（ISOBMFF）

def heif_brand(head: bytes) -> str | None:
    """文件头是 HEIF 家族时返回 heic / heif / avif，否则 None。"""
    if len(head) < 16 or head[4:8] != b"ftyp":
        return None
    size = struct.unpack(">I", head[0:4])[0]
    major = head[8:12]
    compat = [head[i:i + 4] for i in range(16, min(size, len(head)) - 3, 4)]
    brands = {major, *compat}
    if not (brands & _HEIF_BRANDS):
        return None
    for group, name in ((_HEIC_BRANDS, "heic"), (_AVIF_BRANDS, "avif")):
        if major in group:
            return name
    if brands & _HEIC_BRANDS:
        return "heic"
    if brands & _AVIF_BRANDS:
        return "avif"
    return "heif"


def _boxes(buf: bytes, start: int, end: int):
    i = start
    while i + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[i:i + 8])
        hdr = 8
        if size == 1:
            if i + 16 > end:
                return
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - i
        if size < hdr or i + size > end:
            return
        yield typ, i + hdr, i + size
        i += size


def _child(buf: bytes, start: int, end: int, typ: bytes):
    for t, s, e in _boxes(buf, start, end):
        if t == typ:
            return s, e
    return None


def heif_info(path: Path) -> dict:
    buf = Path(path).read_bytes()
    fmt = heif_brand(buf[:64])
    if fmt is None:
        raise ImageInfoError(f"不是 HEIF 文件：{path}")
    meta = _child(buf, 0, len(buf), b"meta")
    if meta is None:
        raise ImageInfoError(f"HEIF 缺少 meta box：{path}")
    ms, me = meta[0] + 4, meta[1]  # meta 是 FullBox
    pitm = _child(buf, ms, me, b"pitm")
    iprp = _child(buf, ms, me, b"iprp")
    if pitm is None or iprp is None:
        raise ImageInfoError(f"HEIF 缺少 pitm / iprp：{path}")
    ver = buf[pitm[0]]
    primary = (struct.unpack(">H", buf[pitm[0] + 4:pitm[0] + 6])[0] if ver == 0
               else struct.unpack(">I", buf[pitm[0] + 4:pitm[0] + 8])[0])
    ipco = _child(buf, iprp[0], iprp[1], b"ipco")
    if ipco is None:
        raise ImageInfoError(f"HEIF 缺少 ipco：{path}")
    props = list(_boxes(buf, ipco[0], ipco[1]))  # 属性索引从 1 开始
    assoc: list[int] = []
    for t, s, e in _boxes(buf, iprp[0], iprp[1]):
        if t != b"ipma":
            continue
        ver, flags = buf[s], int.from_bytes(buf[s + 1:s + 4], "big")
        p = s + 4
        (n,) = struct.unpack(">I", buf[p:p + 4])
        p += 4
        for _ in range(n):
            if ver < 1:
                (item,) = struct.unpack(">H", buf[p:p + 2])
                p += 2
            else:
                (item,) = struct.unpack(">I", buf[p:p + 4])
                p += 4
            cnt = buf[p]
            p += 1
            for _ in range(cnt):
                if flags & 1:
                    (v,) = struct.unpack(">H", buf[p:p + 2])
                    p += 2
                    idx = v & 0x7FFF
                else:
                    idx = buf[p] & 0x7F
                    p += 1
                if item == primary:
                    assoc.append(idx)
    size = None
    crop = None
    rot = 0
    for idx in assoc:
        if not 1 <= idx <= len(props):
            continue
        t, s, e = props[idx - 1]
        if t == b"ispe" and e - s >= 12:
            size = struct.unpack(">II", buf[s + 4:s + 12])
        elif t == b"clap" and e - s >= 32:  # clean aperture：编码时补齐到块大小，显示时裁掉
            wn, wd, hn, hd = struct.unpack(">IIII", buf[s:s + 16])
            if wd and hd:
                crop = (int(round(wn / wd)), int(round(hn / hd)))
        elif t == b"irot" and e - s >= 1:
            rot = buf[s] & 3
    if size is None:
        raise ImageInfoError(f"HEIF 主图没有 ispe（尺寸）属性：{path}")
    w, h = crop if crop else size
    if rot in (1, 3):
        w, h = h, w
    return {"format": fmt, "width": int(w), "height": int(h), "exif_orientation": None}
