"""bench.collect 的测试：在临时目录里设置 PINDOU_DATA_DIR，用 PIL 现场生成小图。"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from PIL import Image

from bench.collect.__main__ import main
from bench.collect.imageinfo import heif_info, probe
from bench.collect.lock import check_lock
from bench.collect.manifest import CollectError, add_images, verify
from bench.collect.split import plan_split
from bench.common import dataset as ds
from bench.common import formats


@pytest.fixture
def env(tmp_path, monkeypatch):
    """私有数据目录 + 临时 bench/datasets（monkeypatch formats.DATASETS / MANIFEST）。"""
    data = tmp_path / "data"
    (data / "inbox").mkdir(parents=True)
    dsets = tmp_path / "datasets"
    dsets.mkdir()
    monkeypatch.setenv("PINDOU_DATA_DIR", str(data))
    monkeypatch.setattr(formats, "DATASETS", dsets)
    monkeypatch.setattr(formats, "MANIFEST", dsets / "manifest.jsonl")
    return data, dsets


def make_img(path, w=40, h=30, seed=0, fmt=None, **kw):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path, format=fmt, **kw)
    return path


def manifest_rows(dsets):
    return formats.read_jsonl(dsets / "manifest.jsonl")


# --------------------------------------------------------------------------- add

def test_add_copies_original_bytes(env):
    data, dsets = env
    src = make_img(data / "inbox" / "a.png", 50, 20)
    rc = main(["add", str(src), "--platform", "xhs", "--note-id", "6aa72c3e000000002901a806", "--author", "画师",
               "--url", "https://www.xiaohongshu.com/explore/6aa72c3e000000002901a806", "--saved-at",
               "2026-10-28T10:00:00+08:00", "--source-path", "app_save", "--type", "T1", "--grid", "58x53",
               "--palette-hint", "mard-221", "--tags", "watermark,white_vs_empty"])
    assert rc == 0
    rows = manifest_rows(dsets)
    assert len(rows) == 1
    r = rows[0]
    assert r["image_id"] == "xhs_6aa72c3e_0"
    assert r["file"] == "xhs_6aa72c3e_0.png"
    assert (r["width"], r["height"], r["format"]) == (50, 20, "png")
    assert r["size_bucket"] == "M" and r["grid"] == {"cols": 58, "rows": 53}
    assert r["palette_hint"] == "MARD-221"
    assert r["tags"] == ["watermark", "white_vs_empty"]
    assert r["group"] is None and r["source_path"] == "app_save"
    dest = data / "images" / r["file"]
    assert dest.read_bytes() == src.read_bytes()          # 原字节复制
    assert r["sha256"] == formats.sha256_file(src)
    assert list(r)[:18] == ["image_id", "file", "sha256", "width", "height", "format", "platform", "note_id",
                            "author", "url", "saved_at", "source_path", "type", "size_bucket", "palette_hint",
                            "tags", "group", "notes"]


def test_add_rejects_duplicate_sha_atomically(env):
    data, dsets = env
    a = make_img(data / "inbox" / "a.png", seed=1)
    add_images([(a, {"platform": "xhs", "note_id": "n1", "source_path": "app_save", "type": "T1",
                     "size_bucket": "S"})], data)
    b = make_img(data / "inbox" / "b.png", seed=2)
    dup = data / "inbox" / "dup.png"
    dup.write_bytes(a.read_bytes())
    with pytest.raises(CollectError) as ei:
        add_images([(b, {"platform": "xhs", "note_id": "n2", "source_path": "app_save", "type": "T1",
                         "size_bucket": "S"}),
                    (dup, {"platform": "xhs", "note_id": "n3", "source_path": "app_save", "type": "T1",
                           "size_bucket": "S"})], data)
    assert any("重复" in e for e in ei.value.errors)
    # 整批都不登记
    assert len(manifest_rows(dsets)) == 1
    assert sorted(p.name for p in (data / "images").iterdir()) == ["xhs_n1_0.png"]


def test_add_inbox_sequence_and_format_detection(env):
    data, dsets = env
    inbox = data / "inbox"
    make_img(inbox / "img_2.png", seed=3)
    make_img(inbox / "img_10.png", seed=4)
    make_img(inbox / "fake.jpg", seed=5, fmt="WEBP", lossless=True)   # 扩展名和实际格式不符
    rc = main(["add", "--platform", "xhs", "--note-id", "abcdef1234", "--source-path", "app_save", "--type", "T1",
               "--size-bucket", "s", "--move"])
    assert rc == 0
    rows = manifest_rows(dsets)
    # 自然排序：fake.jpg, img_2, img_10
    assert [r["image_id"] for r in rows] == ["xhs_abcdef12_0", "xhs_abcdef12_1", "xhs_abcdef12_2"]
    assert rows[0]["format"] == "webp" and rows[0]["file"] == "xhs_abcdef12_0.webp"
    assert (inbox / "_registered" / "img_10.png").exists() and not (inbox / "img_10.png").exists()
    # 序号接着已有的继续
    make_img(inbox / "more.png", seed=6)
    assert main(["add", "--platform", "xhs", "--note-id", "abcdef1234", "--source-path", "app_save",
                 "--type", "T1", "--size-bucket", "S"]) == 0
    assert manifest_rows(dsets)[-1]["image_id"] == "xhs_abcdef12_3"


def test_add_validation_errors(env, capsys):
    data, dsets = env
    src = make_img(data / "inbox" / "a.png")
    assert main(["add", str(src), "--platform", "xhs", "--source-path", "scraped", "--type", "T9"]) == 1
    err = capsys.readouterr().err
    assert "source_path" in err and "type" in err
    assert not (dsets / "manifest.jsonl").exists()
    # 非 T7 必须给 size_bucket 或 grid
    assert main(["add", str(src), "--platform", "xhs", "--source-path", "app_save", "--type", "T1"]) == 1
    # T7 可以不给
    assert main(["add", str(src), "--platform", "xhs", "--source-path", "app_save", "--type", "T7"]) == 0


def test_add_csv_t4_group_and_jpeg_orientation(env):
    data, dsets = env
    inbox = data / "inbox"
    for i in range(3):
        make_img(inbox / f"p{i}.png", seed=10 + i)
    exif = Image.Exif()
    exif[0x0112] = 6
    make_img(inbox / "photo.jpg", 30, 20, seed=20, fmt="JPEG", exif=exif.tobytes())
    (inbox / "batch.csv").write_text(
        "file,note_id,type,size_bucket,tags,source_path,platform\n"
        "p0.png,6bb00000aaaa,T4,,watermark;blurry,,\n"
        "p1.png,6bb00000aaaa,T4,M,,,\n"
        "p2.png,6bb00000aaaa,T4,,,,\n"
        "photo.jpg,,T6,L,,photo,photo\n", encoding="utf-8")
    rc = main(["add", "--csv", str(inbox / "batch.csv"), "--platform", "xhs", "--source-path", "app_save"])
    assert rc == 0
    rows = {r["image_id"]: r for r in manifest_rows(dsets)}
    pages = [rows[f"xhs_6bb00000_{i}"] for i in range(3)]
    assert all(p["group"] == "xhs_6bb00000" and p["size_bucket"] == "XL" for p in pages)
    assert pages[0]["tags"] == ["watermark", "blurry"]
    photo = next(r for r in rows.values() if r["type"] == "T6")
    assert photo["platform"] == "photo" and photo["format"] == "jpeg" and photo["exif_orientation"] == 6
    assert (photo["width"], photo["height"]) == (30, 20)   # 存储尺寸，不按 EXIF 旋转
    units = ds.build_units(formats.load_manifest(dsets / "manifest.jsonl"))
    assert [p["image_id"] for p in units["xhs_6bb00000"]] == ["xhs_6bb00000_0", "xhs_6bb00000_1", "xhs_6bb00000_2"]


# --------------------------------------------------------------------------- HEIF

def _box(t: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", 8 + len(payload), t) + payload


def _fullbox(t: bytes, payload: bytes, version=0, flags=0) -> bytes:
    return _box(t, bytes([version]) + flags.to_bytes(3, "big") + payload)


def synthetic_heic(w=4032, h=3024, rot=1) -> bytes:
    ftyp = _box(b"ftyp", b"mif1" + b"\0\0\0\0" + b"mif1heic")
    pitm = _fullbox(b"pitm", struct.pack(">H", 7))
    ipco = _box(b"ipco", _fullbox(b"hvcC", b"") + _fullbox(b"ispe", struct.pack(">II", w, h)) + _box(b"irot", bytes([rot])))
    # item 7 关联属性 2（ispe，essential）和 3（irot）
    ipma = _fullbox(b"ipma", struct.pack(">I", 1) + struct.pack(">H", 7) + bytes([2, 0x82, 0x03]))
    meta = _fullbox(b"meta", _fullbox(b"hdlr", b"\0" * 20) + pitm + _box(b"iprp", ipco + ipma))
    return ftyp + meta + _box(b"mdat", b"\0" * 16)


def test_heif_header_parser(tmp_path):
    p = tmp_path / "x.heic"
    p.write_bytes(synthetic_heic(rot=1))
    assert heif_info(p) == {"format": "heic", "width": 3024, "height": 4032, "exif_orientation": None}
    p.write_bytes(synthetic_heic(rot=2))
    assert probe(p)["width"] == 4032


def test_add_heic_without_decoder(env):
    data, dsets = env
    src = data / "inbox" / "IMG_0001.HEIC"
    src.write_bytes(synthetic_heic(rot=0))
    assert main(["add", str(src), "--platform", "photo", "--source-path", "photo", "--type", "T6",
                 "--size-bucket", "M"]) == 0
    r = manifest_rows(dsets)[0]
    assert r["format"] == "heic" and r["file"].endswith(".heic") and (r["width"], r["height"]) == (4032, 3024)
    errors, warnings = verify()
    assert errors == [] and any("convert" in w for w in warnings)


def test_convert_heic_to_png(env):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    data, dsets = env
    src = make_img(data / "inbox" / "a.heic", 60, 40, seed=7, fmt="HEIF")
    assert main(["add", str(src), "--platform", "photo", "--source-path", "photo", "--type", "T6",
                 "--size-bucket", "S"]) == 0
    r = manifest_rows(dsets)[0]
    assert (r["width"], r["height"], r["format"]) == (60, 40, "heic")
    assert main(["convert"]) == 0
    out = ds.decoded_path(r["image_id"])
    with Image.open(out) as im:
        assert im.format == "PNG" and im.size == (60, 40)
        png = np.asarray(im.convert("RGB"))
    with Image.open(ds.image_path(r)) as im:
        assert np.array_equal(png, np.asarray(im.convert("RGB")))   # PNG 与 HEIC 解码结果逐像素一致
    assert ds.readable_image_path(r) == out
    assert (data / "images" / r["file"]).read_bytes() == src.read_bytes()   # 原图不动


# --------------------------------------------------------------------------- verify

def test_verify_detects_tampering(env, capsys):
    data, dsets = env
    src = make_img(data / "inbox" / "a.png", seed=8)
    add_images([(src, {"platform": "xhs", "note_id": "n1", "source_path": "app_save", "type": "T1",
                       "size_bucket": "S"})], data)
    assert main(["verify"]) == 0
    f = data / "images" / "xhs_n1_0.png"
    make_img(f, seed=9)  # 改动图片
    (data / "images" / "stray.png").write_bytes(b"x")
    errors, warnings = verify()
    assert any("sha256 不一致" in e for e in errors)
    assert any("stray.png" in w for w in warnings)
    f.unlink()
    errors, _ = verify()
    assert any("不存在" in e for e in errors)
    assert main(["verify"]) == 1


# --------------------------------------------------------------------------- split / lock

def _fake_manifest(n_per_stratum=None):
    """直接构造 manifest（不需要真实图片）：各分层若干单元 + 一套 T4 + 一张 T7 + 一张 Web 版本。"""
    m = {}
    n_per_stratum = n_per_stratum or {("S", "T1"): 6, ("M", "T1"): 7, ("L", "T2"): 4, ("M", "T6"): 1}
    k = 0
    for (b, t), n in n_per_stratum.items():
        for _ in range(n):
            iid = f"xhs_{k:08d}_0"
            m[iid] = {"image_id": iid, "file": iid + ".png", "sha256": f"{k:064x}", "type": t, "size_bucket": b,
                      "source_path": "app_save", "group": None, "tags": []}
            k += 1
    for i in range(4):
        iid = f"xhs_t4000000_{i}"
        m[iid] = {"image_id": iid, "file": iid + ".png", "sha256": f"{1000 + i:064x}", "type": "T4",
                  "size_bucket": "XL", "source_path": "app_save", "group": "xhs_t4000000", "tags": []}
    m["xhs_t7000000_0"] = {"image_id": "xhs_t7000000_0", "file": "t7.png", "sha256": "f" * 64, "type": "T7",
                           "size_bucket": None, "source_path": "app_save", "group": None, "tags": []}
    m["xhs_00000000_0_web"] = {"image_id": "xhs_00000000_0_web", "file": "w.webp", "sha256": "e" * 64,
                               "type": "T1", "size_bucket": "S", "source_path": "web",
                               "pair_of": "xhs_00000000_0", "group": None, "tags": []}
    return m


def test_split_stratified_deterministic():
    m = _fake_manifest()
    dev, test, warnings = plan_split(m, [], [], seed=1)
    dev2, test2, _ = plan_split(m, [], [], seed=1)
    assert (dev, test) == (dev2, test2)
    assert not set(dev) & set(test)
    assert "xhs_t7000000_0" not in dev + test and "xhs_00000000_0_web" not in dev + test
    t4 = {f"xhs_t4000000_{i}" for i in range(4)}
    assert t4 <= set(test) or t4 <= set(dev)          # 同组不拆开
    assert t4 <= set(test)                            # 单元数 1 的分层归 test
    for (b, t), n in {("S", "T1"): 6, ("M", "T1"): 7, ("L", "T2"): 4}.items():
        n_test = sum(1 for i in test if m[i]["size_bucket"] == b and m[i]["type"] == t)
        assert n_test == (2 * n + 1) // 3
    assert any(m[i]["type"] == "T6" for i in test)
    dev3, test3, _ = plan_split(m, [], [], seed=2)
    assert set(test3) != set(test)                    # 换种子结果不同


def test_split_incremental_keeps_existing():
    m = _fake_manifest()
    dev, test, _ = plan_split(m, [], [], seed=1)
    m2 = dict(m)
    for k in range(6):   # 补充 6 个 S/T1
        iid = f"dy_{k:08d}_0"
        m2[iid] = {"image_id": iid, "file": iid + ".png", "sha256": f"{5000 + k:064x}", "type": "T1",
                   "size_bucket": "S", "source_path": "app_save", "group": None, "tags": []}
    dev2, test2, warnings = plan_split(m2, dev, test, seed=1)
    assert set(dev) <= set(dev2) and set(test) <= set(test2)   # 已有的划分不动
    s_test = sum(1 for i in test2 if m2[i]["size_bucket"] == "S" and m2[i]["type"] == "T1")
    assert s_test == 8                                         # 12 个单元 → test 8
    # manifest 里删掉的 id 会被丢弃并警告
    dev3, test3, warnings = plan_split(m, dev + ["gone_0"], test, seed=1)
    assert "gone_0" not in dev3 and any("gone_0" in w for w in warnings)


def _register_many(data, n=6):
    entries = []
    for i in range(n):
        p = make_img(data / "inbox" / f"x{i}.png", seed=100 + i)
        entries.append((p, {"platform": "xhs", "note_id": f"note{i:04d}", "source_path": "app_save",
                            "type": "T1", "size_bucket": "S" if i % 2 else "M"}))
    add_images(entries, data)


def test_split_lock_and_check_lock(env, capsys):
    data, dsets = env
    _register_many(data)
    assert main(["split"]) == 0
    test_ids = formats.read_split("test")
    dev_ids = formats.read_split("dev")
    assert len(test_ids) == 4 and len(dev_ids) == 2
    assert (dsets / "test.txt").read_text(encoding="utf-8").startswith("#")
    assert check_lock() and "不存在" in check_lock()[0]
    # 没有真值时默认拒绝锁定
    assert main(["lock-test"]) == 1
    assert main(["lock-test", "--allow-unlabeled"]) == 0
    lock = json.loads((dsets / "test.lock").read_text(encoding="utf-8"))
    assert {e["image_id"] for e in lock["entries"]} == set(test_ids)
    import hashlib
    assert lock["ids_sha256"] == hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest()
    assert check_lock() == []
    from bench.collect import check_lock as exported
    assert exported() == []
    # 锁定后：不能再锁、不能再划分
    assert main(["lock-test", "--allow-unlabeled"]) == 1
    assert main(["split"]) == 1
    # 改动 test.txt → 校验失败
    p = dsets / "test.txt"
    orig = p.read_text(encoding="utf-8")
    p.write_text(orig + dev_ids[0] + "\n", encoding="utf-8")
    assert any("ids_sha256" in e for e in check_lock())
    p.write_text(orig, encoding="utf-8")
    assert check_lock() == []
    # 改动图片文件 → 校验失败
    rec = formats.load_manifest(dsets / "manifest.jsonl")[test_ids[0]]
    make_img(data / "images" / rec["file"], seed=999)
    assert any("sha256" in e for e in check_lock())
    assert check_lock(check_files=False) == []
    assert main(["verify"]) == 1


def test_lock_requires_valid_labels(env):
    data, dsets = env
    _register_many(data, 3)
    assert main(["split"]) == 0
    for key in formats.read_split("test"):
        lab = {"image_id": key, "type": "T1", "images": [{"file": key + ".png", "sha256": ""}],
               "grid": {"cols": 1, "rows": 1},
               "pages": [{"image_index": 0, "bbox": [0, 0, 1, 1], "x_lines": [0, 1], "y_lines": [0, 1],
                          "col_offset": 0, "row_offset": 0, "cols": 1, "rows": 1}],
               "legend": [{"code": "A1", "count": 2}], "cells": [["A1"]]}
        formats.save_json(ds.label_path(key), lab)
    assert main(["lock-test"]) == 1           # 图例校验不通过
    for key in formats.read_split("test"):
        lab = formats.load_json(ds.label_path(key))
        lab["legend"][0]["count"] = 1
        formats.save_json(ds.label_path(key), lab)
    assert main(["lock-test"]) == 0


# --------------------------------------------------------------------------- stats / resolution

def test_stats_reports_coverage(env, capsys):
    data, dsets = env
    _register_many(data, 4)
    assert main(["split"]) == 0
    capsys.readouterr()
    assert main(["stats"]) == 0
    out = capsys.readouterr().out
    assert "## 尺寸档" in out and "## 类型" in out and "## 品牌" in out and "## 难点标签" in out
    assert "样本数 4 < 150" in out
    assert "test 集里 T2 只有 0 个" in out
    assert "test 集里 Hard 只有 0 个" in out


def test_resolution_pairs(env, capsys):
    data, dsets = env
    app = make_img(data / "inbox" / "app.png", 120, 90, seed=31)
    web = make_img(data / "inbox" / "web.webp", 60, 45, seed=32, fmt="WEBP", quality=80)
    assert main(["add", str(app), "--platform", "xhs", "--note-id", "6aa72c3e00", "--source-path", "app_save",
                 "--type", "T1", "--grid", "30x20"]) == 0
    assert main(["add", str(web), "--pair-of", "xhs_6aa72c3e_0"]) == 0
    rows = manifest_rows(dsets)
    assert rows[1]["image_id"] == "xhs_6aa72c3e_0_web" and rows[1]["source_path"] == "web"
    assert rows[1]["type"] == "T1" and rows[1]["platform"] == "xhs"
    capsys.readouterr()
    assert main(["resolution"]) == 0
    out = capsys.readouterr().out
    assert "| xhs_6aa72c3e_0 | 120×90 | png |" in out
    assert "60×45 | webp" in out and "| 2.00 |" in out and "≤ 4.0×4.5" in out and "≤ 2.0×2.2" in out
    # Web 版本不进入划分和统计
    assert main(["split"]) == 0
    assert "xhs_6aa72c3e_0_web" not in formats.read_split("dev") + formats.read_split("test")
