"""合成生成器：真值格式、图例校验和、附加标注一致性、确定性。"""
import hashlib
from collections import Counter

import pytest

from bench.common.formats import load_json, read_jsonl, size_bucket_of, validate_label
from bench.synth.generate import TYPES, GenOptions, Generator


@pytest.fixture(scope="module")
def mixed(tmp_path_factory):
    """每种类型、每种风格都覆盖到的一批样本（边长限制在 48 以内，测试快一些）。"""
    out = tmp_path_factory.mktemp("synth_mixed")
    rows = []
    k = 0
    for t in TYPES:
        gen = Generator(GenOptions(types=(t,), max_side=48, allow_fallback_font=True))
        for _ in range(5 if t in ("T1", "T4") else 3):
            rows.extend(gen.generate(11, k, out))
            k += 1
    gen = Generator(GenOptions(types=("T1", "T2", "T3"), allow_fallback_font=True,
                               buckets=("L", "XL")))  # 再来几张不限尺寸的大图
    for _ in range(3):
        rows.extend(gen.generate(11, k, out))
        k += 1
    return out, rows


def _labels(out):
    return [load_json(p) for p in sorted((out / "labels").glob("*.json"))]


def test_labels_validate(mixed):
    out, _ = mixed
    labs = _labels(out)
    assert {lab["type"] for lab in labs} == set(TYPES)
    for lab in labs:
        assert validate_label(lab) == [], lab["image_id"]
        assert lab["size_bucket"] == size_bucket_of(lab["grid"]["cols"], lab["grid"]["rows"], paged=lab["type"] == "T4")
        assert lab["source"]["platform"] == "synth"
        for im in lab["images"]:
            data = (out / "images" / im["file"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == im["sha256"]
        n_beads = sum(v is not None for row in lab["cells"] for v in row)
        if lab["legend"]:
            assert sum(it["count"] for it in lab["legend"]) == n_beads


def test_paged_labels_cover_grid_once(mixed):
    out, _ = mixed
    for lab in _labels(out):
        if lab["type"] != "T4":
            continue
        assert len(lab["images"]) >= 2
        cover = Counter()
        for p in lab["pages"]:
            for r in range(p["row_offset"], p["row_offset"] + p["rows"]):
                for c in range(p["col_offset"], p["col_offset"] + p["cols"]):
                    cover[(r, c)] += 1
        assert len(cover) == lab["grid"]["rows"] * lab["grid"]["cols"]
        assert set(cover.values()) == {1}


def test_extra_consistent(mixed):
    out, _ = mixed
    for lab in _labels(out):
        ex = load_json(out / "extra" / f"{lab['image_id']}.extra.json")
        assert ex["image_id"] == lab["image_id"]
        assert len(ex["images"]) == len(lab["images"])
        counts = Counter(v for row in lab["cells"] for v in row if v is not None)
        assert {e["code"]: e["count"] for e in ex["codebook"] if e["count"]} == dict(counts)
        rows, cols = lab["grid"]["rows"], lab["grid"]["cols"]
        assert len(ex["cells"]["has_text"]) == rows and all(len(s) == cols for s in ex["cells"]["has_text"])
        assert len(ex["cells"]["text_color"]) == rows
        for r in range(rows):
            for c in range(cols):
                tc = ex["cells"]["text_color"][r][c]
                assert (tc != ".") == (ex["cells"]["has_text"][r][c] == "1")
                if lab["cells"][r][c] is None:
                    assert tc == "."
        if any(s.count("1") for s in ex["cells"]["has_text"]):
            assert set(ex["cell_text_boxes"]) == set(counts)
        # 图例条目（附加标注）和真值图例一致；坐标文字和格子序号一致
        leg = [it for im in ex["images"] for it in im["legend"] if it.get("panel", 0) == 0]
        if lab["legend"]:
            assert sorted((it["code"], it["count"]) for it in leg) == sorted((it["code"], it["count"]) for it in lab["legend"])
            for it in leg:
                assert str(it["count"]) in it["count_text"]
                assert it["code_text"] == it["code"]
                for k in ("swatch_box", "code_box", "count_box"):
                    x0, y0, x1, y1 = it[k]
                    assert x1 > x0 and y1 > y0
        start = ex["style"]["coord_start"]
        for im in ex["images"]:
            for cd in im["coords"]:
                assert cd["text"] == str(cd["index"] + start)
                if cd.get("panel", 0) == 0:
                    assert 0 <= cd["index"] < (cols if cd["axis"] == "col" else rows)


def test_manifest_rows(mixed):
    out, rows = mixed
    for r in rows:
        assert r["platform"] == "synth" and r["source_path"] in ("own_export", "photo")
        assert (out / "images" / r["file"]).exists()
    groups = Counter(r["group"] for r in rows if r["group"])
    assert groups and all(v >= 2 for v in groups.values())


def test_deterministic(tmp_path):
    def run(d):
        gen = Generator(GenOptions(max_side=40, allow_fallback_font=True))
        for i in range(8):
            gen.generate(5, i, d)
        return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert a.keys() == b.keys() and len(a) >= 24
    for k in a:
        assert a[k] == b[k], k
    gen = Generator(GenOptions(max_side=40, allow_fallback_font=True))
    gen.generate(6, 0, tmp_path / "c")
    lab_c = (tmp_path / "c" / "labels" / "synth_s6_000000.json").read_bytes()
    assert lab_c != a["labels/synth_s5_000000.json"]


def test_without_downloaded_fonts(tmp_path, monkeypatch):
    """CI 上没有字体二进制：--allow-fallback-font 时所有类型都能生成；不加时明确报错。"""
    from bench.synth import fonts as F
    monkeypatch.setattr(F, "available_faces", lambda: ())
    with pytest.raises(SystemExit):
        Generator(GenOptions())
    gen = Generator(GenOptions(max_side=32, allow_fallback_font=True))
    for i, t in enumerate(TYPES):
        gen.opts.types = (t,)
        gen.generate(8, i, tmp_path)
    labs = _labels(tmp_path)
    assert len(labs) == len(TYPES) and all(validate_label(lab) == [] for lab in labs)


def test_cli_generate_and_sheet(tmp_path):
    from bench.synth.__main__ import main
    out = tmp_path / "cli"
    assert main(["generate", "--n", "3", "--out", str(out), "--seed", "2", "--type", "T1,T2",
                 "--size-bucket", "S", "--allow-fallback-font"]) == 0
    assert len(read_jsonl(out / "manifest.jsonl")) == 3
    assert main(["sheet", "--in", str(out), "--n", "3", "--overlay"]) == 0
    assert (out / "contact_sheet_overlay.png").exists()
    assert main(["generate", "--n", "1", "--out", str(out), "--seed", "2", "--type", "T9"]) == 2
