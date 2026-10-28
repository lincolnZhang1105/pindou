"""合成数据不得使用 palettes_unverified 的色值（bench/README.md「色卡」、palettes/SOURCES.md §2）。"""
import dataclasses
import json

import numpy as np
import pytest

from bench.common.formats import load_json
from bench.synth import palettes as P
from bench.synth.generate import GenOptions, Generator

SENTINEL = [13, 254, 77]


def _fake_bench(tmp_path):
    """伪造一份 bench 色卡：存疑色卡的所有色值都是哨兵色。"""
    pal = []
    for pid, codes in (("MARD-221", [f"{s}{i}" for s in "ABCDEFGH" for i in range(1, 25)]),
                       ("COCO-291", [f"Z{i:02d}" for i in range(1, 80)])):
        pal.append({"id": pid, "brand": "x", "verified": False,
                    "colors": [{"code": c, "rgb": SENTINEL, "lab": [90, -80, 70]} for c in codes]})
    p = tmp_path / "palettes_bench.json"
    p.write_text(json.dumps({"palettes": pal}), encoding="utf-8")
    return p


def test_code_table_has_no_color_fields(tmp_path):
    tables = P.load_code_tables(_fake_bench(tmp_path))
    t = tables["MARD-221"]
    assert {f.name for f in dataclasses.fields(P.CodeTable)} == {"id", "brand", "codes", "clean"}
    assert not hasattr(t, "rgb") and not hasattr(t, "lab")
    assert t.clean is False and t.codes[:2] == ("A1", "A2")
    assert tables["ARTKAL-S"].clean is True


def test_clean_loader_rejects_unverified(tmp_path):
    bad = tmp_path / "palettes.json"
    bad.write_text(json.dumps({"palettes": [{"id": "X", "verified": False,
                                             "colors": [{"code": "A1", "rgb": [1, 2, 3]}]}]}), encoding="utf-8")
    with pytest.raises(P.PaletteError):
        P.load_clean_palettes(bad)


def test_generated_colors_never_from_unverified(tmp_path, monkeypatch):
    fake = _fake_bench(tmp_path)
    # 让 text 方案一律选伪造的 MARD-221，确保「只用色号文字」这条路径被走到
    monkeypatch.setattr(P, "TEXT_SCHEMES", {"MARD-221": 1.0})
    opts = GenOptions(types=("T1", "T2", "T4"), max_side=40, allow_fallback_font=True, bench_path=fake,
                      degrade={"fmt": "png", "resize": 0, "watermark": False})
    gen = Generator(opts)
    out = tmp_path / "out"
    clean = P.load_clean_palettes()
    used_fake = 0
    for i in range(14):
        gen.generate(3, i, out)
    for ex_p in sorted((out / "extra").glob("*.extra.json")):
        ex = load_json(ex_p)
        lab = load_json(out / "labels" / ex_p.name.replace(".extra.json", ".json"))
        if lab["palette_hint"] == "MARD-221":
            used_fake += 1
            assert ex["palette"]["native"] is False
        for e in ex["codebook"]:
            assert e["rgb"] != SENTINEL, ex_p.name
            src = e["source"]
            if src.startswith("palette:"):
                _, pid, code = src.split(":", 2)
                assert pid in clean, src                       # 只能是干净色卡
                k = clean[pid].codes.index(code)
                assert list(clean[pid].rgb[k]) == e["rgb"]
            else:
                assert src in ("random", "empty_bg"), src
    assert used_fake >= 3


def test_color_pool_only_clean():
    pool = P.clean_color_pool()
    clean = P.load_clean_palettes()
    ids = {s.split(":")[1] for s in pool.sources}
    assert ids <= set(clean)
    assert len(pool.rgb) == len(pool.sources) == len({tuple(c) for c in pool.rgb.tolist()})
    assert np.all(pool.rgb >= 0)
