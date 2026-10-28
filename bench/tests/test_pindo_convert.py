"""bench.competitors.pindo：输出转换（纯函数）和 run 子命令的单元约定。不需要 Node；
有打包好的 bundle 时另跑一次端到端。"""
import json
import shutil

import pytest
from PIL import Image, ImageDraw

from bench.competitors.pindo import convert as C
from bench.competitors.pindo.__main__ import bundle_path, load_lock, main


def _out(codes, cols, rows, white="T01", ok=True):
    return {"key": "k", "ok": ok, "grid_source": "auto", "cols": cols, "rows": rows, "codes": codes,
            "white_code": white, "palette_size": 291, "ms": 12.5}


def test_brand_mapping():
    assert C.pindo_brand("MARD-221") == "mard"
    assert C.pindo_brand("mard-291") == "mard"
    assert C.pindo_brand("COCO-291") == "coco"
    assert C.pindo_brand("HAMA-MIDI") == "hama"
    assert C.pindo_brand("ARTKAL-S") == "artkal-s"
    assert C.pindo_brand(None) == "mard"          # Pindo 网页默认品牌
    assert C.pindo_brand("ARTKAL-A") is None      # Pindo 没有这套色卡
    assert C.pindo_brand("PERLER") is None        # 编号体系不同（P01 vs 80-15xxx）
    assert C.pindo_brand("YANT") is None


def test_size_hint_sources():
    pages = [{"grid": {"cols": 58, "rows": 53}}]
    assert C.size_hint(pages, "manifest") == ((58, 53), "manifest")
    assert C.size_hint([{}], "manifest") == (C.DEFAULT_SIZE, "default")
    assert C.size_hint(pages, "default") == (C.DEFAULT_SIZE, "default")
    assert C.size_hint([{}], "label", {"grid": {"cols": 16, "rows": 16}}) == ((16, 16), "label")


def test_to_prediction_codebook_and_cells():
    p = C.to_prediction(_out(["A01", "T01", "A01", "H07", None, "T01"], 3, 2), "MARD-221")
    assert p["grid"] == {"cols": 3, "rows": 2}
    assert [e["code"] for e in p["codebook"]] == ["A01", "T01", "H07"]
    assert all(e["palette_id"] == "MARD-221" for e in p["codebook"])
    assert p["cells"] == [1, 2, 1, 3, 0, 2]
    assert p["legend"] == [] and p["legend_check"]["unavailable"]
    assert p["timing_ms"] == {"total": 12.5}
    assert "uncertain" not in p   # Pindo 没有不确定标记，评测器按「没有标记」处理


def test_white_as_empty_drops_white_code():
    p = C.to_prediction(_out(["A01", "T01", "T01", "H07"], 2, 2), "MARD-221", white_as_empty=True)
    assert [e["code"] for e in p["codebook"]] == ["A01", "H07"]
    assert p["cells"] == [1, 0, 0, 2]
    assert p["pindo"]["white_as_empty"] is True


def test_to_prediction_errors():
    assert C.to_prediction(_out([], 0, 0, ok=False) | {"error": "boom"}, None)["error"]["code"] == "internal"
    bad = C.to_prediction(_out(["A01"], 2, 2), None)
    assert bad["error"]["code"] == "internal" and "格子数" in bad["error"]["message"]
    # 没有 palette_hint 时 codebook 写 Pindo 默认品牌
    assert C.to_prediction(_out(["A01"], 1, 1), None)["codebook"][0]["palette_id"] == "MARD"


def test_prediction_scores_with_evaluator_codes():
    """A01（Pindo 写法）和真值的 A1 在评测器里是同一个色号。"""
    from bench.eval.codes import CodeMapper
    m = CodeMapper("MARD-221")
    p = C.to_prediction(_out(["A01"], 1, 1), "MARD-221")
    assert m.pred_key(p["codebook"][0]["palette_id"], p["codebook"][0]["code"]) == m.raw_key("A1")


def _write_dataset(root, recs):
    (root / "images").mkdir(parents=True)
    rows = []
    for rec in recs:
        im = Image.new("RGB", (120, 120), "white")
        d = ImageDraw.Draw(im)
        for i in range(0, 121, 20):
            d.line([i, 0, i, 119], fill=(180, 180, 180))
            d.line([0, i, 119, i], fill=(180, 180, 180))
        d.rectangle([21, 21, 59, 59], fill=(200, 30, 40))
        im.save(root / "images" / rec["file"])
        rows.append({"image_id": rec["image_id"], "file": rec["file"], "platform": "synth", "source_path": "own_export",
                     "type": rec.get("type", "T1"), "palette_hint": rec.get("hint"), "group": rec.get("group")})
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root / "manifest.jsonl"


def test_run_marks_unsupported_without_node(tmp_path, monkeypatch):
    """分页图和不支持的色卡不需要调用 Node，直接写成失败。"""
    if not bundle_path(load_lock()).exists() or shutil.which("node") is None:
        pytest.skip("没有打包好的 Pindo bundle（python -m bench.competitors.pindo setup）")
    man = _write_dataset(tmp_path, [
        {"image_id": "g1_1", "file": "g1_1.png", "group": "g1", "type": "T4", "hint": "MARD-221"},
        {"image_id": "g1_2", "file": "g1_2.png", "group": "g1", "type": "T4", "hint": "MARD-221"},
        {"image_id": "art", "file": "art.png", "hint": "ARTKAL-A"},
        {"image_id": "ok1", "file": "ok1.png", "hint": "MARD-221"},
    ])
    monkeypatch.setenv("PINDOU_DATA_DIR", str(tmp_path))
    assert main(["run", "--ids", "g1_1", "g1_2", "art", "ok1", "--run", "t", "--manifest", str(man),
                 "--white-as-empty", "--jobs", "1"]) == 0
    preds = tmp_path / "preds"
    for run in ("t", "t-wae"):
        assert json.loads((preds / run / "g1.json").read_text(encoding="utf-8"))["error"]["code"] == "unsupported"
        assert json.loads((preds / run / "art.json").read_text(encoding="utf-8"))["error"]["code"] == "unsupported_palette"
        ok = json.loads((preds / run / "ok1.json").read_text(encoding="utf-8"))
        assert "error" not in ok and len(ok["cells"]) == ok["grid"]["cols"] * ok["grid"]["rows"]
        meta = json.loads((preds / run / "_run.json").read_text(encoding="utf-8"))
        assert meta["engine"] == "pindo" and meta["pindo"]["license"] == "GPL-3.0-only"
        assert meta["options"]["white_as_empty"] == (run == "t-wae")
