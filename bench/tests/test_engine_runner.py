"""bench.engine 端到端：现场画一张小图纸，用 beadcore-cli 跑一遍，检查预测结果文件和命名约定。
没有编译好的 CLI 时跳过。"""
import json

import pytest
from PIL import Image, ImageDraw

from bench.common import formats as F
from bench.engine.__main__ import default_cli, main


def _draw_pattern(path, cols=12, rows=10, pitch=20, margin=30):
    W, H = margin * 2 + cols * pitch, margin * 2 + rows * pitch + 80
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    colors = [(236, 197, 172), (0, 0, 0), (171, 35, 47)]
    for r in range(rows):
        for c in range(cols):
            if 2 <= r < rows - 2 and 2 <= c < cols - 2:
                x, y = margin + c * pitch, margin + r * pitch
                d.rectangle([x, y, x + pitch, y + pitch], fill=colors[(r + c) % 3])
    for i in range(cols + 1):
        x = margin + i * pitch
        d.line([x, margin, x, margin + rows * pitch], fill=(200, 200, 200))
    for j in range(rows + 1):
        y = margin + j * pitch
        d.line([margin, y, margin + cols * pitch, y], fill=(200, 200, 200))
    for k, col in enumerate(colors):
        d.rectangle([margin + k * 50, margin + rows * pitch + 20, margin + k * 50 + 40, margin + rows * pitch + 60], fill=col)
    im.save(path)


@pytest.mark.skipif(default_cli() is None, reason="没有编译好的 beadcore-cli")
def test_run_writes_preds(tmp_path, monkeypatch):
    monkeypatch.setenv("PINDOU_DATA_DIR", str(tmp_path))
    (tmp_path / "images").mkdir()
    _draw_pattern(tmp_path / "images" / "syn_00000001_0.png")
    man = tmp_path / "manifest.jsonl"
    F.write_jsonl(man, [{"image_id": "syn_00000001_0", "file": "syn_00000001_0.png", "format": "png",
                         "source_path": "own_export", "type": "T1"}])
    pal = tmp_path / "pal.json"
    pal.write_text('{"palettes":[]}', encoding="utf-8")
    rc = main(["run", "--ids", "syn_00000001_0", "--run", "t1", "--manifest", str(man), "--palettes", str(pal),
               "--jobs", "1"])
    assert rc == 0
    pred = json.loads((tmp_path / "preds" / "t1" / "syn_00000001_0.json").read_text(encoding="utf-8"))
    assert "error" not in pred, pred
    assert pred["grid"] == {"cols": 12, "rows": 10}
    meta = json.loads((tmp_path / "preds" / "t1" / "_run.json").read_text(encoding="utf-8"))
    assert meta["units"] == 1 and not meta["failed"]


def test_test_split_requires_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("PINDOU_DATA_DIR", str(tmp_path))
    pal = tmp_path / "pal.json"
    pal.write_text('{"palettes":[]}', encoding="utf-8")
    cli = tmp_path / "fake-cli"
    cli.write_text("", encoding="utf-8")
    assert main(["run", "--split", "test", "--run", "x", "--cli", str(cli), "--palettes", str(pal)]) == 2
