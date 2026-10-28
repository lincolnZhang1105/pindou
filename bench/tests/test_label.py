"""bench.label 的测试：在线程里启动服务，用 http 客户端请求接口；导出的 JSON 必须通过 validate_label。

测试图纸用 numpy 现场画：格子底色 + 格内「文字」（中间一小块反色）+ 1 px 网格线 + 底部图例色块。
白豆 H2 和空格底色相同，只靠格内有没有文字区分。
"""
from __future__ import annotations

import http.client
import json

import numpy as np
import pytest
from PIL import Image

from bench.collect.manifest import add_images
from bench.common import dataset as ds
from bench.common import formats
from bench.label.server import serve_in_thread
from bench.label.store import legend_diff, label_to_draft

PALETTE = {"A1": (250, 240, 200), "H7": (0, 0, 0), "H2": (255, 255, 255), "G3": (236, 197, 172)}
CELLS = [
    ["H7", "H7", None, "A1", "A1", None],
    ["H7", "H2", "H2", "A1", "G3", None],
    [None, "H2", "G3", "G3", "G3", "H7"],
    [None, None, "A1", "H2", None, "H7"],
    ["G3", "G3", "A1", "A1", "H2", "H7"],
]
CELL, X0, Y0 = 20, 30, 25
LEGEND_Y = Y0 + len(CELLS) * CELL + 30


def legend_of(cells):
    from collections import Counter
    c = Counter(v for r in cells for v in r if v)
    return [{"code": k, "count": n} for k, n in sorted(c.items())]


def true_lines(n, start):
    return [start + i * CELL + 0.5 for i in range(n + 1)]


def swatch_xy(i):
    return X0 + i * 40 + 15, LEGEND_Y + 15


def draw_pattern(cells=CELLS) -> np.ndarray:
    rows, cols = len(cells), len(cells[0])
    w, h = max(X0 * 2 + cols * CELL + 1, X0 + len(PALETTE) * 40 + 10), LEGEND_Y + 60
    img = np.full((h, w, 3), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            code = cells[r][c]
            y, x = Y0 + r * CELL, X0 + c * CELL
            if code:
                img[y:y + CELL, x:x + CELL] = PALETTE[code]
                ink = (255, 255, 255) if code == "H7" else (0, 0, 0)
                img[y + 8:y + 12, x + 7:x + 13] = ink        # 格内「色号文字」
    for i in range(cols + 1):
        img[Y0:Y0 + rows * CELL + 1, X0 + i * CELL] = (180, 180, 200)
    for j in range(rows + 1):
        img[Y0 + j * CELL, X0:X0 + cols * CELL + 1] = (180, 180, 200)
    for i, code in enumerate(PALETTE):
        x, y = X0 + i * 40, LEGEND_Y
        img[y:y + 30, x:x + 30] = PALETTE[code]
        img[y:y + 30, x] = img[y:y + 30, x + 29] = img[y, x:x + 30] = img[y + 29, x:x + 30] = (120, 120, 120)
    return img


@pytest.fixture
def env(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "inbox").mkdir(parents=True)
    dsets = tmp_path / "datasets"
    dsets.mkdir()
    monkeypatch.setenv("PINDOU_DATA_DIR", str(data))
    monkeypatch.setattr(formats, "DATASETS", dsets)
    monkeypatch.setattr(formats, "MANIFEST", dsets / "manifest.jsonl")
    img = draw_pattern()
    Image.fromarray(img).save(data / "inbox" / "single.png")
    # 分页图：左 3 列一页、右 3 列一页
    left = [r[:3] for r in CELLS]
    right = [r[3:] for r in CELLS]
    Image.fromarray(draw_pattern(left)).save(data / "inbox" / "p0.png")
    Image.fromarray(draw_pattern(right)).save(data / "inbox" / "p1.png")
    add_images([(data / "inbox" / "single.png", {"platform": "xhs", "note_id": "aaaa0000", "source_path": "app_save",
                                                 "type": "T1", "grid": "6x5", "palette_hint": "MARD-221",
                                                 "tags": "white_vs_empty"}),
                (data / "inbox" / "p0.png", {"platform": "xhs", "note_id": "bbbb0000", "source_path": "app_save",
                                             "type": "T4"}),
                (data / "inbox" / "p1.png", {"platform": "xhs", "note_id": "bbbb0000", "source_path": "app_save",
                                             "type": "T4"})], data)
    srv, _ = serve_in_thread(data, port=0)
    yield data, srv
    srv.shutdown()
    srv.server_close()


class Client:
    def __init__(self, srv, host=None):
        self.port = srv.server_address[1]
        self.host = host

    def req(self, method, path, body=None, ctype="application/json", raw=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        headers = {}
        if self.host:
            headers["Host"] = self.host
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            headers["Content-Type"] = ctype
        conn.request(method, path, body=data, headers=headers)
        r = conn.getresponse()
        payload = r.read()
        conn.close()
        if raw:
            return r.status, payload, r.getheader("Content-Type")
        return r.status, json.loads(payload) if payload else None


SINGLE = "xhs_aaaa0000_0"
GROUP = "xhs_bbbb0000"


def page_of(cols=6, rows=5, **kw):
    p = {"image_index": 0, "cols": cols, "rows": rows, "col_offset": 0, "row_offset": 0,
         "x_lines": true_lines(cols, X0), "y_lines": true_lines(rows, Y0)}
    p.update(kw)
    return p


def full_draft(cells=CELLS, legend=None, **kw):
    d = {"type": "T1", "palette_hint": "MARD-221", "tags": ["white_vs_empty"], "grid": {"cols": 6, "rows": 5},
         "pages": [page_of()], "cells": cells, "legend": legend if legend is not None else legend_of(cells),
         "no_legend": False, "flags": []}
    d.update(kw)
    return d


# --------------------------------------------------------------------------- 静态文件和安全

def test_static_and_security(env):
    data, srv = env
    c = Client(srv)
    st, body, ctype = c.req("GET", "/", raw=True)
    assert st == 200 and ctype.startswith("text/html") and "真值标注".encode() in body
    st, body, ctype = c.req("GET", "/static/app.js", raw=True)
    assert st == 200 and "javascript" in ctype
    for bad in ("/static/../store.py", "/static/..%2Fstore.py", "/static/%2e%2e%5cstore.py", "/static/sub/x.js",
                "/img/..%2F..%2Fmanifest", "/api/task/..%2Fx", "/etc/passwd"):
        st, _ = c.req("GET", bad)
        assert st in (400, 404), bad
    st, body = Client(srv, host="evil.example:80").req("GET", "/api/tasks")
    assert st == 403
    st, body = Client(srv, host=f"localhost:{srv.server_address[1]}").req("GET", "/api/tasks")
    assert st == 200
    st, body = c.req("PUT", f"/api/draft/{SINGLE}", b"{}", ctype="text/plain")
    assert st == 415                                   # 跨站简单请求带不了 application/json
    st, body = c.req("PUT", "/api/draft/not_in_manifest", {"cells": []})
    assert st == 404
    assert srv.server_address[0] == "127.0.0.1"
    # manifest 的 file 字段被改成 ../ 也读不到数据目录之外的文件
    rows = formats.read_jsonl(formats.MANIFEST)
    (data.parent / "outside.png").write_bytes((data / "images" / rows[0]["file"]).read_bytes())
    rows[0]["file"] = "../../outside.png"
    formats.write_jsonl(formats.MANIFEST, rows)
    st, body = c.req("GET", f"/img/{rows[0]['image_id']}")
    assert st == 403


def test_task_list_and_image(env):
    data, srv = env
    c = Client(srv)
    st, body = c.req("GET", "/api/tasks")
    tasks = {t["key"]: t for t in body["tasks"]}
    assert set(tasks) == {SINGLE, GROUP}
    assert tasks[GROUP]["image_ids"] == ["xhs_bbbb0000_0", "xhs_bbbb0000_1"] and tasks[GROUP]["needs_review"]
    assert tasks[SINGLE]["status"]["code"] == "unlabeled" and not tasks[SINGLE]["needs_review"]
    st, t = c.req("GET", f"/api/task/{SINGLE}")
    assert t["meta"]["grid"] == {"cols": 6, "rows": 5} and t["draft"] is None and t["images"][0]["readable"]
    st, raw, ctype = c.req("GET", f"/img/{SINGLE}", raw=True)
    assert st == 200 and ctype == "image/png"
    assert raw == (data / "images" / f"{SINGLE}.png").read_bytes()   # 原图字节，不缩放不转码
    st, pal = c.req("GET", "/api/palettes")
    assert any(p["id"] == "MARD-221" for p in pal["palettes"])


# --------------------------------------------------------------------------- 取色、吸附、自动填充

def test_sample_ignores_text(env):
    data, srv = env
    c = Client(srv)
    # 点在 H2 格子正中（正好是「文字」上），取到的应该是白色底色
    x, y = X0 + 1 * CELL + 10, Y0 + 1 * CELL + 10
    st, body = c.req("GET", f"/api/sample/{SINGLE}?image_index=0&x={x}&y={y}")
    assert st == 200 and body["rgb"] == [255, 255, 255]
    x, y = swatch_xy(3)
    st, body = c.req("GET", f"/api/sample/{SINGLE}?image_index=0&x={x}&y={y}")
    assert body["rgb"] == list(PALETTE["G3"])
    st, body = c.req("GET", f"/api/sample/{SINGLE}?image_index=0&x=-50&y=-50")
    assert st == 400


def test_snap_fixes_drift(env):
    data, srv = env
    c = Client(srv)
    truth_x, truth_y = true_lines(6, X0), true_lines(5, Y0)
    # 均匀但整体偏 3 px，且越往右漂得越多
    xl = [v + 3 + i * 0.8 for i, v in enumerate(truth_x)]
    yl = [v - 3 for v in truth_y]
    st, body = c.req("POST", f"/api/snap/{SINGLE}", {"page": {"image_index": 0, "x_lines": xl, "y_lines": yl}})
    assert st == 200 and body["moved"] > 0
    assert max(abs(a - b) for a, b in zip(body["x_lines"], truth_x)) <= 1.0
    assert max(abs(a - b) for a, b in zip(body["y_lines"], truth_y)) <= 1.0


def test_autofill_white_vs_empty(env):
    data, srv = env
    c = Client(srv)
    swatches = [{"code": k, "rgb": list(v)} for k, v in PALETTE.items()] + [{"code": None, "rgb": [255, 255, 255]}]
    body = {"grid": {"cols": 6, "rows": 5}, "pages": [page_of()], "swatches": swatches}
    st, res = c.req("POST", f"/api/autofill/{SINGLE}", body)
    assert st == 200
    assert res["cells"] == CELLS          # 白豆（有文字）和空格（没文字）都分对了
    assert res["filled"] == 30
    # overwrite=False 只填空格（None）
    pre = [[("X1" if (r + c) % 2 else None) for c in range(6)] for r in range(5)]
    body.update(cells=pre, options={"overwrite": False})
    st, res = c.req("POST", f"/api/autofill/{SINGLE}", body)
    assert all(res["cells"][r][cc] == "X1" for r in range(5) for cc in range(6) if (r + cc) % 2)
    # 没给空格色块：空格会被填成 H2，并且没有「孪生」可以区分
    body = {"grid": {"cols": 6, "rows": 5}, "pages": [page_of()], "swatches": swatches[:-1]}
    st, res = c.req("POST", f"/api/autofill/{SINGLE}", body)
    assert res["cells"][0][2] == "H2"


# --------------------------------------------------------------------------- 草稿、导出、复核

def test_draft_export_and_legend_gate(env):
    data, srv = env
    c = Client(srv)
    good = full_draft()
    bad = json.loads(json.dumps(good))
    bad["cells"][0][2] = "H7"
    st, r = c.req("PUT", f"/api/draft/{SINGLE}", bad)
    assert st == 200 and r["rev"] == 1 and r["status"]["code"] == "legend_fail"
    assert ds.draft_path(SINGLE, data).is_file()
    st, t = c.req("GET", f"/api/task/{SINGLE}")
    assert t["draft_source"] == "draft" and t["draft"]["cells"][0][2] == "H7"
    # 图例校验不通过 → 禁止导出
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": bad, "user": "alice"})
    assert st == 422 and not r["ok"] and any("图例校验不通过" in e for e in r["errors"])
    assert not ds.label_path(SINGLE, data).exists()
    # 没填名字
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": good, "user": " "})
    assert st == 422
    # 图例颗数没填
    nocount = full_draft(legend=[{"code": "A1", "count": None}])
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": nocount, "user": "alice"})
    assert st == 422 and any("颗数" in e for e in r["errors"])
    # 网格线不递增
    badgrid = full_draft(pages=[page_of(x_lines=list(reversed(true_lines(6, X0))))])
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": badgrid, "user": "alice"})
    assert st == 422 and any("递增" in e for e in r["errors"])
    # 正常导出
    good["cells"] = [[(v.lower().replace("1", "01") if v else v) for v in row] for row in CELLS]  # a01 → A1
    good["flags"] = [[0, 0, "far"]]
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": good, "user": "alice"})
    assert st == 200 and r["ok"] and r["path"] == f"labels/{SINGLE}.json"
    assert any("可疑格" in w for w in r["warnings"])
    lab = formats.load_json(ds.label_path(SINGLE, data))
    assert formats.validate_label(lab) == []
    assert lab["cells"] == CELLS
    assert lab["image_id"] == SINGLE and lab["type"] == "T1" and lab["size_bucket"] == "S"
    assert lab["grid"] == {"cols": 6, "rows": 5} and lab["palette_hint"] == "MARD-221"
    rec = formats.load_manifest(formats.MANIFEST)[SINGLE]
    assert lab["images"] == [{"file": rec["file"], "sha256": rec["sha256"]}]
    assert lab["source"]["note_id"] == "aaaa0000" and lab["source"]["source_path"] == "app_save"
    p = lab["pages"][0]
    assert p["bbox"] == [p["x_lines"][0], p["y_lines"][0], p["x_lines"][-1], p["y_lines"][-1]]
    assert len(p["x_lines"]) == 7 and len(p["y_lines"]) == 6
    assert lab["annotator"] == "alice" and lab["needs_review"] is False and lab["reviewed_by"] is None
    assert list(lab)[:12] == ["image_id", "source", "type", "size_bucket", "palette_hint", "images", "grid", "pages",
                              "legend", "cells", "tags", "needs_review"]
    st, body = c.req("GET", "/api/tasks")
    status = {t["key"]: t["status"] for t in body["tasks"]}[SINGLE]
    assert status["code"] == "done" and not status["draft_newer"]
    # 导出后再改草稿 → 标记「有未导出的修改」
    st, r = c.req("PUT", f"/api/draft/{SINGLE}", good)
    assert r["status"]["code"] == "done" and r["status"]["draft_newer"]
    # 不需要复核的图，没有草稿时从真值继续编辑
    ds.draft_path(SINGLE, data).unlink()
    st, t = c.req("GET", f"/api/task/{SINGLE}")
    assert t["draft_source"] == "label" and t["draft"]["cells"] == CELLS


def test_no_legend_export(env):
    data, srv = env
    c = Client(srv)
    d = full_draft(legend=[], no_legend=False)
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": d, "user": "alice"})
    assert st == 422 and any("图例为空" in e for e in r["errors"])
    d["no_legend"] = True
    st, r = c.req("POST", f"/api/export/{SINGLE}", {"draft": d, "user": "alice"})
    assert st == 200 and r["ok"]
    lab = formats.load_json(ds.label_path(SINGLE, data))
    assert lab["legend"] == [] and formats.validate_label(lab) == []
    assert label_to_draft(lab)["no_legend"] is True


def test_paged_export_and_review(env):
    data, srv = env
    c = Client(srv)
    # 两页：每页 3×5，第二页列偏移 3
    pages = [page_of(3, 5, image_index=0), page_of(3, 5, image_index=1, col_offset=3)]
    swatches = [{"code": k, "rgb": list(v)} for k, v in PALETTE.items()] + [{"code": None, "rgb": [255, 255, 255]}]
    st, res = c.req("POST", f"/api/autofill/{GROUP}", {"grid": {"cols": 6, "rows": 5}, "pages": pages,
                                                       "swatches": swatches})
    assert st == 200 and res["cells"] == CELLS
    d = full_draft(type="T4", pages=pages)
    st, r = c.req("POST", f"/api/export/{GROUP}", {"draft": d, "user": "alice"})
    assert st == 200 and r["needs_review"] and r["reviewed_by"] is None
    assert r["status"]["code"] == "needs_review"
    lab = formats.load_json(ds.label_path(GROUP, data))
    assert formats.validate_label(lab) == []
    assert lab["image_id"] == GROUP and lab["size_bucket"] == "XL" and len(lab["images"]) == 2
    assert [p["col_offset"] for p in lab["pages"]] == [0, 3]
    # 复核：同一个人不行
    st, r = c.req("POST", f"/api/export/{GROUP}", {"draft": d, "user": "alice", "review": True})
    assert st == 422 and any("另一个人" in e for e in r["errors"])
    st, r = c.req("POST", f"/api/export/{GROUP}", {"draft": d, "user": "bob", "review": True})
    assert st == 200 and r["reviewed_by"] == "bob" and r["status"]["code"] == "done"
    lab = formats.load_json(ds.label_path(GROUP, data))
    assert lab["annotator"] == "alice" and lab["reviewed_by"] == "bob" and lab["reviewed_at"]
    # 标注人重新导出 → 复核作废
    st, r = c.req("POST", f"/api/export/{GROUP}", {"draft": d, "user": "alice"})
    assert formats.load_json(ds.label_path(GROUP, data))["reviewed_by"] is None
    # 评测器按 manifest 的 group 找真值：每页的 image_id → labels/<group>.json
    for iid in ("xhs_bbbb0000_0", "xhs_bbbb0000_1"):
        assert ds.label_path(ds.unit_key(formats.load_manifest(formats.MANIFEST)[iid]), data).is_file()


# --------------------------------------------------------------------------- 预测结果导入

def test_pred_import(env):
    data, srv = env
    c = Client(srv)
    book = ["A1", "G3", "H2", "H7"]
    flat = [0 if v is None else book.index(v) + 1 for row in CELLS for v in row]
    pred = {"engine_version": "0.0.1", "palette_guess": {"palette_id": "MARD-221", "confidence": 0.9},
            "grid": {"cols": 6, "rows": 5},
            "pages": [{"image_index": 0, "bbox": [0, 0, 0, 0], "x_lines": true_lines(6, X0), "y_lines": true_lines(5, Y0),
                       "col_offset": 0, "row_offset": 0, "cols": 6, "rows": 5, "source": "auto"}],
            "codebook": [{"palette_id": "MARD-221", "palette_version": "x", "code": k, "legend_rgb": [0, 0, 0]} for k in
                         ["A01", "G03", "H02", "H07"]],
            "cells": flat, "confidence": [1.0] * 30, "uncertain": [7, 29],
            "legend": [{"palette_id": "MARD-221", "code": "H07", "count": 6}]}
    run = ds.preds_dir(data) / "baseline_pindo"
    run.mkdir(parents=True)
    (run / f"{SINGLE}.json").write_text(json.dumps(pred), encoding="utf-8")
    err = ds.preds_dir(data) / "broken"
    err.mkdir()
    (err / f"{SINGLE}.json").write_text(json.dumps({"error": {"code": "no_grid_found", "message": "x"}}), encoding="utf-8")
    st, body = c.req("GET", f"/api/preds/{SINGLE}")
    assert {r["run"] for r in body["runs"]} == {"baseline_pindo", "broken"}
    st, d = c.req("GET", f"/api/pred/{SINGLE}/baseline_pindo")
    assert st == 200 and d["cells"] == CELLS and d["grid"] == {"cols": 6, "rows": 5}
    assert d["flags"] == [[1, 1, "uncertain"], [4, 5, "uncertain"]]
    assert d["legend"] == [{"code": "H7", "count": 6}] and d["palette_guess"] == "MARD-221"
    assert d["pages"][0]["x_lines"] == true_lines(6, X0)
    st, d = c.req("GET", f"/api/pred/{SINGLE}/broken")
    assert "no_grid_found" in d["error"]
    st, d = c.req("GET", f"/api/pred/{SINGLE}/nope")
    assert st == 404


def test_heic_without_conversion(env):
    data, srv = env
    from bench.tests.test_collect import synthetic_heic
    src = data / "inbox" / "IMG_1.heic"
    src.write_bytes(synthetic_heic(rot=0))
    add_images([(src, {"platform": "photo", "source_path": "photo", "type": "T6", "size_bucket": "M"})], data)
    iid = [r["image_id"] for r in formats.read_jsonl(formats.MANIFEST) if r["format"] == "heic"][0]
    c = Client(srv)
    st, t = c.req("GET", f"/api/task/{iid}")
    assert t["images"][0]["readable"] is False
    st, body = c.req("GET", f"/img/{iid}")
    assert st == 409 and "convert" in body["error"]


def test_legend_diff_helper():
    cells = [["A01", None], ["a1", "H7"]]
    assert legend_diff(cells, [{"code": "A1", "count": 2}, {"code": "H07", "count": 1}]) == []
    assert legend_diff(cells, [{"code": "A1", "count": 3}]) == [
        {"code": "A1", "expected": 3, "actual": 2}, {"code": "H7", "expected": 0, "actual": 1}]
