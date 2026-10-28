"""几何真值随变换同步：在真值给出的网格线 / 格子位置采样，和图片内容对得上。

- 无缩放：网格线位置的像素就是线色；格子内部的中位色就是该格的颜色（逐格精确）；
- 缩放到 1080 宽、截图边框（T5）、分页（T4 board / nine）、合集（T7）：按真值取格子内部中位色，
  在「图例色 + 空格色」里取最近色，必须和真值一致（≥ 99.5%，关掉格内文字以排除文字干扰）；
- 拍照（T6）：用 homography 把格子中心映射到照片里采样，做一次全局颜色 / 光照校正后分类，准确率 ≥ 90%。
"""
import numpy as np
import pytest
from PIL import Image

from bench.common.formats import load_json
from bench.synth.generate import GenOptions, Generator
from bench.synth.geom import apply_pts


def _load(out, image_id):
    lab = load_json(out / "labels" / f"{image_id}.json")
    ex = load_json(out / "extra" / f"{image_id}.extra.json")
    imgs = [np.asarray(Image.open(out / "images" / im["file"]).convert("RGB"), dtype=np.int32) for im in lab["images"]]
    return lab, ex, imgs


def _palette(ex):
    codes = [e["code"] for e in ex["codebook"]] + [None]
    cols = np.array([e["rgb"] for e in ex["codebook"]] + [ex["empty_rgb"]], dtype=np.float64)
    return codes, cols


def _classify_pages(lab, ex, imgs, scale_hint=1.0):
    """按真值的格子位置取内部中位色、最近色分类；返回 (正确数, 总数)。"""
    codes, cols = _palette(ex)
    st = ex["style"]
    tline = max(st["thin_w"], st["bold_w"] if st["bold_period"] else 0, st["border_w"], 1)
    ok = tot = 0
    for p in lab["pages"]:
        img = imgs[p["image_index"]]
        xs, ys = p["x_lines"], p["y_lines"]
        sx = (xs[-1] - xs[0]) / p["cols"] / st["pitch"]  # 渲染 → 本图的缩放
        m = tline * sx / 2 + 1.0
        for r in range(p["rows"]):
            for c in range(p["cols"]):
                x0, x1 = int(np.ceil(xs[c] + m)), int(np.floor(xs[c + 1] - m))
                y0, y1 = int(np.ceil(ys[r] + m)), int(np.floor(ys[r + 1] - m))
                if x1 <= x0 or y1 <= y0:
                    continue
                med = np.median(img[y0:y1, x0:x1].reshape(-1, 3), axis=0)
                k = int(np.argmin(((cols - med) ** 2).sum(1)))
                want = lab["cells"][p["row_offset"] + r][p["col_offset"] + c]
                if np.allclose(cols[k], cols[codes.index(want)]):  # 白豆和空格同色时两者都算对
                    ok += 1
                tot += 1
    return ok, tot


def _gen(tmp_path, n, **kw):
    kw.setdefault("allow_fallback_font", True)
    gen = Generator(GenOptions(**kw))
    for i in range(n):
        gen.generate(21, i, tmp_path)
    return sorted(p.stem for p in (tmp_path / "labels").glob("*.json"))


def test_lines_exact_without_resize(tmp_path):
    ids = _gen(tmp_path, 6, types=("T1", "T2"), max_side=40,
               degrade={"fmt": "png", "resize": 0, "watermark": False, "blur": 0.0})
    for iid in ids:
        lab, ex, imgs = _load(tmp_path, iid)
        st = ex["style"]
        line_cols = {tuple(st[k]) for k in ("thin_rgb", "bold_rgb", "border_rgb")}
        p = lab["pages"][0]
        img = imgs[0]
        xs, ys = p["x_lines"], p["y_lines"]
        if st["thin_w"] or st["border_w"]:
            for i, x in enumerate(xs):
                if not (st["thin_w"] or i in (0, len(xs) - 1)):
                    continue
                for r in range(0, p["rows"], max(1, p["rows"] // 7)):
                    y = int((ys[r] + ys[r + 1]) / 2)
                    assert tuple(img[y, int(np.floor(x))]) in line_cols, (iid, i, r)
        ok, tot = _classify_pages(lab, ex, imgs)
        assert tot > 0 and ok == tot, (iid, ok, tot)


@pytest.mark.parametrize("kind", ["resize", "screenshot", "paged_board", "paged_nine", "collage"])
def test_cells_align_after_transform(tmp_path, kind):
    deg = {"fmt": "png", "watermark": False, "blur": 0.0}
    # 这里只验证几何：关掉格内文字（缩小后 9 px 的格子里文字会占满内部，中位色不再是底色）。
    # 带文字时的逐格精确对齐由 test_lines_exact_without_resize 覆盖。
    kw = {"max_side": 44, "style_overrides": {"cell_text": False}}
    if kind == "resize":
        kw.update(types=("T1", "T2"), buckets=("M", "L"), max_side=None)
        deg["resize"] = 1080
    elif kind == "screenshot":
        kw["types"] = ("T5",)
        deg["pre_jpeg"] = 0
    elif kind == "paged_board":
        kw["types"] = ("T4",)
        deg.update(t4_mode="board", resize=1080)
    elif kind == "paged_nine":
        kw["types"] = ("T4",)
        deg.update(t4_mode="nine", resize=0)
    else:
        kw["types"] = ("T7",)
    ids = _gen(tmp_path, 4, degrade=deg, **kw)
    for iid in ids:
        lab, ex, imgs = _load(tmp_path, iid)
        if kind == "resize":
            assert any(op["op"] == "resize" for im in ex["images"] for op in im["degradations"]) or \
                ex["images"][0]["width"] <= 1080
        ok, tot = _classify_pages(lab, ex, imgs)
        assert tot > 0 and ok / tot >= 0.995, (iid, kind, ok, tot)


def test_lines_after_resize_are_lines(tmp_path):
    """缩放后：真值竖线位置的像素比相邻格子中心更接近线色（T2 无文字、方格）。"""
    ids = _gen(tmp_path, 4, types=("T2",), buckets=("M",),
               style_overrides={"cell_shape": "square", "thin_w": 2, "thin_rgb": (40, 40, 40), "bold_period": 0,
                                "border_w": 2, "border_rgb": (40, 40, 40), "pitch": 30},
               degrade={"fmt": "png", "resize": 1080, "watermark": False, "blur": 0.0})
    for iid in ids:
        lab, ex, imgs = _load(tmp_path, iid)
        p, img = lab["pages"][0], imgs[0]
        assert img.shape[1] == 1080
        xs, ys = p["x_lines"], p["y_lines"]
        line = np.array([40, 40, 40])
        better = total = 0
        for i in range(1, len(xs) - 1):
            for r in range(0, p["rows"], 3):
                y = int((ys[r] + ys[r + 1]) / 2)
                on = img[y, int(np.floor(xs[i]))]
                mid = img[y, int((xs[i] + xs[i + 1]) / 2)]
                better += np.abs(on - line).sum() < np.abs(mid - line).sum()
                total += 1
        assert better / total >= 0.97, (iid, better, total)
        # 整体偏移半格后就不成立了（说明检查有区分度）
        sh = (xs[1] - xs[0]) / 2
        worse = sum(np.abs(img[int((ys[r] + ys[r + 1]) / 2), int(xs[i] + sh)] - line).sum()
                    < np.abs(img[int((ys[r] + ys[r + 1]) / 2), int(np.floor(xs[i]))] - line).sum()
                    for i in range(1, len(xs) - 2) for r in range(0, p["rows"], 3))
        assert worse / total < 0.1


def test_photo_homography(tmp_path):
    ids = _gen(tmp_path, 3, types=("T6",), styles=("blocks",), buckets=("S",),
               style_overrides={"cell_shape": "square", "legend": "chips_in", "pitch": 40},
               degrade={"watermark": False})
    for iid in ids:
        lab, ex, imgs = _load(tmp_path, iid)
        p, img = lab["pages"][0], imgs[0]
        H = np.array(p["homography"])
        fx, fy = p["flat_x_lines"], p["flat_y_lines"]
        pts, want = [], []
        for r in range(p["rows"]):
            for c in range(p["cols"]):
                pts.append([(fx[c] + fx[c + 1]) / 2, (fy[r] + fy[r + 1]) / 2])
                want.append(lab["cells"][r][c])
        q = apply_pts(H, pts)
        h, w = img.shape[:2]
        assert (q[:, 0] > 0).all() and (q[:, 0] < w).all() and (q[:, 1] > 0).all() and (q[:, 1] < h).all()
        samp = np.array([np.median(img[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4].reshape(-1, 3), axis=0)
                         for x, y in q], dtype=np.float64)
        codes, cols = _palette(ex)
        truth = np.array([cols[codes.index(v)] for v in want])
        # 拍照的光照不均是乘性的：在 log 空间里用「颜色 + 位置的二次项」拟合一次全局校正，再做最近色分类
        u, v = q[:, 0] / w - 0.5, q[:, 1] / h - 0.5
        A = np.column_stack([np.log1p(samp), u, v, u * u, v * v, u * v, np.ones(len(samp))])
        coef, *_ = np.linalg.lstsq(A, np.log1p(truth), rcond=None)
        corr = np.expm1(A @ coef)
        pred = [cols[int(np.argmin(((cols - v) ** 2).sum(1)))] for v in corr]
        acc = np.mean([np.allclose(a, b) for a, b in zip(pred, truth)])  # 白豆和空格同色时两者都算对
        assert acc >= 0.9, (iid, acc)
        # 近似的 x_lines（线中点的投影）落在 bbox 里
        x0, y0, x1, y1 = p["bbox"]
        assert x0 - 1 <= min(p["x_lines"]) and max(p["x_lines"]) <= x1 + 1


def test_legend_boxes_hit_swatches(tmp_path):
    ids = _gen(tmp_path, 6, types=("T1",), max_side=30,
               degrade={"fmt": "png", "resize": 0, "watermark": False, "blur": 0.0})
    for iid in ids:
        lab, ex, imgs = _load(tmp_path, iid)
        rgb = {e["code"]: e["rgb"] for e in ex["codebook"]}
        for im_ex, img in zip(ex["images"], imgs):
            for it in im_ex["legend"]:
                x0, y0, x1, y1 = it["swatch_box"]
                # 色块水平居中、靠上的位置（避开中间的色号文字和圆角）
                x, y = int((x0 + x1) / 2), int(y0 + (y1 - y0) * 0.15)
                assert np.abs(img[y, x] - np.array(rgb[it["code"]])).max() <= 2, (iid, it["code"])
