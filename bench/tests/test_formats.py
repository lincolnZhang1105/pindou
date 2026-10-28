import numpy as np

from bench.common.formats import code_key, normalize_code, pred_cells, size_bucket_of, validate_label


def _label():
    return {
        "image_id": "x", "type": "T1", "images": [{"file": "x.png", "sha256": ""}],
        "grid": {"cols": 3, "rows": 2},
        "pages": [{"image_index": 0, "bbox": [0, 0, 30, 20], "x_lines": [0, 10, 20, 30], "y_lines": [0, 10, 20],
                   "col_offset": 0, "row_offset": 0, "cols": 3, "rows": 2}],
        "legend": [{"code": "A1", "count": 2}, {"code": "H07", "count": 1}],
        "cells": [["A01", None, "H7"], [None, "A1", None]],
    }


def test_normalize():
    assert normalize_code(" a01 ") == "A1"
    assert code_key("ZG:zg01") == "ZG:ZG1"
    assert code_key(None) is None


def test_validate_ok_and_checksum():
    lab = _label()
    assert validate_label(lab) == []
    lab["legend"][0]["count"] = 3
    errs = validate_label(lab)
    assert any("图例校验不通过" in e for e in errs)


def test_pred_cells():
    pred = {"grid": {"cols": 2, "rows": 1}, "codebook": [{"palette_id": "M", "code": "A01"}], "cells": [1, 0]}
    arr = pred_cells(pred)
    assert arr.shape == (1, 2) and arr[0, 0] == "A1" and arr[0, 1] is None


def test_size_bucket():
    assert size_bucket_of(16, 16) == "S"
    assert size_bucket_of(58, 53) == "M"
    assert size_bucket_of(104, 104) == "L"
    assert size_bucket_of(208, 208) == "XL"
    assert size_bucket_of(52, 52, paged=True) == "XL"
