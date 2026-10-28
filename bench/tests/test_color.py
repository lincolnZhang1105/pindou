import csv
from pathlib import Path

import numpy as np

from bench.common.color import delta_e_2000, lab_to_srgb8, srgb8_to_lab

DATA = Path(__file__).resolve().parents[2] / "core" / "tests" / "data"


def test_sharma_34_pairs():
    rows = [list(map(float, l.split())) for l in (DATA / "ciede2000_sharma.txt").read_text().splitlines() if l.strip()]
    arr = np.array(rows)
    assert arr.shape == (34, 7)
    de = delta_e_2000(arr[:, 0:3], arr[:, 3:6])
    assert np.max(np.abs(de - arr[:, 6])) < 1e-4


def test_matches_colour_science_golden():
    with (DATA / "colour_golden.csv").open() as f:
        r = csv.reader(f)
        next(r)
        a = np.array([list(map(float, row)) for row in r])
    lab1 = srgb8_to_lab(a[:, 0:3])
    lab2 = srgb8_to_lab(a[:, 3:6])
    assert np.max(np.abs(lab1 - a[:, 6:9])) < 1e-6
    assert np.max(np.abs(lab2 - a[:, 9:12])) < 1e-6
    assert np.max(np.abs(delta_e_2000(lab1, lab2) - a[:, 12])) < 1e-6


def test_lab_roundtrip():
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, size=(5000, 3))
    assert np.array_equal(lab_to_srgb8(srgb8_to_lab(rgb)), rgb)
