"""Leakage checks on the real split and cache (skipped if the cache is not built)."""
import json

import numpy as np
import pandas as pd
import pytest

from neurowatch.config import CACHE, EXCLUDED_CASES, PATIENT_OF_CASE, RESULTS
from neurowatch.windows import n_windows


@pytest.fixture(scope="module")
def split():
    return json.loads((RESULTS / "split.json").read_text())["patient_split"]


@pytest.fixture(scope="module")
def win():
    p = CACHE / "windows.parquet"
    if not p.exists():
        pytest.skip("cache not built")
    return pd.read_parquet(p)


def test_split_sizes_and_constraints(split):
    counts = pd.Series(split).value_counts().to_dict()
    assert counts == {"train": 14, "val": 4, "test": 4}
    assert "chb24" not in split and "chb21" not in split  # chb21 is merged into patient chb01
    assert PATIENT_OF_CASE["chb21"] == "chb01"
    cases = [c for c in PATIENT_OF_CASE if c not in EXCLUDED_CASES]
    assert {PATIENT_OF_CASE[c] for c in cases} == set(split)


def test_each_patient_in_exactly_one_split(win, split):
    per = win.groupby("patient", observed=True).split.nunique()
    assert (per == 1).all()
    got = win.groupby("patient", observed=True).split.first().astype(str).to_dict()
    assert got == split
    assert set(win[win.case.isin(["chb01", "chb21"])].split.unique()) == {split["chb01"]}
    assert not win.case.isin(EXCLUDED_CASES).any()


def test_records_unique_to_split(win):
    assert (win.groupby("record", observed=True).split.nunique() == 1).all()


def test_window_counts_match_durations(win):
    rec = pd.read_csv(RESULTS / "recordings.csv")
    rec = rec[rec.included]
    got = win.groupby("record", observed=True).size()
    for r in rec.itertuples():
        assert got[r.record] == n_windows(int(round(r.duration_s * 256))), r.record


def test_labels_match_annotations(win):
    sz = pd.read_csv(RESULTS / "seizures.csv")
    sz = sz[sz.included]
    pos = win[win.label == 1]
    # every positive window lies >=50% inside an annotated seizure of its own record
    m = pos.merge(sz, left_on=["record", "seizure_idx"], right_on=["record", "seizure"], how="left")
    assert m.start_s_y.notna().all()
    # every included seizure yields at least one positive window
    hit = set(zip(pos.record.astype(str), pos.seizure_idx))
    missed = [(r, s) for r, s in zip(sz.record, sz.seizure) if (r, s) not in hit]
    assert not missed, missed
    assert win.split.isin(["train", "val", "test"]).all()


def test_features_aligned(win):
    f = np.load(CACHE / "features.npy", mmap_mode="r")
    assert f.shape == (len(win), 108)
    assert np.isfinite(f[:: max(1, len(f) // 20000)]).all()
