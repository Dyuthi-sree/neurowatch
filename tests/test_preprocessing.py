"""Unit tests for montage rebuilding, causal filtering, windowing, labels and features."""
import numpy as np
import pytest

from neurowatch.config import CHANNELS_18, FS
from neurowatch.features import columns_for_channels, feature_columns, window_features
from neurowatch.filtering import causal_bandpass
from neurowatch.montage import apply_plan, electrode_name, resolve
from neurowatch.summary_parser import parse_summary_text
from neurowatch.windows import frame, label_windows, n_windows, window_starts_s

ELECTRODES = sorted({e for c in CHANNELS_18 for e in c.split("-")})


# ---------------- montage ----------------
def test_native_bipolar_uses_first_duplicate():
    labels = CHANNELS_18[:15] + ["-"] + CHANNELS_18[15:] + ["T8-P8"]
    r = resolve(labels)
    assert r["mode"] == "bipolar"
    assert r["plan"][CHANNELS_18.index("T8-P8")] == (labels.index("T8-P8"), None)


@pytest.mark.parametrize("style", ["cs2", "bare_with_01_typo"])
def test_referential_rebuild_equals_electrode_differences(style):
    rng = np.random.default_rng(0)
    pot = {e: rng.normal(size=500) for e in ELECTRODES}
    ref = rng.normal(size=500) * 5  # common reference cancels out
    labels, rows = [], []
    for e in ELECTRODES + ["C2"]:
        lab = e
        if style == "cs2":
            lab = f"{e}-CS2"
        elif e == "O1":
            lab = "01"  # typo present in chb12_28/29
        labels.append(lab)
        rows.append(pot.get(e, np.zeros(500)) - ref)
    labels.insert(3, "-")
    rows.insert(3, np.zeros(500))
    r = resolve(labels)
    assert r["mode"] == "rebuilt"
    out = apply_plan(np.array(rows), r["plan"])
    for k, ch in enumerate(CHANNELS_18):
        a, b = ch.split("-")
        np.testing.assert_allclose(out[k], pot[a] - pot[b], atol=1e-12)


def test_missing_electrode_raises():
    labels = [e for e in ELECTRODES if e != "PZ"]
    with pytest.raises(KeyError):
        resolve(labels)


def test_electrode_name():
    assert electrode_name("01") == "O1"
    assert electrode_name("fp1-cs2") == "FP1"
    assert electrode_name("FP1-F7") is None
    assert electrode_name("-") is None


# ---------------- filtering: causality ----------------
@pytest.mark.parametrize("init", ["first", "zero"])
def test_filter_is_causal(init):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 20 * FS))
    cut = 11 * FS + 17
    x2 = x.copy()
    x2[:, cut:] += rng.normal(size=(3, x.shape[1] - cut)) * 100
    y, y2 = causal_bandpass(x, init=init), causal_bandpass(x2, init=init)
    np.testing.assert_array_equal(y[:, :cut], y2[:, :cut])
    assert not np.allclose(y[:, cut:], y2[:, cut:])


def test_filter_passband_and_stopband():
    t = np.arange(60 * FS) / FS
    for f, lo, hi in [(10.0, 0.9, 1.1), (60.0, 0.0, 0.2), (100.0, 0.0, 0.02), (0.05, 0.0, 0.01)]:
        y = causal_bandpass(np.sin(2 * np.pi * f * t))[20 * FS:]
        gain = y.std() / np.sin(2 * np.pi * f * t[20 * FS:]).std()
        assert lo <= gain <= hi, (f, gain)


def test_filter_zi_removes_dc_transient():
    x = np.full((2, 10 * FS), 500.0)  # pure DC offset
    assert np.abs(causal_bandpass(x, init="first")).max() < 1e-6
    assert np.abs(causal_bandpass(x, init="zero")).max() > 10


# ---------------- windowing and labels ----------------
def test_window_counts_and_bounds():
    for n_sec in [3.99, 4, 5.99, 6, 3600, 3624]:
        n = int(round(n_sec * FS))
        k = n_windows(n)
        assert k == (0 if n < 4 * FS else (n - 4 * FS) // (2 * FS) + 1)
        st = window_starts_s(n)
        if k:
            assert st[-1] * FS + 4 * FS <= n  # never runs past the file end
            assert np.all(np.diff(st) == 2.0)


def test_frame_matches_slicing():
    x = np.arange(2 * 20 * FS, dtype=float).reshape(2, -1)
    v = frame(x)
    assert v.shape == (n_windows(x.shape[1]), 2, 4 * FS)
    for i in [0, 3, len(v) - 1]:
        np.testing.assert_array_equal(v[i], x[:, i * 2 * FS: i * 2 * FS + 4 * FS])


def test_label_50pct_boundary():
    starts = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    # seizure 6.0-12.0: windows [2,6] overlap 0 s; [4,8] 2 s (=50%); [6,10] 4 s; [8,12] 4 s; [10,14] 2 s
    frac, lab, idx = label_windows(starts, [(6.0, 12.0)])
    np.testing.assert_allclose(frac, [0, 0, 0.5, 1, 1, 0.5])
    assert lab.tolist() == [0, 0, 1, 1, 1, 1]
    frac, lab, _ = label_windows(starts, [(6.01, 12.0)])
    assert lab[2] == 0  # 1.99 s overlap < 50%
    assert idx.tolist() == [0, 0, 1, 1, 1, 1]


def test_label_two_seizures_overlap_summed():
    frac, lab, idx = label_windows(np.array([0.0]), [(0.0, 1.0), (2.5, 10.0)])
    assert frac[0] == pytest.approx(0.625) and lab[0] == 1 and idx[0] == 2  # largest overlap


# ---------------- features ----------------
def test_features_shape_and_relative_power():
    t = np.arange(4 * FS) / FS
    w = np.stack([np.sin(2 * np.pi * f * t) for f in [2.0, 6.0, 10.0, 20.0]])[None]
    f = window_features(w).reshape(4, 6)
    for ch in range(4):  # each pure tone falls in its own band
        assert f[ch, 2 + ch] > 0.95
    assert len(feature_columns()) == 108
    cols = columns_for_channels(["FP1-F7", "CZ-PZ"])
    assert [feature_columns()[c] for c in cols][0] == "FP1-F7|std"
    assert [feature_columns()[c] for c in cols][-1] == "CZ-PZ|rel_beta"


def test_pipeline_features_causal():
    """Perturbing the signal after time T leaves every window ending at or before T unchanged."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(18, 60 * FS)) * 20
    T = 31 * FS
    x2 = x.copy()
    x2[:, T:] *= 7
    f1 = window_features(frame(causal_bandpass(x)))
    f2 = window_features(frame(causal_bandpass(x2)))
    ends = window_starts_s(x.shape[1]) + 4
    done = ends * FS <= T
    np.testing.assert_array_equal(f1[done], f2[done])
    assert not np.allclose(f1[~done], f2[~done])


def test_summary_parser():
    txt = """File Name: chb99_01.edf
File Start Time: 10:00:00
File End Time: 11:00:00
Number of Seizures in File: 2
Seizure 1 Start Time: 100 seconds
Seizure 1 End Time: 150 seconds
Seizure 2 Start Time:  300 seconds
Seizure 2 End Time: 320 seconds

File Name: chb99_02.edf
Number of Seizures in File: 0
"""
    e = parse_summary_text(txt)
    assert [x.file for x in e] == ["chb99_01.edf", "chb99_02.edf"]
    assert e[0].seizures == [(100.0, 150.0), (300.0, 320.0)] and e[1].seizures == []
