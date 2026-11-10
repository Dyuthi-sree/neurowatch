"""Per-window, per-channel handcrafted features for the Random Forest.

Each feature uses only the samples inside its own 4 s window (of the causally filtered
signal), so no information from later in the recording can enter a window's features.
Feature vector layout: channel-major, i.e. [ch0_std, ch0_line_length, ch0_rel_delta, ...].
"""
from __future__ import annotations

import numpy as np

from .config import BAND, BANDS, CHANNELS_18, FEATURE_NAMES, FS

_EPS = 1e-12


def window_features(w: np.ndarray, fs: float = FS) -> np.ndarray:
    """w: (n_win, n_ch, n_samp) -> (n_win, n_ch * 6) float32."""
    w = np.asarray(w, dtype=np.float64)
    n_win, n_ch, n = w.shape
    std = w.std(axis=-1)
    ll = np.abs(np.diff(w, axis=-1)).mean(axis=-1)
    spec = np.abs(np.fft.rfft((w - w.mean(-1, keepdims=True)) * np.hanning(n), axis=-1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    total = spec[..., (f >= BAND[0]) & (f < BAND[1])].sum(-1) + _EPS
    rel = [spec[..., (f >= lo) & (f < hi)].sum(-1) / total for lo, hi in BANDS.values()]
    feats = np.stack([std, ll, *rel], axis=-1)  # (n_win, n_ch, 6)
    return feats.reshape(n_win, n_ch * len(FEATURE_NAMES)).astype(np.float32)


def feature_columns(channels: list[str] = CHANNELS_18) -> list[str]:
    return [f"{c}|{n}" for c in channels for n in FEATURE_NAMES]


def columns_for_channels(subset: list[str], channels: list[str] = CHANNELS_18) -> list[int]:
    """Indices of the feature columns that belong to a channel subset."""
    k = len(FEATURE_NAMES)
    return [channels.index(c) * k + j for c in subset for j in range(k)]
