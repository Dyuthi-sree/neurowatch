"""Causal band-pass filtering (no future samples) via MNE's IIR design."""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import sosfilt

from .config import BAND, FS

_SOS_CACHE: dict = {}


def causal_sos(fs: float = FS, l_freq: float = BAND[0], h_freq: float = BAND[1], order: int = 4):
    key = (fs, l_freq, h_freq, order)
    if key not in _SOS_CACHE:
        params = mne.filter.construct_iir_filter(
            dict(order=order, ftype="butter", output="sos"),
            f_pass=[l_freq, h_freq], sfreq=fs, btype="bandpass", return_copy=False, verbose=False,
        )
        _SOS_CACHE[key] = params["sos"]
    return _SOS_CACHE[key]


def causal_bandpass(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """Filter along the last axis, forward only (single pass => causal).

    Uses MNE's Butterworth design and applies it with scipy's sosfilt, which is the
    same operation as mne.filter.filter_data(..., method='iir', phase='forward').
    """
    return sosfilt(causal_sos(fs), np.asarray(x, dtype=np.float64), axis=-1)
