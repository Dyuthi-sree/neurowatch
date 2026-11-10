"""Causal band-pass filtering (no future samples) via MNE's IIR design."""
from __future__ import annotations

import mne
import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

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


def causal_bandpass(x: np.ndarray, fs: float = FS, init: str = "first") -> np.ndarray:
    """Filter along the last axis, forward only (single pass => causal).

    Uses MNE's Butterworth design and applies it with scipy's sosfilt, which is the
    same operation as mne.filter.filter_data(..., method='iir', phase='forward').

    init="first": start the filter in the steady state for a constant input equal to the
    first sample (sosfilt_zi * x[0]); this depends only on x[0], so it stays causal, and
    it suppresses the start-of-file transient caused by any DC offset. init="zero": zero state.
    """
    x = np.asarray(x, dtype=np.float64)
    sos = causal_sos(fs)
    if init == "zero":
        return sosfilt(sos, x, axis=-1)
    zi0 = sosfilt_zi(sos)  # (n_sections, 2)
    # scipy wants zi of shape (n_sections, *x.shape[:-1], 2)
    zi = zi0.reshape((zi0.shape[0],) + (1,) * (x.ndim - 1) + (2,)) * x[..., :1][None]
    y, _ = sosfilt(sos, x, axis=-1, zi=zi)
    return y
