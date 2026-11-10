"""Fixed-length windowing and window labels."""
from __future__ import annotations

import numpy as np

from .config import FS, STEP_S, WIN_S

POS_OVERLAP = 0.5  # window is positive if >= 50% of it lies inside annotated seizure(s)


def n_windows(n_samples: int, fs: float = FS, win_s: float = WIN_S, step_s: float = STEP_S) -> int:
    w, s = int(round(win_s * fs)), int(round(step_s * fs))
    return 0 if n_samples < w else (n_samples - w) // s + 1


def window_starts_s(n_samples: int, fs: float = FS, win_s: float = WIN_S, step_s: float = STEP_S):
    """Start times (s) of all complete windows; windows never run past the file end."""
    return np.arange(n_windows(n_samples, fs, win_s, step_s)) * step_s


def label_windows(starts_s: np.ndarray, seizures: list[tuple[float, float]], win_s: float = WIN_S):
    """Return (overlap_fraction, label, seizure_idx) per window.

    overlap_fraction = total seconds of the window inside any annotated seizure / win_s.
    seizure_idx = 1-based index (within the file) of the seizure with the largest overlap,
    0 if the window overlaps no seizure.
    """
    starts_s = np.asarray(starts_s, float)
    ends = starts_s + win_s
    ov = np.zeros((len(starts_s), max(len(seizures), 1)))
    for k, (s, e) in enumerate(seizures):
        ov[:, k] = np.clip(np.minimum(ends, e) - np.maximum(starts_s, s), 0, None)
    frac = ov.sum(1) / win_s
    sz_idx = np.where(ov.max(1) > 0, ov.argmax(1) + 1, 0)
    return frac, (frac >= POS_OVERLAP - 1e-9).astype(np.int8), sz_idx.astype(np.int16)


def frame(x: np.ndarray, fs: float = FS, win_s: float = WIN_S, step_s: float = STEP_S):
    """(n_ch, n_samples) -> read-only view (n_win, n_ch, win)."""
    w, s = int(round(win_s * fs)), int(round(step_s * fs))
    v = np.lib.stride_tricks.sliding_window_view(x, w, axis=-1)[:, ::s]
    return v.transpose(1, 0, 2)[: n_windows(x.shape[-1], fs, win_s, step_s)]
