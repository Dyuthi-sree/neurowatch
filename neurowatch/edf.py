"""Load the reference bipolar channels from a CHB-MIT EDF file with MNE."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from .channels import channel_index, read_edf_header
from .config import CHANNELS_18, FS


def load_channels(path: Path, channels: list[str] = CHANNELS_18) -> np.ndarray:
    """Return array (n_channels, n_samples) in microvolts, in the order of `channels`.

    Channel positions come from the EDF header (first occurrence of each label), so
    MNE's renaming of duplicates ("T8-P8-0", "T8-P8-1") does not matter.
    """
    hdr = read_edf_header(path)
    idx = channel_index(hdr["labels"], channels)
    missing = [c for c, i in idx.items() if i is None]
    if missing:
        raise KeyError(f"{path.name}: missing channels {missing}")
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raise ValueError(f"{path.name}: fs={raw.info['sfreq']}")
    picks = [idx[c] for c in channels]
    return raw.get_data(picks=picks) * 1e6
