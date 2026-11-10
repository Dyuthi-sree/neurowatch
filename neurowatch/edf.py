"""Load the 18 reference bipolar channels from a CHB-MIT EDF file with MNE."""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from .channels import read_edf_header
from .config import CHANNELS_18, FS
from .montage import apply_plan, resolve


def load_channels(path: Path, channels: list[str] = CHANNELS_18, return_mode: bool = False):
    """Return array (n_channels, n_samples) in microvolts, in the order of `channels`.

    Channel positions come from the EDF header (first occurrence of each label), so
    MNE's renaming of duplicates ("T8-P8-0", "T8-P8-1") does not matter. Files recorded
    referentially (three chb12 files) are converted to bipolar by electrode differences.
    """
    hdr = read_edf_header(path)
    res = resolve(hdr["labels"], channels)
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if abs(raw.info["sfreq"] - FS) > 1e-6:
        raise ValueError(f"{Path(path).name}: fs={raw.info['sfreq']}")
    needed = sorted({i for p in res["plan"] for i in p if i is not None})
    sub = raw.get_data(picks=needed) * 1e6
    pos = {i: k for k, i in enumerate(needed)}
    plan = [(pos[i], None if j is None else pos[j]) for i, j in res["plan"]]
    x = apply_plan(sub, plan)
    return (x, res["mode"]) if return_mode else x
