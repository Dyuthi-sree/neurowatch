"""Bipolar montage resolution.

Most CHB-MIT files store the 18 longitudinal bipolar channels directly. Three chb12
files (chb12_27, chb12_28, chb12_29) were recorded referentially instead:

  * chb12_27: every electrode against a common reference, labelled e.g. "F7-CS2"
  * chb12_28, chb12_29: bare electrode labels ("F7", "T7", ...), with the left occipital
    electrode mislabelled "01" (zero-one) instead of "O1" in the EDF header and summary.

Project decision (see README): these files are recovered by rebuilding the bipolar
channels as electrode differences, A-B = (A - ref) - (B - ref); the common reference
cancels exactly, so this is the same quantity a bipolar recording measures.
"""
from __future__ import annotations

from .channels import norm_label
from .config import CHANNELS_18

# label typos found in the referential chb12 files
_ELECTRODE_ALIASES = {"01": "O1"}
_REF_SUFFIXES = ("-CS2", "-REF")


def electrode_name(label: str) -> str | None:
    """Return the electrode a referential label measures, or None if it is not one."""
    lab = norm_label(label)
    for suf in _REF_SUFFIXES:
        if lab.endswith(suf):
            lab = lab[: -len(suf)]
            break
    lab = _ELECTRODE_ALIASES.get(lab, lab)
    if not lab or "-" in lab or lab in {".", "EKG1", "ECG", "VNS"}:
        return None
    return lab


def resolve(labels: list[str], wanted: list[str] = CHANNELS_18) -> dict:
    """Decide how to obtain each wanted bipolar channel from a file's signal labels.

    Returns {"mode": "bipolar"|"rebuilt", "plan": [(i_plus, i_minus_or_None), ...]}
    where i_* are signal indices in the file. In "bipolar" mode the second index is
    None (channel read directly, first occurrence of duplicate labels). Raises KeyError
    if a channel can be obtained neither way.
    """
    norm = [norm_label(l) for l in labels]
    if all(w in norm for w in wanted):
        return {"mode": "bipolar", "plan": [(norm.index(w), None) for w in wanted]}
    elec: dict[str, int] = {}
    for i, l in enumerate(labels):
        e = electrode_name(l)
        if e is not None and e not in elec:
            elec[e] = i
    plan, missing = [], []
    for w in wanted:
        a, b = w.split("-")
        if a in elec and b in elec:
            plan.append((elec[a], elec[b]))
        else:
            missing.append(w)
    if missing:
        raise KeyError(f"cannot obtain channels {missing} (bipolar or referential)")
    return {"mode": "rebuilt", "plan": plan}


def apply_plan(data, plan):
    """data: (n_signals, n_samples) array -> (len(plan), n_samples) bipolar array."""
    import numpy as np

    out = np.empty((len(plan), data.shape[1]), dtype=data.dtype)
    for k, (i, j) in enumerate(plan):
        out[k] = data[i] if j is None else data[i] - data[j]
    return out
