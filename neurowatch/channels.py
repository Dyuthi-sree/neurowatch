"""EDF header parsing and channel-name normalisation."""
from __future__ import annotations

from pathlib import Path

from .config import CHANNELS_18


def norm_label(label: str) -> str:
    return label.strip().upper().replace(" ", "")


def parse_edf_header(data: bytes) -> dict:
    """Parse the fixed + per-signal EDF header. Returns labels, fs, duration."""
    ns = int(data[252:256].decode("ascii").strip())
    n_rec = int(data[236:244].decode("ascii").strip())
    rec_dur = float(data[244:252].decode("ascii").strip())
    off = 256
    labels = [data[off + 16 * i: off + 16 * (i + 1)].decode("ascii", "replace") for i in range(ns)]
    off_spr = 256 + ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
    spr = [int(data[off_spr + 8 * i: off_spr + 8 * (i + 1)].decode("ascii").strip()) for i in range(ns)]
    return {
        "n_signals": ns,
        "labels": [norm_label(l) for l in labels],
        "fs": [s / rec_dur for s in spr],
        "duration_s": n_rec * rec_dur,
    }


def read_edf_header(path: Path) -> dict:
    with open(path, "rb") as f:
        head = f.read(256)
        ns = int(head[252:256].decode("ascii").strip())
        return parse_edf_header(head + f.read(256 * ns))


def channel_index(labels: list[str], wanted: list[str] = CHANNELS_18) -> dict[str, int | None]:
    """Map each wanted bipolar channel to the index of its FIRST occurrence in labels.

    CHB-MIT stores T8-P8 twice in many files; the first occurrence is used. Dummy
    channels ('-', '.') are ignored. Returns None for channels that are missing.
    """
    idx: dict[str, int | None] = {}
    norm = [norm_label(l) for l in labels]
    for w in wanted:
        idx[w] = norm.index(w) if w in norm else None
    return idx
