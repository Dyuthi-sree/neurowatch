#!/usr/bin/env python
"""Summarise the Stage 2 cache: windows, labels, splits (numbers computed from the cache)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neurowatch.config import CACHE, RESULTS  # noqa: E402
from neurowatch.report import md_table  # noqa: E402


def main():
    win = pd.read_parquet(CACHE / "windows.parquet")
    st = pd.read_csv(CACHE / "build_status.csv")
    sz = pd.read_csv(RESULTS / "seizures.csv")
    sz = sz[sz.included]
    rec = pd.read_csv(RESULTS / "recordings.csv")
    rec = rec[rec.included]
    split = json.loads((RESULTS / "split.json").read_text())["patient_split"]
    rec["split"] = rec.patient.map(split)
    sz["split"] = sz.patient.map(split)

    rows = []
    for s in ["train", "val", "test", "all"]:
        w = win if s == "all" else win[win.split == s]
        r = rec if s == "all" else rec[rec.split == s]
        z = sz if s == "all" else sz[sz.split == s]
        rows.append({
            "split": s, "patients": r.patient.nunique(), "files": len(r),
            "hours": round(r.duration_s.sum() / 3600, 2), "seizures": len(z),
            "ictal_minutes": round(z.duration_s.sum() / 60, 1),
            "windows": len(w), "positive_windows": int(w.label.sum()),
            "positive_pct": round(100 * w.label.mean(), 3),
            "windows_partial_overlap_negative": int(((w.overlap > 0) & (w.label == 0)).sum()),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "stage2_windows.csv", index=False)

    pos = win[win.label == 1]
    per_sz = pos.groupby(["record", "seizure_idx"], observed=True).size()
    flat = win.min_std < 0.5  # any channel with std < 0.5 µV in the window
    big = win.max_abs > 1000  # any filtered sample above 1 mV
    lines = ["# Stage 2: preprocessing, windowing and cache (computed by scripts/stage2_report.py)\n",
             "Pipeline per recording: 18 bipolar channels (µV; 3 chb12 files rebuilt from electrodes) → "
             "causal 4th-order Butterworth 0.5–40 Hz (zi = steady state at the first sample) → "
             "4 s windows every 2 s → label = 1 if ≥ 50 % of the window lies in an annotated seizure → "
             "6 features × 18 channels; filtered signal also cached at 128 Hz (float16) for the CNN.\n",
             "## Windows per split\n", md_table(tab), "",
             f"- Recordings built: {len(st)} ({int((st['mode'] == 'rebuilt').sum())} with rebuilt montage)",
             f"- Values clipped when casting the 128 Hz signal to float16 (|x| > 60000 µV): "
             f"{int(st.n_clipped_f16.sum())}",
             f"- Seizures with ≥ 1 positive window: {per_sz.shape[0]} of {len(sz)}",
             f"- Positive windows per seizure: min {per_sz.min()}, median {per_sz.median():.0f}, max {per_sz.max()}",
             f"- Windows with a near-flat channel (std < 0.5 µV): {int(flat.sum())} "
             f"({100 * flat.mean():.2f} %), of which positive: {int((flat & (win.label == 1)).sum())}",
             f"- Windows with a filtered sample above 1000 µV: {int(big.sum())} ({100 * big.mean():.2f} %)",
             f"- Feature matrix: {np.load(CACHE / 'features.npy', mmap_mode='r').shape}, float32",
             "",
             "Leakage controls: split is by patient (chb01+chb21 together, chb24 excluded); "
             "every step is per recording; the filter is causal and features use only samples "
             "inside their window; nothing is fitted on data before the split. "
             "See tests/test_leakage.py and tests/test_preprocessing.py."]
    text = "\n".join(lines)
    (RESULTS / "stage2_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
