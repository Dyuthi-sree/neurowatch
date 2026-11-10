#!/usr/bin/env python
"""Stage 2: preprocess every included recording and cache windows, labels and features.

Per recording (independently, so nothing crosses file or patient boundaries):
  1. load the 18 bipolar channels (rebuilt from electrodes for chb12_27/28/29), µV
  2. causal 0.5-40 Hz Butterworth band-pass over the whole continuous file
  3. 4 s windows every 2 s (complete windows only), label = >=50% seizure overlap
  4. RF features (6 per channel) from each window of the 256 Hz filtered signal
  5. filtered signal decimated 2x to 128 Hz, float16, for the CNN (windows are cut on
     the fly at sample 256*k, which aligns with the 256 Hz windows)

Outputs in data/cache/:
  files/<case>/<file>.npz      per-file features + window table (resumable)
  sig128/<case>/<file>.npy     (18, n_samples/2) float16
  windows.parquet              one row per window (record, patient, split, start_s, label, ...)
  features.npy                 (n_windows, 108) float32, row-aligned with windows.parquet
  meta.json                    parameters used to build the cache
Run:  python scripts/build_cache.py --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neurowatch.config import (BAND, CACHE, CHANNELS_18, CNN_FS, FEATURE_NAMES, FS, RAW,  # noqa: E402
                               RESULTS, STEP_S, WIN_S)
from neurowatch.edf import load_channels  # noqa: E402
from neurowatch.features import window_features  # noqa: E402
from neurowatch.filtering import causal_bandpass  # noqa: E402
from neurowatch.splits import load_split  # noqa: E402
from neurowatch.windows import POS_OVERLAP, frame, label_windows, window_starts_s  # noqa: E402

DECIM = FS // CNN_FS
F16_MAX = 60000.0
CHUNK = 256  # windows per feature batch
META = {"fs": FS, "cnn_fs": CNN_FS, "band": BAND, "filter": "butter order 4, causal sosfilt, zi=x[0] steady state",
        "win_s": WIN_S, "step_s": STEP_S, "pos_overlap": POS_OVERLAP, "channels": CHANNELS_18,
        "features": FEATURE_NAMES, "version": 1}


def meta_hash() -> str:
    return hashlib.sha1(json.dumps(META, sort_keys=True).encode()).hexdigest()[:10]


def process(record: str, seizures: list[tuple[float, float]]) -> dict:
    case, fname = record.split("/")
    stem = fname[:-4]
    out_npz = CACHE / "files" / case / f"{stem}.npz"
    out_sig = CACHE / "sig128" / case / f"{stem}.npy"
    if out_npz.exists() and out_sig.exists():
        with np.load(out_npz) as z:
            return {"record": record, "status": "cached", "mode": str(z["mode"]),
                    "seconds": 0.0, "n_clipped_f16": int(z["n_clipped_f16"])}
    t0 = time.time()
    x, mode = load_channels(RAW / record, return_mode=True)
    y = causal_bandpass(x)
    del x
    v = frame(y)
    feats = np.concatenate([window_features(v[i:i + CHUNK]) for i in range(0, len(v), CHUNK)]
                           ) if len(v) else np.zeros((0, 18 * len(FEATURE_NAMES)), np.float32)
    max_abs = np.abs(v).max(axis=(1, 2)).astype(np.float32) if len(v) else np.zeros(0, np.float32)
    min_std = feats[:, 0::len(FEATURE_NAMES)].min(1) if len(v) else np.zeros(0, np.float32)
    starts = window_starts_s(y.shape[1])
    frac, lab, szi = label_windows(starts, seizures)
    s128 = y[:, ::DECIM]
    n_clip = int((np.abs(s128) > F16_MAX).sum())
    s128 = np.clip(s128, -F16_MAX, F16_MAX).astype(np.float16)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_sig.parent.mkdir(parents=True, exist_ok=True)
    tmp_sig = out_sig.with_suffix(".tmp.npy")
    np.save(tmp_sig, s128)
    os.replace(tmp_sig, out_sig)
    tmp_npz = out_npz.with_suffix(".tmp.npz")
    np.savez(tmp_npz, features=feats, start_s=starts.astype(np.float32), overlap=frac.astype(np.float32),
             label=lab, seizure_idx=szi, max_abs=max_abs, min_std=min_std, mode=mode,
             n_samples=y.shape[1], n_clipped_f16=n_clip)
    os.replace(tmp_npz, out_npz)  # npz written last => its presence marks the file done
    return {"record": record, "status": "built", "mode": mode, "seconds": time.time() - t0,
            "n_clipped_f16": n_clip}


def merge(rec: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows, feats = [], []
    for r in rec.itertuples():
        case, fname = r.record.split("/")
        with np.load(CACHE / "files" / case / f"{fname[:-4]}.npz") as z:
            n = len(z["start_s"])
            rows.append(pd.DataFrame({
                "record": r.record, "case": case, "patient": r.patient, "split": r.split,
                "win_idx": np.arange(n, dtype=np.int32), "start_s": z["start_s"],
                "overlap": z["overlap"], "label": z["label"], "seizure_idx": z["seizure_idx"],
                "max_abs": z["max_abs"], "min_std": z["min_std"],
            }))
            feats.append(z["features"])
    win = pd.concat(rows, ignore_index=True)
    for c in ("record", "case", "patient", "split"):
        win[c] = win[c].astype("category")
    return win, np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N recordings (smoke test)")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    mpath = CACHE / "meta.json"
    if mpath.exists() and json.loads(mpath.read_text()).get("hash") != meta_hash():
        sys.exit("cache/meta.json parameters differ from current config; delete data/cache to rebuild")
    mpath.write_text(json.dumps({**META, "hash": meta_hash()}, indent=2))

    rec = pd.read_csv(RESULTS / "recordings.csv")
    rec = rec[rec.included].copy()
    split = load_split()
    rec["split"] = rec.case.map(split)
    assert rec.split.notna().all()
    sz = pd.read_csv(RESULTS / "seizures.csv")
    szmap = {r: list(zip(g.start_s, g.end_s)) for r, g in sz.groupby("record")}
    if a.limit:
        rec = rec.head(a.limit)

    t0 = time.time()
    status = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, r, szmap.get(r, [])): r for r in rec.record}
        for i, f in enumerate(as_completed(futs), 1):
            s = f.result()
            status.append(s)
            if s["status"] == "built":
                print(f"[{i}/{len(futs)}] {s['record']} {s['mode']} {s['seconds']:.1f}s", flush=True)
    st = pd.DataFrame(status)
    print(f"built {int((st.status == 'built').sum())}, cached {int((st.status == 'cached').sum())}, "
          f"wall {time.time() - t0:.0f}s")
    if a.limit:
        return
    win, feats = merge(rec)
    win.to_parquet(CACHE / "windows.parquet", index=False)
    np.save(CACHE / "features.npy", feats)
    st.to_csv(CACHE / "build_status.csv", index=False)
    print(f"windows: {len(win)}, features: {feats.shape}")


if __name__ == "__main__":
    main()
