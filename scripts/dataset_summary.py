#!/usr/bin/env python
"""Stage 1: count what is actually in CHB-MIT, check channel availability, fix the
reference channel set, log excluded recordings, and draw the dataset figures.

Needs: metadata + EDF headers (scripts/download_data.py --metadata-only / --headers-only).
The sample EEG trace additionally needs the EDF given by --trace (default chb01_03).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from neurowatch import plotting as P  # noqa: E402
from neurowatch.channels import channel_index, read_edf_header  # noqa: E402
from neurowatch.montage import resolve  # noqa: E402
from neurowatch.config import (CHANNELS_18, EXCLUDED_CASES, FIGURES, FS, HEADERS,  # noqa: E402
                               PATIENT_OF_CASE, RAW, RESULTS)
from neurowatch.summary_parser import parse_summary  # noqa: E402

CASES = [f"chb{i:02d}" for i in range(1, 25)]


def lines(p: Path) -> list[str]:
    return [l.strip() for l in p.read_text().splitlines() if l.strip()]


def subject_info() -> pd.DataFrame:
    rows = []
    for l in lines(RAW / "SUBJECT-INFO")[1:]:
        parts = l.split()
        rows.append({"case": parts[0], "sex": parts[1], "age_years": float(parts[2])})
    return pd.DataFrame(rows)


def build_tables():
    records = lines(RAW / "RECORDS")
    rws = {r.split("/")[1] for r in lines(RAW / "RECORDS-WITH-SEIZURES")}
    summ = {}
    for c in CASES:
        for e in parse_summary(RAW / c / f"{c}-summary.txt"):
            summ[e.file] = e
    rec_rows, sz_rows = [], []
    for r in records:
        case, fname = r.split("/")
        h = read_edf_header(HEADERS / r)
        idx = channel_index(h["labels"])
        missing = [k for k, v in idx.items() if v is None]
        try:
            montage = resolve(h["labels"])["mode"]
        except KeyError:
            montage = "unavailable"
        e = summ.get(fname)
        n_sz = len(e.starts) if e else 0
        rec_rows.append({
            "case": case, "patient": PATIENT_OF_CASE[case], "file": fname, "record": r,
            "duration_s": h["duration_s"], "n_signals": h["n_signals"],
            "fs": sorted(set(h["fs"]))[0] if len(set(h["fs"])) == 1 else str(sorted(set(h["fs"]))),
            "n_ref18_present": 18 - len(missing), "missing_channels": ";".join(missing),
            "montage": montage,
            "in_summary": e is not None, "in_records_with_seizures": fname in rws,
            "n_seizures": n_sz,
        })
        if e:
            for i, (s, t) in enumerate(e.seizures, 1):
                sz_rows.append({"case": case, "patient": PATIENT_OF_CASE[case], "file": fname,
                                "record": r, "seizure": i, "start_s": s, "end_s": t,
                                "duration_s": t - s})
    rec = pd.DataFrame(rec_rows)
    sz = pd.DataFrame(sz_rows)
    # seizure annotations beyond the file end would be a data problem -> report
    m = sz.merge(rec[["record", "duration_s"]].rename(columns={"duration_s": "file_dur"}), on="record")
    beyond = m[m.end_s > m.file_dur]
    return rec, sz, summ, rws, beyond


def choose_reference(rec: pd.DataFrame) -> dict:
    inc = rec[~rec.case.isin(EXCLUDED_CASES)]
    native = inc[inc.n_ref18_present == 18]
    full = inc[inc.montage != "unavailable"]
    out = {
        "candidate": "all 18 bipolar channels",
        "channels": CHANNELS_18,
        "recordings_considered": int(len(inc)),
        "recordings_with_native_bipolar_18": int(len(native)),
        "seizures_with_native_bipolar_18": int(native.n_seizures.sum()),
        "recordings_rebuilt_from_referential": sorted(inc[inc.montage == "rebuilt"].record.tolist()),
        "recordings_with_all_18": int(len(full)),
        "hours_considered": round(inc.duration_s.sum() / 3600, 2),
        "hours_with_all_18": round(full.duration_s.sum() / 3600, 2),
        "seizures_considered": int(inc.n_seizures.sum()),
        "seizures_with_all_18": int(full.n_seizures.sum()),
    }
    # which channels, if dropped, would rescue the incomplete recordings?
    partial = inc[inc.n_ref18_present.between(1, 17)]
    out["recordings_with_some_but_not_all_18_native"] = int(len(partial))
    out["recordings_with_none_of_18_native"] = int((inc.n_ref18_present == 0).sum())
    rule = "use all 18 if >=95% of recordings and >=90% of seizures keep them"
    ok = (len(full) / len(inc) >= 0.95) and (full.n_seizures.sum() / inc.n_seizures.sum() >= 0.90)
    out["decision_rule"] = rule
    out["decision"] = "all 18" if ok else "NEEDS REDUCED SET"
    return out


def plot_dataset(per_case: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    y = np.arange(len(per_case))[::-1]
    colors = [P.GRID if c in EXCLUDED_CASES else P.SERIES[0] for c in per_case.case]
    axes[0].barh(y, per_case.hours, color=colors, height=0.7)
    axes[0].set_xlabel("Recorded EEG (hours)")
    axes[0].set_title("Recording time per case")
    colors2 = [P.GRID if c in EXCLUDED_CASES else P.SERIES[1] for c in per_case.case]
    axes[1].barh(y, per_case.n_seizures, color=colors2, height=0.7)
    axes[1].set_xlabel("Annotated seizures (count)")
    axes[1].set_title("Seizures per case")
    labels = [f"{c} (= chb01 patient)" if c == "chb21" else (f"{c} (excluded)" if c in EXCLUDED_CASES else c)
              for c in per_case.case]
    axes[0].set_yticks(y, labels)
    axes[0].set_ylabel("Case")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for yi, v in zip(y, per_case.n_seizures):
        axes[1].text(v + 0.5, yi, str(int(v)), va="center", fontsize=7, color=P.TEXT2)
    fig.suptitle("CHB-MIT Scalp EEG Database: counts computed from summary files and EDF headers",
                 fontsize=10, color=P.TEXT)
    fig.savefig(FIGURES / "dataset_summary.png")
    plt.close(fig)


def plot_trace(record: str, seizures: pd.DataFrame, pad_s: float = 40.0):
    from neurowatch.edf import load_channels
    from neurowatch.filtering import causal_bandpass

    path = RAW / record
    if not path.exists():
        print(f"[trace] {record} not downloaded yet; skipping sample trace")
        return None
    s = seizures[seizures.record == record].iloc[0]
    x = causal_bandpass(load_channels(path))  # filter whole file causally, then crop
    t0, t1 = max(0.0, s.start_s - pad_s), s.end_s + pad_s
    seg = x[:, int(t0 * FS): int(t1 * FS)]
    t = t0 + np.arange(seg.shape[1]) / FS
    pre = seg[:, : int((s.start_s - t0) * FS)]
    # spacing from pre-seizure amplitude, rounded to 50 uV; traces clipped to +-spacing
    spacing = float(max(50.0, np.ceil(8 * np.median(pre.std(axis=1)) / 50) * 50))
    seg = np.clip(seg, -spacing, spacing)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axvspan(s.start_s, s.end_s, color=P.SEIZURE_SHADE, alpha=0.18, lw=0,
               label=f"Annotated seizure ({s.start_s:.0f}-{s.end_s:.0f} s)")
    for i, ch in enumerate(CHANNELS_18):
        ax.plot(t, seg[i] - i * spacing, color=P.TEXT, lw=0.4)
    ax.set_yticks([-i * spacing for i in range(18)], CHANNELS_18)
    ax.set_ylim(-18 * spacing, spacing)
    ax.set_xlim(t0, t1)
    ax.set_xlabel("Time from start of recording (s)")
    ax.set_ylabel(f"Bipolar channel (offset {spacing:.0f} µV per trace)")
    ax.set_title(f"{record}: 0.5–40 Hz causal band-pass, seizure {int(s.seizure)} marked "
                 f"(traces clipped at ±{spacing:.0f} µV)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=1)
    fig.savefig(FIGURES / "sample_eeg_trace.png")
    plt.close(fig)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="chb01/chb01_03.edf")
    a = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    rec, sz, summ, rws, beyond = build_tables()
    info = subject_info()
    ref = choose_reference(rec)
    use18 = ref["decision"] == "all 18"

    rec["excluded_reason"] = ""
    rec.loc[rec.case.isin(EXCLUDED_CASES), "excluded_reason"] = "case excluded (chb24)"
    lack = (~rec.case.isin(EXCLUDED_CASES)) & (rec.montage == "unavailable")
    rec.loc[lack, "excluded_reason"] = "missing reference channels"
    rec["included"] = rec.excluded_reason == ""
    rec.to_csv(RESULTS / "recordings.csv", index=False)
    sz = sz.merge(rec[["record", "included"]], on="record")
    sz.to_csv(RESULTS / "seizures.csv", index=False)
    rec[lack][["case", "file", "duration_s", "n_seizures", "missing_channels"]].assign(
        reason="reference channels unavailable (neither bipolar nor rebuildable)"
    ).to_csv(RESULTS / "excluded_recordings.csv", index=False)
    rb = (~rec.case.isin(EXCLUDED_CASES)) & (rec.montage == "rebuilt")
    rec[rb][["case", "file", "duration_s", "n_seizures"]].assign(
        how="18 bipolar channels rebuilt as electrode differences (referential recording)"
    ).to_csv(RESULTS / "rebuilt_recordings.csv", index=False)
    with open(RESULTS / "reference_channels.json", "w") as f:
        json.dump(ref, f, indent=2)

    g = rec.groupby("case")
    per_case = pd.DataFrame({
        "case": CASES,
        "patient": [PATIENT_OF_CASE[c] for c in CASES],
        "n_files": g.size().reindex(CASES).values,
        "hours": (g.duration_s.sum() / 3600).reindex(CASES).round(2).values,
        "n_seizures": g.n_seizures.sum().reindex(CASES).values,
        "n_files_used": g.included.sum().reindex(CASES).values,
        "hours_used": (rec[rec.included].groupby("case").duration_s.sum() / 3600).reindex(CASES).fillna(0).round(2).values,
        "n_seizures_used": rec[rec.included].groupby("case").n_seizures.sum().reindex(CASES).fillna(0).astype(int).values,
    }).merge(info, on="case", how="left")
    per_case.to_csv(RESULTS / "dataset_summary.csv", index=False)
    plot_dataset(per_case)
    traced = plot_trace(a.trace, sz)

    # ---------------- report ----------------
    no24 = per_case[~per_case.case.isin(EXCLUDED_CASES)]
    used = rec[rec.included]
    rep = []
    rep.append("# Dataset summary (computed by scripts/dataset_summary.py)\n")
    rep.append(f"- Cases in database: {len(per_case)} (chb01–chb24); files in RECORDS: {len(rec)}; "
               f"total hours: {rec.duration_s.sum() / 3600:.2f}; total annotated seizures: {int(per_case.n_seizures.sum())}")
    rep.append(f"- Excluding chb24: {len(no24)} cases, {no24.patient.nunique()} patients, "
               f"{int(no24.n_files.sum())} files, {rec[~rec.case.isin(EXCLUDED_CASES)].duration_s.sum() / 3600:.2f} h, {int(no24.n_seizures.sum())} seizures; "
               f"ages {no24.age_years.min():g}–{no24.age_years.max():g} y "
               f"({(no24.sex == 'F').sum()} F / {(no24.sex == 'M').sum()} M cases)")
    rep.append(f"- chb24: {int(per_case.set_index('case').loc['chb24', 'n_files'])} files, "
               f"{int(per_case.set_index('case').loc['chb24', 'n_seizures'])} seizures; "
               f"age in SUBJECT-INFO: {per_case.set_index('case').loc['chb24', 'age_years']}")
    rep.append(f"- Sampling rate: all files {sorted(rec.fs.unique().tolist())} Hz")
    rep.append(f"- After channel check (used for modelling): {len(used)} files, "
               f"{used.duration_s.sum() / 3600:.2f} h, {int(used.n_seizures.sum())} seizures, "
               f"{used.patient.nunique()} patients")
    rep.append("\n## Reference channel set\n")
    rep.append("```json\n" + json.dumps(ref, indent=2) + "\n```")
    rep.append("\n## Recordings rebuilt from referential electrodes (kept)\n")
    for _, r in rec[rb].iterrows():
        rep.append(f"- {r.record}: {r.n_seizures} seizures, {r.duration_s / 3600:.2f} h")
    rep.append("\n## Excluded recordings (reference channels unavailable)\n")
    ex = rec[lack]
    if ex.empty:
        rep.append("- none")
    for _, r in ex.iterrows():
        rep.append(f"- {r.record}: {r.n_seizures} seizures, {r.duration_s / 3600:.2f} h, "
                   f"{18 - r.n_ref18_present} of 18 channels missing")
    rep.append("\n## Metadata inconsistencies found\n")
    recnames = set(rec.file)
    rep.append(f"- Files in RECORDS but not listed in any summary file: "
               f"{sorted(recnames - set(summ))}")
    rep.append(f"- Files in summaries but not in RECORDS: {sorted(set(summ) - recnames)}")
    sz_files = set(sz.file)
    rep.append(f"- Files with seizures per summary but not in RECORDS-WITH-SEIZURES: {sorted(sz_files - rws)}")
    rep.append(f"- Files in RECORDS-WITH-SEIZURES with no seizure in summary: {sorted(rws - sz_files)}")
    rep.append(f"- Seizure annotations extending past end of file: "
               f"{beyond[['record', 'seizure', 'end_s', 'file_dur']].to_dict('records')}")
    rep.append(f"- Seizure durations (excl. chb24): min {sz[sz.case != 'chb24'].duration_s.min():.0f} s, "
               f"median {sz[sz.case != 'chb24'].duration_s.median():.0f} s, "
               f"max {sz[sz.case != 'chb24'].duration_s.max():.0f} s")
    rep.append(f"\nSample trace: {traced or 'not drawn (file not downloaded)'}")
    text = "\n".join(rep)
    (RESULTS / "dataset_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
