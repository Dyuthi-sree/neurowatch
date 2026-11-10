#!/usr/bin/env python
"""Stage 3: Random Forest baseline with event-level evaluation.

Protocol
  * train on the 14 training patients only: all positive windows + a seeded random sample
    of N_NEG negative windows (class_weight='balanced_subsample' on top)
  * RF hyper-parameters are fixed a priori (no tuning)
  * post-processing (threshold theta, causal smoothing length k) is chosen on the 4
    validation patients with SELECTION_RULE; the 4 test patients are scored once with it
  * the test grid is also saved for transparency but is never used for any choice
Run:  python scripts/train_rf.py [--channels all18]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from neurowatch import plotting as P  # noqa: E402
from neurowatch.config import CACHE, CHANNELS_18, FIGURES, MODELS, RESULTS, SEED  # noqa: E402
from neurowatch.evaluation import evaluate  # noqa: E402
from neurowatch.features import columns_for_channels, feature_columns  # noqa: E402
from neurowatch.report import md_table  # noqa: E402

N_NEG = 200_000
RF_PARAMS = dict(n_estimators=300, max_features="sqrt", min_samples_leaf=5,
                 class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
THETAS = np.round(np.arange(0.05, 0.96, 0.05), 2)
KS = [1, 3, 5, 8]
FA_TARGET = 1.0
SELECTION_RULE = (f"maximise validation sensitivity subject to FA/h <= {FA_TARGET}; ties -> lower FA/h, "
                  "then lower median delay; if no setting meets the FA target, minimise FA/h")


def grid(win, prob, sz, dur):
    rows = []
    for k in KS:
        for th in THETAS:
            m = evaluate(win, prob, sz, dur, theta=float(th), k=k)
            m.pop("per_patient")
            rows.append({"k": k, "theta": float(th), **m})
    return pd.DataFrame(rows)


def select(g: pd.DataFrame) -> pd.Series:
    ok = g[g.fa_per_hour <= FA_TARGET]
    if len(ok):
        return ok.sort_values(["sensitivity", "fa_per_hour", "median_delay_s"],
                              ascending=[False, True, True]).iloc[0]
    return g.sort_values(["fa_per_hour", "sensitivity"], ascending=[True, False]).iloc[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="all18", help="'all18' or a JSON file with a channel list")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    chans = CHANNELS_18 if a.channels == "all18" else json.loads(Path(a.channels).read_text())
    name = a.name or f"rf_{a.channels if a.channels == 'all18' else Path(a.channels).stem}"
    cols = columns_for_channels(chans)

    t0 = time.time()
    win = pd.read_parquet(CACHE / "windows.parquet")
    X = np.load(CACHE / "features.npy", mmap_mode="r")
    rec = pd.read_csv(RESULTS / "recordings.csv")
    dur = rec.set_index("record").duration_s
    sz = pd.read_csv(RESULTS / "seizures.csv")
    sz = sz[sz.included]

    tr = np.flatnonzero((win.split == "train").to_numpy())
    y = win.label.to_numpy()
    pos, neg = tr[y[tr] == 1], tr[y[tr] == 0]
    rng = np.random.default_rng(SEED)
    neg_s = np.sort(rng.choice(neg, size=min(N_NEG, len(neg)), replace=False))
    idx = np.sort(np.r_[pos, neg_s])
    Xtr = np.asarray(X[idx][:, cols])
    print(f"train windows: {len(pos)} positive + {len(neg_s)} of {len(neg)} negative; "
          f"{len(cols)} features")
    rf = RandomForestClassifier(**RF_PARAMS).fit(Xtr, y[idx])
    t_fit = time.time() - t0
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "channels": chans, "columns": cols}, MODELS / f"{name}.joblib")

    out = {"name": name, "channels": chans, "n_features": len(cols), "rf_params": RF_PARAMS,
           "n_train_pos": int(len(pos)), "n_train_neg_sampled": int(len(neg_s)),
           "n_train_neg_total": int(len(neg)), "fit_seconds": round(t_fit, 1),
           "selection_rule": SELECTION_RULE, "merge_gap_s": 30.0}
    preds, grids = {}, {}
    pred_dir = RESULTS / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for s in ["val", "test"]:
        m = (win.split == s).to_numpy()
        w = win[m].reset_index(drop=True)
        p = np.concatenate([rf.predict_proba(np.asarray(X[ch][:, cols]))[:, 1]
                            for ch in np.array_split(np.flatnonzero(m), 20)])
        preds[s] = (w, p)
        w[["record", "patient", "start_s", "label"]].assign(prob=p.astype(np.float32)).to_parquet(
            pred_dir / f"{name}_{s}.parquet", index=False)
        out[f"{s}_window_auroc"] = float(roc_auc_score(w.label, p))
        out[f"{s}_window_auprc"] = float(average_precision_score(w.label, p))
        out[f"{s}_positive_rate"] = float(w.label.mean())
        grids[s] = grid(w, p, sz[sz.patient.isin(w.patient.unique())], dur)
        grids[s].to_csv(RESULTS / f"{name}_grid_{s}.csv", index=False)

    best = select(grids["val"])
    k, th = int(best.k), float(best.theta)
    out["selected"] = {"k": k, "theta": th}
    for s in ["val", "test"]:
        w, p = preds[s]
        ev = evaluate(w, p, sz[sz.patient.isin(w.patient.unique())], dur, theta=th, k=k)
        out[f"{s}_event"] = ev
    imp = pd.Series(rf.feature_importances_, index=[feature_columns()[c] for c in cols])
    by_ch = imp.groupby(lambda c: c.split("|")[0]).sum().sort_values(ascending=False)
    by_feat = imp.groupby(lambda c: c.split("|")[1]).sum().sort_values(ascending=False)
    out["importance_by_channel"] = by_ch.round(4).to_dict()
    out["importance_by_feature"] = by_feat.round(4).to_dict()
    out["total_seconds"] = round(time.time() - t0, 1)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n")
    plot(name, grids, k, out)
    write_report(name, out, grids)


def plot(name, grids, k, out):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for s, c in [("val", P.SERIES[0]), ("test", P.SERIES[1])]:
        g = grids[s][grids[s].k == k].sort_values("theta")
        ax.plot(g.fa_per_hour, g.sensitivity, color=c, marker="o", ms=3, label=f"{s} patients (k={k})")
        e = out[f"{s}_event"]
        ax.plot(e["fa_per_hour"], e["sensitivity"], marker="*", ms=13, color=c, mec=P.TEXT, mew=0.6)
    ax.axvline(FA_TARGET, color=P.TEXT2, lw=0.8, ls="--")
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("False alarms per hour")
    ax.set_ylabel("Event sensitivity")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)
    ax.set_title(f"{name}: threshold sweep; ★ = θ={out['selected']['theta']} chosen on val")
    ax.legend(loc="lower right")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}_sweep.png")
    plt.close(fig)


def write_report(name, out, grids):
    def row(s):
        e = out[f"{s}_event"]
        return {"split": s, "seizures": e["n_seizures"], "detected": e["n_detected"],
                "sensitivity": round(e["sensitivity"], 3), "false_alarms": e["n_false_alarms"],
                "hours": round(e["hours"], 2), "FA/h": round(e["fa_per_hour"], 3),
                "median_delay_s": round(e["median_delay_s"], 1),
                "window_AUROC": round(out[f"{s}_window_auroc"], 3),
                "window_AUPRC": round(out[f"{s}_window_auprc"], 3)}
    per = []
    for s in ["val", "test"]:
        for p, e in out[f"{s}_event"]["per_patient"].items():
            per.append({"split": s, "patient": p, "seizures": e["n_seizures"], "detected": e["n_detected"],
                        "false_alarms": e["n_false_alarms"], "hours": round(e["hours"], 2),
                        "FA/h": round(e["fa_per_hour"], 3), "median_delay_s": round(e["median_delay_s"], 1)})
    calib = []
    for s in ["val", "test"]:
        p = pd.read_parquet(RESULTS / "predictions" / f"{name}_{s}.parquet")
        q = p.groupby(["patient", "label"], observed=True).prob.median().unstack()
        for pt, r in q.iterrows():
            calib.append({"split": s, "patient": pt, "median_prob_nonseizure": round(float(r[0]), 3),
                          "median_prob_seizure": round(float(r[1]), 3)})
    gv = grids["val"]
    k = out["selected"]["k"]
    sweep = grids["test"][grids["test"].k == k][["theta", "sensitivity", "fa_per_hour", "median_delay_s"]].round(3)
    lines = [f"# {name}: Random Forest baseline (computed by scripts/train_rf.py)\n",
             f"- Channels: {len(out['channels'])}; features: {out['n_features']}",
             f"- Training windows (14 train patients): {out['n_train_pos']} positive + "
             f"{out['n_train_neg_sampled']} sampled of {out['n_train_neg_total']} negative",
             f"- RF: {json.dumps({k_: v for k_, v in out['rf_params'].items() if k_ != 'n_jobs'})}",
             f"- Post-processing chosen on validation patients: k = {out['selected']['k']} "
             f"(causal moving average), θ = {out['selected']['theta']}; alarm events merged within 30 s",
             f"- Selection rule: {out['selection_rule']}",
             f"- Validation settings meeting FA/h ≤ {FA_TARGET}: {int((gv.fa_per_hour <= FA_TARGET).sum())} of {len(gv)}",
             "\n## Event-level results at the selected operating point\n",
             md_table(pd.DataFrame([row("val"), row("test")])),
             "\n## Per patient\n", md_table(pd.DataFrame(per)),
             "\n## Per-patient probability calibration (window level)\n", md_table(pd.DataFrame(calib)),
             f"\n## Test-patient sweep at k = {k} (for reporting only; not used for any choice)\n",
             md_table(sweep),
             "\n## Importance (impurity, summed)\n",
             "By feature: " + ", ".join(f"{k_} {v:.3f}" for k_, v in out["importance_by_feature"].items()),
             "\nBy channel: " + ", ".join(f"{k_} {v:.3f}" for k_, v in out["importance_by_channel"].items()),
             f"\nRuntime: fit {out['fit_seconds']} s, total {out['total_seconds']} s. "
             f"Figure: figures/{name}_sweep.png"]
    text = "\n".join(lines)
    (RESULTS / f"{name}_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
