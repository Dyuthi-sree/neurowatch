"""Patient-level 14/4/4 train/validation/test split.

Patients, not cases, are split: chb21 is the same subject as chb01 and is merged into
patient "chb01", and chb24 is excluded, leaving 22 patients. The split is chosen using
recording counts only (never model results): among N_DRAWS seeded random 14/4/4
assignments, keep the one whose validation and test sets are closest to 4/22 of the
total seizures and 4/22 of the total hours.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import EXCLUDED_CASES, PATIENT_OF_CASE, RESULTS, SEED

SIZES = {"train": 14, "val": 4, "test": 4}
N_DRAWS = 20000


def patient_table(per_case: pd.DataFrame) -> pd.DataFrame:
    pc = per_case[~per_case.case.isin(EXCLUDED_CASES)].copy()
    pc["patient"] = pc.case.map(PATIENT_OF_CASE)
    return (pc.groupby("patient")
              .agg(cases=("case", lambda s: ",".join(sorted(s))), hours=("hours_used", "sum"),
                   n_seizures=("n_seizures_used", "sum"))
              .reset_index())


def choose_split(pt: pd.DataFrame, seed: int = SEED, n_draws: int = N_DRAWS) -> dict:
    assert len(pt) == sum(SIZES.values()), len(pt)
    rng = np.random.default_rng(seed)
    pats = pt.patient.to_numpy()
    sz = pt.n_seizures.to_numpy(float)
    hr = pt.hours.to_numpy(float)
    target = SIZES["val"] / len(pt)
    best, best_score = None, np.inf
    for _ in range(n_draws):
        perm = rng.permutation(len(pt))
        val, test = perm[:4], perm[4:8]
        score = 0.0
        for g in (val, test):
            score += abs(sz[g].sum() / sz.sum() - target) + abs(hr[g].sum() / hr.sum() - target)
        if score < best_score:
            best_score, best = score, perm.copy()
    assign = {}
    for i in best[:4]:
        assign[pats[i]] = "val"
    for i in best[4:8]:
        assign[pats[i]] = "test"
    for i in best[8:]:
        assign[pats[i]] = "train"
    return {"seed": seed, "n_draws": n_draws, "score": round(float(best_score), 5),
            "rule": "min sum |fraction - 4/22| of seizures and hours over val and test",
            "patient_split": dict(sorted(assign.items()))}


def load_split() -> dict[str, str]:
    """case -> split (chb21 gets chb01's split)."""
    sp = json.loads((RESULTS / "split.json").read_text())["patient_split"]
    return {c: sp[p] for c, p in PATIENT_OF_CASE.items() if p in sp}
