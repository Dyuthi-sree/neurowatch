#!/usr/bin/env python
"""Create the 14/4/4 patient split from results/dataset_summary.csv (counts only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from neurowatch.config import RESULTS  # noqa: E402
from neurowatch.splits import choose_split, patient_table  # noqa: E402


def main():
    pt = patient_table(pd.read_csv(RESULTS / "dataset_summary.csv"))
    sp = choose_split(pt)
    (RESULTS / "split.json").write_text(json.dumps(sp, indent=2) + "\n")
    pt["split"] = pt.patient.map(sp["patient_split"])
    pt = pt.sort_values(["split", "patient"])
    pt.to_csv(RESULTS / "split_table.csv", index=False)
    agg = pt.groupby("split").agg(patients=("patient", "size"), hours=("hours", "sum"),
                                  seizures=("n_seizures", "sum"))
    agg["hours_frac"] = (agg.hours / agg.hours.sum()).round(3)
    agg["seizure_frac"] = (agg.seizures / agg.seizures.sum()).round(3)
    print(pt.to_string(index=False))
    print(agg.round(2).to_string())


if __name__ == "__main__":
    main()
