# NeuroWatch — cross-patient seizure detection on CHB-MIT (CSCE/BMEN 5280)

Event-level seizure detection from scalp EEG: Random Forest on handcrafted features vs. a
small 1D CNN, evaluated on held-out patients, plus reduced 8- and 4-channel montages and a
Streamlit dashboard.

## Study design

| Item | Choice |
|---|---|
| Data | CHB-MIT Scalp EEG Database v1.0.0 (PhysioNet), 256 Hz |
| Excluded | chb24 (no age/sex in SUBJECT-INFO; 10 of its 22 files are missing from its summary file) |
| Patients | 22: chb21 is the same subject as chb01 and is merged into patient `chb01` |
| Split | 14 / 4 / 4 patients (train / val / test); chb01 + chb21 always in the same split |
| Channels | 18 longitudinal bipolar channels (double banana) |
| Filter | causal 4th-order Butterworth band-pass 0.5–40 Hz (forward-only `sosfilt`) |
| Windows | 4 s, every 2 s; positive if ≥ 50 % of the window lies inside an annotated seizure |
| Models | Random Forest (6 features × channel) vs. small 1D CNN (128 Hz signal) |
| Evaluation | event level on held-out patients: sensitivity, false alarms/hour, median detection delay |
| Subsets | 8- and 4-channel montages |

## Data decision: chb12 referential recordings are recovered

Three chb12 recordings (`chb12_27`, `chb12_28`, `chb12_29`; 13 seizures, 3.01 h) were not
stored in the bipolar montage: `chb12_27` has every electrode against a common reference
(labels like `F7-CS2`), and `chb12_28/29` have bare electrode labels, with the left
occipital electrode mislabelled `01` instead of `O1`.

**Decision:** these files are kept and the 18 bipolar channels are rebuilt as electrode
differences, `A-B = (A − ref) − (B − ref)`. The common reference cancels exactly, so the
result is the same quantity a bipolar recording measures. This restores the dataset to
**664 files, 961.64 h, 182 seizures** (it would be 661 files, 958.63 h, 169 seizures without
them). Implementation: `neurowatch/montage.py`; list: `results/rebuilt_recordings.csv`.

Sanity check (filtered signal, median |x| per channel, seizure-free minutes): rebuilt
`chb12_28` 11–22 µV and `chb12_29` 17–28 µV vs. native-bipolar `chb12_24` 20–44 µV and
`chb12_33` 12–23 µV. `chb12_27` is larger at the start of the file (18–91 µV, 40 % of the
power in the beta band, consistent with muscle artefact) but 17–27 µV in its final 10 minutes.

## Dataset (computed by `scripts/dataset_summary.py`, see `results/dataset_report.md`)

- Whole database: 24 cases, 686 files, 982.94 h, 198 seizures
- Used (chb24 excluded): 23 cases = 22 patients, 664 files, 961.64 h, 182 seizures
- Metadata quirks: `chb07_19` has a seizure in its summary but is absent from
  RECORDS-WITH-SEIZURES (and `chb07_18` the reverse); the summary file is used.
  `chb17b_69.edf` has 256 bytes beyond the length declared in its header (ignored by the reader).

## Split (`scripts/make_split.py`, `results/split.json`)

Chosen from counts only: among 20 000 seeded random 14/4/4 assignments, the one whose val
and test sets are closest to 4/22 of all seizures and of all hours.

| Split | Patients | Hours | Seizures |
|---|---|---|---|
| train | chb02 chb04 chb07 chb08 chb09 chb12 chb14 chb15 chb17 chb18 chb19 chb20 chb22 chb23 | 607.7 | 117 |
| val | chb05 chb06 chb10 chb16 | 174.8 | 32 |
| test | chb01 (+chb21) chb03 chb11 chb13 | 179.2 | 33 |

## Pipeline

```bash
source .venv/bin/activate
python scripts/download_data.py --patients all --workers 4   # resumable, ~42 GB without chb24
python scripts/download_data.py --headers-only --patients all
python scripts/dataset_summary.py      # Stage 1: counts, channel check, figures
python scripts/make_split.py           # patient split
python scripts/build_cache.py --workers 8   # Stage 2: filter, window, label, features (~2 min)
python scripts/stage2_report.py
python scripts/train_rf.py             # Stage 3: Random Forest + event-level evaluation
python -m pytest tests -q
```

## Stages

1. Repo, downloader, dataset summary, channel availability, sample trace — done
2. Preprocessing, windowing, caching, leakage tests — done (`results/stage2_report.md`)
3. Random Forest baseline with event-level evaluation — done (`results/rf_all18_report.md`); see note below
4. 1D CNN — todo
5. 8 / 4 channel subsets — todo
6. Streamlit dashboard — todo

## Stage 3 note: RF operating point does not transfer from val to test

With the pre-specified rule (max validation sensitivity at FA/h ≤ 1), validation selects
k = 5, θ = 0.95 → val 9/32 seizures, 0.023 FA/h; **test 0/33 seizures, 0 FA/h**. Window-level
ranking is reasonable (AUROC val 0.889, test 0.904), but probability scales differ by
patient: validation patient chb06 has a median non-seizure probability of 0.228 vs.
0.006–0.039 for all test patients, which forces the threshold up. The full test sweep is
in `results/rf_all18_report.md` for transparency only; it was not used to choose anything.
