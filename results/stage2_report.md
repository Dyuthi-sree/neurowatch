# Stage 2: preprocessing, windowing and cache (computed by scripts/stage2_report.py)

Pipeline per recording: 18 bipolar channels (µV; 3 chb12 files rebuilt from electrodes) → causal 4th-order Butterworth 0.5–40 Hz (zi = steady state at the first sample) → 4 s windows every 2 s → label = 1 if ≥ 50 % of the window lies in an annotated seizure → 6 features × 18 channels; filtered signal also cached at 128 Hz (float16) for the CNN.

## Windows per split

| split | patients | files | hours | seizures | ictal_minutes | windows | positive_windows | positive_pct | windows_partial_overlap_negative |
|---|---|---|---|---|---|---|---|---|---|
| train | 14 | 382 | 607.7 | 117 | 124.6 | 1093464 | 3789 | 0.347 | 130 |
| val | 4 | 101 | 174.76 | 32 | 20.7 | 314467 | 636 | 0.202 | 34 |
| test | 4 | 181 | 179.18 | 33 | 39.7 | 322337 | 1213 | 0.376 | 24 |
| all | 22 | 664 | 961.64 | 182 | 185.0 | 1730268 | 5638 | 0.326 | 188 |

- Recordings built: 664 (3 with rebuilt montage)
- Values clipped when casting the 128 Hz signal to float16 (|x| > 60000 µV): 0
- Seizures with ≥ 1 positive window: 182 of 182
- Positive windows per seizure: min 4, median 24, max 377
- Windows with a near-flat channel (std < 0.5 µV): 1165 (0.07 %), of which positive: 1
- Windows with a filtered sample above 1000 µV: 38807 (2.24 %)
- Feature matrix: (1730268, 108), float32

Leakage controls: split is by patient (chb01+chb21 together, chb24 excluded); every step is per recording; the filter is causal and features use only samples inside their window; nothing is fitted on data before the split. See tests/test_leakage.py and tests/test_preprocessing.py.
