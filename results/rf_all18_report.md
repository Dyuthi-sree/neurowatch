# rf_all18: Random Forest baseline (computed by scripts/train_rf.py)

- Channels: 18; features: 108
- Training windows (14 train patients): 3789 positive + 200000 sampled of 1089675 negative
- RF: {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 5, "class_weight": "balanced_subsample", "random_state": 42}
- Post-processing chosen on validation patients: k = 5 (causal moving average), θ = 0.95; alarm events merged within 30 s
- Selection rule: maximise validation sensitivity subject to FA/h <= 1.0; ties -> lower FA/h, then lower median delay; if no setting meets the FA target, minimise FA/h
- Validation settings meeting FA/h ≤ 1.0: 8 of 76

## Event-level results at the selected operating point

| split | seizures | detected | sensitivity | false_alarms | hours | FA/h | median_delay_s | window_AUROC | window_AUPRC |
|---|---|---|---|---|---|---|---|---|---|
| val | 32 | 9 | 0.281 | 4 | 174.76 | 0.023 | 65.0 | 0.889 | 0.197 |
| test | 33 | 0 | 0.0 | 0 | 179.18 | 0.0 | nan | 0.904 | 0.274 |

## Per patient

| split | patient | seizures | detected | false_alarms | hours | FA/h | median_delay_s |
|---|---|---|---|---|---|---|---|
| val | chb05 | 5 | 5 | 1 | 39.0 | 0.026 | 71.0 |
| val | chb06 | 10 | 0 | 3 | 66.74 | 0.045 | nan |
| val | chb10 | 7 | 4 | 0 | 50.02 | 0.0 | 63.5 |
| val | chb16 | 10 | 0 | 0 | 19.0 | 0.0 | nan |
| test | chb01 | 11 | 0 | 0 | 73.38 | 0.0 | nan |
| test | chb03 | 7 | 0 | 0 | 38.0 | 0.0 | nan |
| test | chb11 | 3 | 0 | 0 | 34.79 | 0.0 | nan |
| test | chb13 | 12 | 0 | 0 | 33.0 | 0.0 | nan |

## Per-patient probability calibration (window level)

| split | patient | median_prob_nonseizure | median_prob_seizure |
|---|---|---|---|
| val | chb05 | 0.036 | 0.884 |
| val | chb06 | 0.228 | 0.334 |
| val | chb10 | 0.022 | 0.787 |
| val | chb16 | 0.022 | 0.084 |
| test | chb01 | 0.015 | 0.28 |
| test | chb03 | 0.006 | 0.349 |
| test | chb11 | 0.022 | 0.305 |
| test | chb13 | 0.039 | 0.038 |

## Test-patient sweep at k = 5 (for reporting only; not used for any choice)

| theta | sensitivity | fa_per_hour | median_delay_s |
|---|---|---|---|
| 0.05 | 0.848 | 11.514 | 2.5 |
| 0.1 | 0.727 | 4.521 | 7.0 |
| 0.15 | 0.667 | 2.774 | 8.5 |
| 0.2 | 0.606 | 2.054 | 12.0 |
| 0.25 | 0.576 | 1.362 | 14.0 |
| 0.3 | 0.545 | 0.815 | 24.0 |
| 0.35 | 0.515 | 0.536 | 28.0 |
| 0.4 | 0.455 | 0.335 | 30.0 |
| 0.45 | 0.394 | 0.234 | 36.0 |
| 0.5 | 0.364 | 0.156 | 37.0 |
| 0.55 | 0.303 | 0.078 | 38.0 |
| 0.6 | 0.303 | 0.017 | 41.0 |
| 0.65 | 0.212 | 0.0 | 30.0 |
| 0.7 | 0.152 | 0.006 | 33.0 |
| 0.75 | 0.121 | 0.0 | 42.5 |
| 0.8 | 0.091 | 0.0 | 34.0 |
| 0.85 | 0.03 | 0.0 | 58.0 |
| 0.9 | 0.03 | 0.0 | 64.0 |
| 0.95 | 0.0 | 0.0 | nan |

## Importance (impurity, summed)

By feature: line_length 0.370, std 0.267, rel_beta 0.162, rel_theta 0.084, rel_delta 0.064, rel_alpha 0.054

By channel: T7-P7 0.125, FZ-CZ 0.090, P7-O1 0.083, CZ-PZ 0.072, F4-C4 0.065, T8-P8 0.063, P8-O2 0.061, P3-O1 0.057, C4-P4 0.053, F7-T7 0.047, F3-C3 0.041, C3-P3 0.040, FP2-F4 0.040, P4-O2 0.037, F8-T8 0.036, FP2-F8 0.031, FP1-F3 0.030, FP1-F7 0.029

Runtime: fit 33.1 s, total 36.9 s. Figure: figures/rf_all18_sweep.png
