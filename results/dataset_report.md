# Dataset summary (computed by scripts/dataset_summary.py)

- Cases in database: 24 (chb01–chb24); files in RECORDS: 686; total hours: 982.94; total annotated seizures: 198
- Excluding chb24: 23 cases, 22 patients, 664 files, 961.64 h, 182 seizures; ages 1.5–22 y (18 F / 5 M cases)
- chb24: 22 files, 16 seizures; age in SUBJECT-INFO: nan
- Sampling rate: all files [256.0] Hz
- After channel check (used for modelling): 661 files, 958.63 h, 169 seizures, 22 patients

## Reference channel set

```json
{
  "candidate": "all 18 bipolar channels",
  "channels": [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ"
  ],
  "recordings_considered": 664,
  "recordings_with_all_18": 661,
  "hours_considered": 961.64,
  "hours_with_all_18": 958.63,
  "seizures_considered": 182,
  "seizures_with_all_18": 169,
  "recordings_with_some_but_not_all_18": 0,
  "recordings_with_none_of_18": 3,
  "decision_rule": "use all 18 if >=95% of recordings and >=90% of seizures keep them",
  "decision": "all 18"
}
```

## Excluded recordings (missing reference channels)

- chb12/chb12_27.edf: 6 seizures, 1.00 h, 18 of 18 channels missing
- chb12/chb12_28.edf: 1 seizures, 1.00 h, 18 of 18 channels missing
- chb12/chb12_29.edf: 6 seizures, 1.01 h, 18 of 18 channels missing

## Metadata inconsistencies found

- Files in RECORDS but not listed in any summary file: ['chb24_02.edf', 'chb24_05.edf', 'chb24_08.edf', 'chb24_10.edf', 'chb24_12.edf', 'chb24_16.edf', 'chb24_18.edf', 'chb24_19.edf', 'chb24_20.edf', 'chb24_22.edf']
- Files in summaries but not in RECORDS: []
- Files with seizures per summary but not in RECORDS-WITH-SEIZURES: ['chb07_19.edf']
- Files in RECORDS-WITH-SEIZURES with no seizure in summary: ['chb07_18.edf']
- Seizure annotations extending past end of file: []
- Seizure durations (excl. chb24): min 6 s, median 48 s, max 752 s

Sample trace: chb01/chb01_03.edf
