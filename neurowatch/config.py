"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HEADERS = ROOT / "data" / "headers"
CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

SEED = 42
FS = 256  # Hz, CHB-MIT sampling rate
CNN_FS = 128  # Hz, decimated rate cached for the CNN (after 40 Hz low-pass)

# 18 bipolar channels of the longitudinal ("double banana") montage
CHANNELS_18 = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]

EXCLUDED_CASES = ["chb24"]
# chb21 is the same subject as chb01, recorded 1.5 years later
PATIENT_OF_CASE = {f"chb{i:02d}": f"chb{i:02d}" for i in range(1, 25)}
PATIENT_OF_CASE["chb21"] = "chb01"

WIN_S = 4.0
STEP_S = 2.0
BAND = (0.5, 40.0)
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
FEATURE_NAMES = ["std", "line_length", "rel_delta", "rel_theta", "rel_alpha", "rel_beta"]
