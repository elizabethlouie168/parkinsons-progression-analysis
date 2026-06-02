"""Project-wide configuration for Parkinson disease ML validation."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"

TARGET_COLUMN = "total_UPDRS"
PATIENT_ID_COLUMN = "subject#"

RANDOM_STATE = 42
N_SPLITS = 5

VOICE_FEATURES = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]

DEMOGRAPHIC_FEATURES = ["age", "sex", "test_time"]

LEAKAGE_PRONE_FEATURES = ["motor_UPDRS"]
