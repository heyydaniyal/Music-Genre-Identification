"""Single source of truth for paths, column names, labels and global settings.

No other file hard-codes a path, a column name, a label list or a seed:
everything imports it from here.
"""
from pathlib import Path

# --- Paths (all relative to the repository root, so it runs on any machine) ---
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.csv"
SUBMISSION_LOG = RESULTS_DIR / "submission_log.csv"

ENCODING = "utf-8"

# --- Columns ---
ID_COL = "id"
TEXT_COL = "lyrics"
TARGET_COL = "tag"
META_COLS = ["title", "artist", "year", "views", "features"]

# --- Labels (sorted; fixed order for confusion matrices and reports) ---
LABELS = ["country", "misc", "pop", "rap", "rb", "rock"]

# --- Reproducibility and validation ---
SEED = 42
N_SPLITS = 5            # stratified K-fold for final numbers
N_SPLITS_SCREEN = 3     # quick screening runs
SCREEN_SAMPLE_SIZE = 40_000

# --- Evaluation ---
PRIMARY_METRIC = "f1_macro"     # Kaggle metric (confirmed)
SECONDARY_METRIC = "f1_weighted"

# --- Kaggle ---
GROUP_ID = "GroupXX"            # set your group number once, e.g. "Group07"
