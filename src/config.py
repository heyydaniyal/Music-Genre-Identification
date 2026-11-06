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

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DIR / "sample_submission.csv"

EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.csv"
SUBMISSION_LOG = RESULTS_DIR / "submission_log.csv"
SENTIMENT_FINDINGS = RESULTS_DIR / "sentiment_findings.csv"   # written by 04, read by 05

ENCODING = "utf-8"

# --- Columns ---
ID_COL = "id"
TEXT_COL = "lyrics"
TARGET_COL = "tag"
META_COLS = ["title", "artist", "year", "views", "features"]

# --- Labels ---
LABELS = ["country", "misc", "pop", "rap", "rb", "rock"]             # sorted: confusion matrices, reports
GENRE_ORDER = ["pop", "rap", "rock", "rb", "misc", "country"]       # by frequency: plots and tables
MUSIC_GENRES = ["pop", "rap", "rock", "rb", "country"]              # misc is not music (decision D10)

# --- Reproducibility and validation ---
SEED = 42
N_SPLITS = 5

# --- Evaluation ---
PRIMARY_METRIC = "f1_macro"     # Kaggle metric (confirmed)
SECONDARY_METRIC = "f1_weighted"

# --- Sentiment (notebook 04) ---
VADER_POS = 0.05    # Hutto & Gilbert (2014) thresholds on the compound score
VADER_NEG = -0.05

# --- Kaggle ---
GROUP_ID = "Group15"
