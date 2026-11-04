"""Writing and logging Kaggle submission files.

The final cell of notebook 03 uses these functions, so the file uploaded to Kaggle is always the one the notebook
regenerates, and its SHA-256 hash proves it (same hash = byte-for-byte the same file).
"""
import hashlib
from datetime import date

import pandas as pd

from src import config


def write_submission(ids, preds, version: int):
    """Validate the predictions against sample_submission.csv and write submissions/<GROUP_ID>_VersionNN.csv."""
    sample = pd.read_csv(config.SAMPLE_SUBMISSION_CSV, encoding=config.ENCODING)
    sub = pd.DataFrame({config.ID_COL: list(ids), config.TARGET_COL: list(preds)})

    if len(sub) != len(sample):
        raise ValueError(f"{len(sub)} predictions, but Kaggle expects {len(sample)} rows.")
    if sub[config.ID_COL].duplicated().any():
        raise ValueError("Duplicated ids in the predictions.")
    if set(sub[config.ID_COL]) != set(sample[config.ID_COL]):
        raise ValueError("The ids do not match the ids in sample_submission.csv.")
    unknown = set(sub[config.TARGET_COL]) - set(config.LABELS)
    if unknown:
        raise ValueError(f"Unknown genre labels: {sorted(unknown)}")

    # same row order as Kaggle's sample file
    sub = sample[[config.ID_COL]].merge(sub, on=config.ID_COL, how="left")

    path = config.SUBMISSIONS_DIR / f"{config.GROUP_ID}_Version{version:02d}.csv"
    sub.to_csv(path, index=False, encoding=config.ENCODING)
    return path


def file_sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def log_submission(path, cv_macro_f1: float, notes: str = "", exp_id: str = "") -> pd.DataFrame:
    """Add the file to results/submission_log.csv (one row per file name, so re-running the notebook updates the row).

    The public leaderboard score is filled in by hand after uploading to Kaggle; an existing value is kept.
    """
    log = pd.read_csv(config.SUBMISSION_LOG, encoding=config.ENCODING)
    name = path.name
    old = log.loc[log["file_name"] == name, "public_lb_macro_f1"]
    row = {
        "version": name.rsplit("_Version", 1)[1].removesuffix(".csv"),
        "file_name": name,
        "date": date.today().isoformat(),
        "exp_id": exp_id,
        "sha256": file_sha256(path),
        "cv_macro_f1": round(cv_macro_f1, 4),
        "public_lb_macro_f1": old.iloc[0] if len(old) else None,
        "notes": notes,
    }
    log = pd.DataFrame(log[log["file_name"] != name].to_dict("records") + [row], columns=log.columns)
    log.to_csv(config.SUBMISSION_LOG, index=False, encoding=config.ENCODING)
    return log
