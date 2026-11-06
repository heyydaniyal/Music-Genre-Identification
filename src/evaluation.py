"""Experiment logging, so the final numbers of each notebook live in one CSV that notebook 05 can read."""
from datetime import date

import pandas as pd

from src import config


def log_experiment(exp_id: str, **fields) -> pd.DataFrame:
    """Write one row to results/experiment_log.csv. Re-running a notebook replaces its row instead of duplicating it."""
    log = pd.read_csv(config.EXPERIMENT_LOG, encoding=config.ENCODING)
    unknown = set(fields) - set(log.columns)
    if unknown:
        raise ValueError(f"Unknown experiment-log columns: {sorted(unknown)}")

    row = {"exp_id": exp_id, "date": date.today().isoformat(), **fields}
    log = pd.DataFrame(log[log["exp_id"] != exp_id].to_dict("records") + [row], columns=log.columns)
    log.to_csv(config.EXPERIMENT_LOG, index=False, encoding=config.ENCODING)
    return log
