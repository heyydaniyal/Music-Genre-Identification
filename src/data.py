"""Reading the files that notebooks pass to each other.

Notebook 02 writes data/processed/train_clean.pkl and test_clean.pkl; notebooks 03 and 04 read them.
"""
import pandas as pd

from src import config


def load_processed(name: str) -> pd.DataFrame:
    """Load data/processed/<name>.pkl, or stop with a message that says which notebook creates it."""
    path = config.PROCESSED_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} was not found in data/processed/. Run notebook 02_preprocessing first."
        )
    return pd.read_pickle(path)
