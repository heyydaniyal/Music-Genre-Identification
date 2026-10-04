"""Common setup run by every notebook via `%run ./_setup.py` (written once, not copied into each notebook).

Makes the repo root importable, fixes seeds and sets display options.
"""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

random.seed(config.SEED)
np.random.seed(config.SEED)

pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 200)

print(f"Repo root: {ROOT} | seed: {config.SEED}")
