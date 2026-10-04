"""Run every notebook top to bottom, in order, exactly as the graders will (QA, roadmap P8).

Usage (from the repo root):  python scripts/run_all.py [--fresh]
  --fresh  deletes data/processed/* first, to prove no stale intermediate file is needed.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

NOTEBOOK_DIR = ROOT / "notebooks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    if args.fresh:
        for f in config.PROCESSED_DIR.glob("*"):
            if f.name != ".gitkeep":
                f.unlink() if f.is_file() else shutil.rmtree(f)

    for path in sorted(NOTEBOOK_DIR.glob("[0-9][0-9]_*.ipynb")):
        start = time.time()
        nb = nbformat.read(path, as_version=4)
        ExecutePreprocessor(timeout=None, kernel_name="python3").preprocess(
            nb, {"metadata": {"path": str(NOTEBOOK_DIR)}}
        )
        nbformat.write(nb, path)
        print(f"{path.name:<35} OK  {time.time() - start:7.1f}s")


if __name__ == "__main__":
    main()
