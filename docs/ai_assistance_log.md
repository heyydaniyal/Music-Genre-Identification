# AI assistance log

What Claude drafted, what Dani reviewed or changed, and how it was verified. Dani fills in the "reviewed/changed" column.

| Date | Phase | What Claude drafted | What Dani reviewed / changed | Verified how |
|---|---|---|---|---|
| 2026-11-04 | src | `config`, `data`, `features`, `sentiment`, `stats`, `submission`, `evaluation`, `viz`; tests in `tests/`; removed unused stubs | | `python -m pytest -q` (20 passed) |
| 2026-11-04 | 03 | Section 11 (submission from the evaluated model, refit-on-all kept commented with reason, logging); memory fixes (load 3 columns, `del` unused matrices, `src/features.hstack_csr`); commented stopword/section-token check; header fixes | | Run on the real data under a 6 GB memory limit in a Python 3.11 env pinned to `requirements.txt`; holdout numbers identical to the teammates' outputs |
| 2026-11-04 | 04 | Whole notebook: method checks, SQ1–SQ5, findings file | | Run on the real data; numbers in markdown checked against printed outputs |
| 2026-11-04 | 05 | Conclusion, references (existence checked), annexes | | Run after 03 and 04 |
| 2026-11-04 | 01/02 | Abstract rewritten with final results; notebook numbering and paths | | Full `run_all.py --fresh` |
