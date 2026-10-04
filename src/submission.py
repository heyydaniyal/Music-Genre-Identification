"""Kaggle submission writing and logging (keeps notebook and Kaggle scores consistent).

Planned API:
    write_submission(ids, preds, version)  -> Path  # submissions/GroupXX_VersionNN.csv, validated against sample_submission ids
    file_sha256(path)                      -> str
    log_submission(version, path, **fields)         # appends to results/submission_log.csv
"""
