"""Metrics and reporting (identical everywhere, so numbers are comparable).

Planned API:
    score(y_true, y_pred)                    -> dict  # macro F1, weighted F1, accuracy
    per_class_report(y_true, y_pred)         -> DataFrame
    plot_confusion(y_true, y_pred, ax=None)
    log_experiment(**fields)                 # appends a row to results/experiment_log.csv
"""
