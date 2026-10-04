"""Model pipelines, cross-validation and macro-F1 threshold tuning.

Planned API:
    make_pipeline(representation, model)            # vectorizer inside the pipeline, so CV has no leakage
    cross_validate_model(pipe, X, y, n_splits)      # stratified CV -> mean/std macro F1 + out-of-fold predictions
    tune_class_biases(oof_scores, y)                # per-class offsets that maximize macro F1
    fit_final(pipe, X, y)                           # refit on the full training set with the chosen parameters
"""
