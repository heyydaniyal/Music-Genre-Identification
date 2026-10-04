"""Hand-built features as scikit-learn transformers, so they slot into pipelines.

Planned API:
    StructuralFeatures      # line count, line length, repetition ratio, type-token ratio, tag counts, punctuation ratios ...
    MetadataFeatures        # cleaned year bins, log views, collaborator flag/count
    OutOfFoldTargetEncoder  # artist / collaborator encoding fitted inside CV folds only (prevents leakage)
"""
