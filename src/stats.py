"""Statistics for the sentiment analysis: effect sizes, not just p-values (n = 135k).

Planned API:
    kruskal_with_effect(groups)       # H, p, epsilon squared
    cliffs_delta(a, b)
    bootstrap_ci(data, stat, n_boot)  # percentile confidence interval
    cohens_kappa(a, b)                # annotator agreement
"""
