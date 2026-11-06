"""Statistics for the sentiment analysis.

With tens of thousands of songs almost every difference is "significant", so every test here also returns an
effect size (how big the difference is) and the notebook interprets that, not the p-value.
Theory for each method is in the Annexes of notebook 05.
"""
import numpy as np
import pandas as pd
from scipy import stats


def kruskal_epsilon2(groups: dict) -> dict:
    """Kruskal-Wallis test across groups + epsilon squared effect size (share of rank variance explained by group)."""
    values = [np.asarray(v) for v in groups.values()]
    h, p = stats.kruskal(*values)
    n = sum(len(v) for v in values)
    return {"H": h, "p_value": p, "epsilon2": h / (n - 1)}


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b), from -1 to 1. Computed from the Mann-Whitney U statistic (ties count as half)."""
    a, b = np.asarray(a), np.asarray(b)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def delta_size(delta: float) -> str:
    """Verbal label for |Cliff's delta| (thresholds from Romano et al., 2006)."""
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    if d < 0.33:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the chance of any false positive across many tests)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def pairwise_mannwhitney(groups: dict) -> pd.DataFrame:
    """Every pair of groups: Mann-Whitney p-value (Holm-adjusted) and Cliff's delta (first group vs second)."""
    names = list(groups)
    rows = []
    for i, g1 in enumerate(names):
        for g2 in names[i + 1:]:
            p = stats.mannwhitneyu(groups[g1], groups[g2], alternative="two-sided").pvalue
            d = cliffs_delta(groups[g1], groups[g2])
            rows.append({"group_1": g1, "group_2": g2, "p_value": p, "cliffs_delta": d, "size": delta_size(d)})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p_value"])
    return out


def bootstrap_ci(values, statistic=np.median, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple:
    """Percentile bootstrap confidence interval: recompute the statistic on resamples drawn with replacement."""
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    tail = (1 - level) / 2 * 100
    return tuple(np.percentile(boots, [tail, 100 - tail]))


def spearman_ci(x, y, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Spearman correlation with a percentile bootstrap confidence interval (pairs are resampled together)."""
    x, y = np.asarray(x), np.asarray(y)
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        boots.append(stats.spearmanr(x[idx], y[idx])[0])
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(boots, [tail, 100 - tail])
    return {"rho": rho, "ci_low": low, "ci_high": high, "p_value": p, "n": len(x)}
