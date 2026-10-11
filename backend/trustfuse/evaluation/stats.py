"""Statistics for subject-independent evaluation.

The unit of analysis is the *subject* (LOSO fold): window-level observations of one subject are
not independent, so confidence intervals use a subject-cluster bootstrap and hypothesis tests
are paired across subjects.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    v = _clean(values)
    if v.size == 0:
        return {"mean": None, "lo": None, "hi": None, "sd": None, "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "lo": float(v[0]), "hi": float(v[0]), "sd": 0.0, "n": 1}
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi), "sd": float(v.std(ddof=1)), "n": int(v.size)}


def paired(a, b) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    return a[ok], b[ok]


def wilcoxon_greater(a, b) -> dict:
    """One-sided paired Wilcoxon signed-rank test of H1: a > b (across subjects)."""
    a, b = paired(a, b)
    d = a - b
    if d.size < 2 or np.allclose(d, 0):
        return {"stat": None, "p": 1.0, "n": int(d.size)}
    res = st.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n": int(d.size)}


def one_sample_greater(x, mu: float) -> dict:
    x = _clean(x)
    if x.size < 2 or np.allclose(x - mu, 0):
        return {"stat": None, "p": 1.0, "n": int(x.size)}
    res = st.wilcoxon(x - mu, alternative="greater", zero_method="wilcox")
    return {"stat": float(res.statistic), "p": float(res.pvalue), "n": int(x.size)}


def rank_biserial(a, b) -> float | None:
    """Matched-pairs rank-biserial correlation (effect size for the Wilcoxon test)."""
    a, b = paired(a, b)
    d = a - b
    d = d[d != 0]
    if d.size == 0:
        return None
    r = st.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def cohen_dz(a, b) -> float | None:
    a, b = paired(a, b)
    d = a - b
    if d.size < 2 or d.std(ddof=1) == 0:
        return None
    return float(d.mean() / d.std(ddof=1))


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def friedman(matrix: np.ndarray) -> dict:
    """Friedman test over conditions (columns) with subjects as blocks (rows) + Kendall's W."""
    mtx = np.asarray(matrix, dtype=float)
    mtx = mtx[np.isfinite(mtx).all(axis=1)]
    n, k = mtx.shape
    if n < 2 or k < 2:
        return {"stat": None, "p": 1.0, "kendall_w": None, "n": int(n)}
    res = st.friedmanchisquare(*[mtx[:, j] for j in range(k)])
    return {"stat": float(res.statistic), "p": float(res.pvalue),
            "kendall_w": float(res.statistic / (n * (k - 1))), "n": int(n)}


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test on matched windows (discordant pairs only), two-sided."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(min(1.0, 2 * st.binom.cdf(min(n01, n10), n, 0.5)))
    return {"a_only": n01, "b_only": n10, "p": p}


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"
