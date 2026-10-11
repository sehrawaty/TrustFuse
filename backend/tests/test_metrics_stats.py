import numpy as np
from scipy import stats as st
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from trustfuse.evaluation.metrics import auroc, balanced_accuracy, js_divergence, macro_f1
from trustfuse.evaluation.stats import bootstrap_ci, friedman, holm, mcnemar_exact, wilcoxon_greater


def test_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 500)
    p = np.where(rng.random(500) < 0.7, y, rng.integers(0, 3, 500))
    assert np.isclose(macro_f1(y, p, 3), f1_score(y, p, average="macro"))
    assert np.isclose(balanced_accuracy(y, p, 3), balanced_accuracy_score(y, p))


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    neg = np.round(rng.normal(0, 1, 300), 1)
    pos = np.round(rng.normal(0.8, 1, 200), 1)
    ref = roc_auc_score(np.r_[np.zeros(300), np.ones(200)], np.r_[neg, pos])
    assert np.isclose(auroc(neg, pos), ref)


def test_js_bounds():
    p = np.array([[1.0, 0.0], [0.5, 0.5]])
    q = np.array([[0.0, 1.0], [0.5, 0.5]])
    d = js_divergence(p, q)
    assert np.isclose(d[0], 1.0, atol=1e-6) and np.isclose(d[1], 0.0, atol=1e-9)


def test_stats():
    a = np.array([0.9, 0.8, 0.95, 0.85, 0.9, 0.7, 0.88, 0.92])
    b = a - np.array([0.1, 0.2, 0.05, 0.15, 0.1, 0.3, 0.12, 0.08])
    t = wilcoxon_greater(a, b)
    assert np.isclose(t["p"], st.wilcoxon(a, b, alternative="greater").pvalue)
    ci = bootstrap_ci(a - b, n_boot=500)
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    fr = friedman(np.c_[a, b, b - 0.05])
    assert fr["p"] < 0.05 and 0 <= fr["kendall_w"] <= 1
    assert mcnemar_exact(np.ones(10, bool), np.zeros(10, bool))["a_only"] == 10
