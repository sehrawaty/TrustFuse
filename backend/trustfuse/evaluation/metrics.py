"""Dependency-light classification and detection metrics (NumPy only, so the API stays lean)."""

from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y, dtype=np.int64), np.asarray(p, dtype=np.int64)), 1)
    return cm


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    return float(np.mean(y == np.asarray(p))) if y.size else float("nan")


def macro_f1(y: np.ndarray, p: np.ndarray, k: int, present_only: bool = True) -> float:
    """Macro-F1. With ``present_only`` the average runs over classes that occur in ``y``
    (the convention for per-subject scores where a subject may lack a class)."""
    if len(y) == 0:
        return float("nan")
    cm = confusion(y, p, k)
    tp = np.diag(cm).astype(float)
    support = cm.sum(1)
    predicted = cm.sum(0)
    denom = 2 * tp + (support - tp) + (predicted - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    classes = support > 0 if present_only else np.ones(k, bool)
    return float(f1[classes].mean()) if classes.any() else float("nan")


def balanced_accuracy(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return float("nan")
    cm = confusion(y, p, k)
    support = cm.sum(1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(k), where=support > 0)
    return float(recall[support > 0].mean())


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(ece)


def auroc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Probability that a positive (attacked) score exceeds a negative (clean) one (ties = 1/2)."""
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inv]  # 1-based average ranks (ties share)
    r_pos = ranks[neg.size:].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def roc_curve(neg: np.ndarray, pos: np.ndarray, max_points: int = 101) -> dict:
    neg = np.asarray(neg, dtype=float)
    pos = np.asarray(pos, dtype=float)
    if neg.size == 0 or pos.size == 0:
        return {"fpr": [], "tpr": []}
    thresholds = np.unique(np.concatenate([neg, pos]))[::-1]
    if thresholds.size > max_points:
        thresholds = np.quantile(np.concatenate([neg, pos]), np.linspace(1, 0, max_points))
    fpr = [0.0] + [float((neg >= t).mean()) for t in thresholds] + [1.0]
    tpr = [0.0] + [float((pos >= t).mean()) for t in thresholds] + [1.0]
    return {"fpr": fpr, "tpr": tpr}


def entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in bits (bounded in [0, 1])."""
    m = 0.5 * (p + q)
    return (entropy(m) - 0.5 * entropy(p) - 0.5 * entropy(q)) / np.log(2.0)
