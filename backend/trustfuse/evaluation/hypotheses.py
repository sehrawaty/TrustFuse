"""Falsifiable hypotheses H1-H4 (pre-registered in ExperimentConfig.stats) and secondary analyses.

Input: a subject-level table (one row per subject x condition for the primary model, metrics
averaged over seeds). All tests are paired across subjects.

H1  attack effectiveness   full suppression of modality m lowers fixed-fusion macro-F1
H2  modality sensitivity   the degradation differs between modalities (Friedman)
H3  mitigation             reliability-aware fusion beats fixed fusion under the same attack, and
                           its clean macro-F1 loss is within the non-inferiority margin
H4  detectability          suppression is detectable from reliability evidence (AUROC > 0.5) with
                           a clean false-rejection rate within the pre-registered limit
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ..experiments.config import ExperimentConfig
from .stats import (bootstrap_ci, cohen_dz, friedman, holm, one_sample_greater, rank_biserial,
                    wilcoxon_greater)


def _by_subject(df: pd.DataFrame, col: str) -> pd.Series:
    return df.set_index("subject")[col].sort_index()


def _align(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    idx = a.index.intersection(b.index)
    return a.loc[idx].to_numpy(float), b.loc[idx].to_numpy(float)


def primary_rows(subj: pd.DataFrame, cfg: ExperimentConfig, modality: str, severity: float | None = None) -> pd.DataFrame:
    sev = cfg.stats.primary_severity if severity is None else severity
    return subj[(subj.family == "mask") & (subj.fill == cfg.stats.primary_fill)
                & np.isclose(subj.severity.astype(float), sev) & (subj.modality == modality)]


def svi(subj: pd.DataFrame, cfg: ExperimentConfig, modality: str, metric: str = "fixed_f1") -> float:
    """Suppression Vulnerability Index: normalised area under the degradation-severity curve
    (mean macro-F1 drop integrated over severity in [0, 1] with zero drop at severity 0)."""
    clean = subj[subj.family == "clean"][metric].mean()
    sevs, drops = [0.0], [0.0]
    for sev in sorted(cfg.attacks.severities):
        rows = primary_rows(subj, cfg, modality, sev)
        if len(rows):
            sevs.append(sev)
            drops.append(clean - rows[metric].mean())
    if len(sevs) < 2:
        return float("nan")
    area = np.trapezoid(drops, sevs) if hasattr(np, "trapezoid") else np.trapz(drops, sevs)
    return float(area / (sevs[-1] - sevs[0]))


def test_hypotheses(subj: pd.DataFrame, cfg: ExperimentConfig, modalities: tuple[str, ...]) -> dict:
    S = cfg.stats
    alpha = S.alpha
    boot = dict(n_boot=S.n_boot, seed=S.seed)
    clean = subj[subj.family == "clean"]
    c_fixed, c_def = _by_subject(clean, "fixed_f1"), _by_subject(clean, "def_f1")
    out: dict = {"alpha": alpha, "primary_condition": f"full masking ({S.primary_fill}-fill, severity {S.primary_severity:g})"}

    # ---------------- H1: attack effectiveness ----------------
    h1 = []
    for m in modalities:
        a, b = _align(c_fixed, _by_subject(primary_rows(subj, cfg, m), "fixed_f1"))
        t = wilcoxon_greater(a, b)
        h1.append({"modality": m, "clean_f1": bootstrap_ci(a, **boot), "attacked_f1": bootstrap_ci(b, **boot),
                   "drop": bootstrap_ci(a - b, **boot), "p": t["p"], "n": t["n"],
                   "rank_biserial": rank_biserial(a, b), "cohen_dz": cohen_dz(a, b)})
    for r, p_adj in zip(h1, holm([r["p"] for r in h1])):
        r["p_holm"] = p_adj
        r["significant"] = bool(p_adj < alpha)
    sig1 = [r["modality"] for r in h1 if r["significant"]]
    out["H1"] = {
        "title": "Attack effectiveness",
        "statement": "Full suppression of a single modality significantly reduces fixed-fusion macro-F1.",
        "test": "one-sided paired Wilcoxon signed-rank (clean > attacked) per modality, Holm-corrected",
        "per_modality": h1, "supported_modalities": sig1, "supported": bool(sig1),
    }

    # ---------------- H2: modality sensitivity ----------------
    drops = {}
    for m in modalities:
        a, b = _align(c_fixed, _by_subject(primary_rows(subj, cfg, m), "fixed_f1"))
        drops[m] = pd.Series(a - b, index=c_fixed.index.intersection(_by_subject(primary_rows(subj, cfg, m), "fixed_f1").index))
    mtx = pd.DataFrame(drops).dropna()
    fr = friedman(mtx.to_numpy()) if len(modalities) > 2 else {}
    pairs = []
    if len(modalities) == 2:
        x, y = mtx[modalities[0]].to_numpy(), mtx[modalities[1]].to_numpy()
        p2 = float(wilcoxon(x, y).pvalue) if len(x) > 1 and not np.allclose(x, y) else 1.0
        fr = {"stat": None, "p": p2, "kendall_w": None, "n": int(len(x)), "test": "two-sided Wilcoxon (2 modalities)"}
    else:
        raw = []
        for i, m1 in enumerate(modalities):
            for m2 in modalities[i + 1:]:
                x, y = mtx[m1].to_numpy(), mtx[m2].to_numpy()
                p = float(wilcoxon(x, y).pvalue) if len(x) > 1 and not np.allclose(x, y) else 1.0
                raw.append((m1, m2, p, float(np.mean(x - y))))
        for (m1, m2, p, d), pa in zip(raw, holm([r[2] for r in raw])):
            pairs.append({"a": m1, "b": m2, "mean_diff": d, "p": p, "p_holm": pa})
    ranking = sorted(({"modality": m, "mean_drop": float(mtx[m].mean()), "svi": svi(subj, cfg, m)} for m in modalities),
                     key=lambda r: -r["mean_drop"])
    out["H2"] = {
        "title": "Modality sensitivity",
        "statement": "Modalities differ in their vulnerability to targeted suppression.",
        "test": "Friedman test on per-subject macro-F1 drops across modalities (Kendall's W), Holm post-hoc",
        "friedman": fr, "pairwise": pairs, "ranking": ranking,
        "supported": bool(fr["p"] is not None and fr["p"] < alpha),
    }

    # ---------------- H3: mitigation ----------------
    a, b = _align(c_fixed, c_def)
    cost = bootstrap_ci(a - b, **boot)
    noninf = cost["hi"] is not None and cost["hi"] <= S.noninferiority_margin
    h3 = []
    for m in modalities:
        rows = primary_rows(subj, cfg, m)
        d_, f_ = _align(_by_subject(rows, "def_f1"), _by_subject(rows, "fixed_f1"))
        t = wilcoxon_greater(d_, f_)
        h3.append({"modality": m, "fixed_f1": bootstrap_ci(f_, **boot), "def_f1": bootstrap_ci(d_, **boot),
                   "gain": bootstrap_ci(d_ - f_, **boot), "p": t["p"], "rank_biserial": rank_biserial(d_, f_),
                   "cohen_dz": cohen_dz(d_, f_)})
    for r, p_adj in zip(h3, holm([r["p"] for r in h3])):
        r["p_holm"] = p_adj
        r["significant"] = bool(p_adj < alpha)
    mitigated = [r["modality"] for r in h3 if r["significant"] and r["modality"] in sig1]
    attacked = subj[subj.family.isin(["mask", "attenuate", "noise"])]
    overall_gain = bootstrap_ci((attacked.groupby("subject").def_f1.mean() - attacked.groupby("subject").fixed_f1.mean()).to_numpy(), **boot)
    out["H3"] = {
        "title": "Mitigation without material clean loss",
        "statement": ("Reliability-aware fusion improves macro-F1 over fixed fusion under the same attack, while its "
                      f"clean macro-F1 loss stays within the non-inferiority margin ({S.noninferiority_margin:g})."),
        "test": "one-sided paired Wilcoxon (defended > fixed) per modality, Holm-corrected; bootstrap non-inferiority",
        "clean_cost": cost, "noninferior": bool(noninf), "margin": S.noninferiority_margin,
        "per_modality": h3, "mitigated_modalities": mitigated, "mean_gain_all_attacks": overall_gain,
        "supported": bool(noninf and mitigated),
    }

    # ---------------- H4: detectability ----------------
    frr = bootstrap_ci(_by_subject(clean, "flag_rate").to_numpy(), **boot)
    h4 = []
    for m in modalities:
        au = _by_subject(primary_rows(subj, cfg, m), "auroc").to_numpy(float)
        t = one_sample_greater(au, 0.5)
        h4.append({"modality": m, "auroc": bootstrap_ci(au, **boot), "p": t["p"],
                   "tpr": bootstrap_ci(_by_subject(primary_rows(subj, cfg, m), "flag_rate").to_numpy(), **boot)})
    for r, p_adj in zip(h4, holm([r["p"] for r in h4])):
        r["p_holm"] = p_adj
        r["significant"] = bool(p_adj < alpha)
    detectable = [r["modality"] for r in h4 if r["significant"]]
    frr_ok = frr["mean"] is not None and frr["mean"] <= S.frr_max
    out["H4"] = {
        "title": "Detectability of suppression",
        "statement": ("Suppression produces a reliability signature that separates attacked from clean windows "
                      f"(AUROC > 0.5) at a clean false-rejection rate no higher than {S.frr_max:g}."),
        "test": "one-sided one-sample Wilcoxon of per-subject AUROC against 0.5, Holm-corrected; bootstrap FRR",
        "clean_frr": frr, "frr_max": S.frr_max, "frr_ok": bool(frr_ok), "per_modality": h4,
        "detectable_modalities": detectable, "supported": bool(frr_ok and len(detectable) == len(modalities)),
    }

    # ---------------- secondary: strategic vs. random suppression ----------------
    sec = {}
    fam = {f: subj[subj.family == f] for f in ("random_drop", "targeted", "oracle", "adaptive")}
    if len(fam["random_drop"]) and len(fam["targeted"]):
        r_, t_ = _by_subject(fam["random_drop"], "fixed_f1"), _by_subject(fam["targeted"], "fixed_f1")
        a, b = _align(r_, t_)
        test = wilcoxon_greater(a, b)
        sec["targeted_vs_random"] = {
            "statement": "Strategically chosen suppression degrades fixed fusion more than random modality loss.",
            "random_f1": bootstrap_ci(a, **boot), "targeted_f1": bootstrap_ci(b, **boot),
            "difference": bootstrap_ci(a - b, **boot), "p": test["p"], "rank_biserial": rank_biserial(a, b)}
    attackers = []
    for f, rows in fam.items():
        if len(rows):
            attackers.append({"attacker": f, "fixed_f1": bootstrap_ci(rows.fixed_f1, **boot),
                              "def_f1": bootstrap_ci(rows.def_f1, **boot),
                              "flag_rate": bootstrap_ci(rows.flag_rate, **boot),
                              "evasive_def_flip": bootstrap_ci(rows.evasive_def_flip, **boot),
                              "fixed_asr": bootstrap_ci(rows.fixed_asr, **boot),
                              "def_asr": bootstrap_ci(rows.def_asr, **boot)})
    sec["attackers"] = attackers
    out["secondary"] = sec
    return out
