"""Paper-ready figures (Matplotlib) and LaTeX tables generated from a result set.

Colours follow the validated TrustFuse palette (shared with the web UI):
modalities BVP/ACC/EDA/TEMP = magenta/violet/aqua/yellow; fixed vs. defended fusion = orange vs. blue.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MOD_COLORS = {"bvp": "#e87ba4", "acc": "#6250d6", "eda": "#1baf7a", "temp": "#eda100"}
MOD_LABEL = {"bvp": "BVP", "acc": "ACC", "eda": "EDA", "temp": "TEMP"}
FIXED, DEFENDED = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#1d2433", "#6b7280", "#e5e7eb"
ATTACKER_LABEL = {"random_drop": "Random", "targeted": "Targeted", "oracle": "Oracle", "adaptive": "Adaptive"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": "#c3c7cf", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})


def _load(d: Path) -> tuple[dict, dict, dict]:
    return (json.loads((d / "summary.json").read_text()), json.loads((d / "conditions.json").read_text()),
            json.loads((d / "detection.json").read_text()))


def _conds(conditions: dict, variant: str = "fusion") -> list[dict]:
    return [c for c in conditions["conditions"] if c["variant"] == variant]


def _m(c: dict, key: str) -> tuple[float, float, float]:
    v = c.get(key) or {}
    return v.get("mean", np.nan), v.get("lo", np.nan), v.get("hi", np.nan)


def fig_severity(summary, conditions, fill="zero"):
    mods = summary["modalities"]
    conds = _conds(conditions)
    clean = next(c for c in conds if c["family"] == "clean")
    fig, axes = plt.subplots(1, len(mods), figsize=(1.9 * len(mods) + 0.4, 2.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, mods):
        rows = sorted([c for c in conds if c["family"] == "mask" and c["fill"] == fill and c["modality"] == m],
                      key=lambda c: c["severity"])
        x = [0.0] + [c["severity"] for c in rows]
        for key, color, label in (("fixed_f1", FIXED, "Fixed fusion"), ("def_f1", DEFENDED, "Reliability-aware")):
            mean = [_m(clean, key)[0]] + [_m(c, key)[0] for c in rows]
            lo = [_m(clean, key)[1]] + [_m(c, key)[1] for c in rows]
            hi = [_m(clean, key)[2]] + [_m(c, key)[2] for c in rows]
            ax.fill_between(np.array(x) * 100, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(np.array(x) * 100, mean, color=color, marker="o", markersize=3.5, label=label)
        ax.set_title(MOD_LABEL[m], color=INK)
        ax.set_xlabel("Suppressed share of window (%)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle(f"Severity sweep · {fill}-fill masking (subject-bootstrap 95% CI)", fontsize=9.5, color=INK, y=1.04)
    return fig


def fig_heatmap(summary, conditions, fill="zero"):
    mods = summary["modalities"]
    conds = _conds(conditions)
    clean = next(c for c in conds if c["family"] == "clean")
    sevs = sorted({c["severity"] for c in conds if c["family"] == "mask" and c["fill"] == fill})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 0.5 * len(mods) + 1.2))
    for ax, key, title in ((axes[0], "fixed_f1", "Fixed fusion"), (axes[1], "def_f1", "Reliability-aware fusion")):
        mtx = np.full((len(mods), len(sevs)), np.nan)
        for i, m in enumerate(mods):
            for j, s in enumerate(sevs):
                c = next((c for c in conds if c["family"] == "mask" and c["fill"] == fill and c["modality"] == m
                          and abs(c["severity"] - s) < 1e-9), None)
                if c:
                    mtx[i, j] = _m(clean, key)[0] - _m(c, key)[0]
        im = ax.imshow(mtx, cmap="Blues", vmin=0, vmax=max(0.05, np.nanmax(np.abs(mtx))), aspect="auto")
        ax.set_xticks(range(len(sevs)), [f"{s:.0%}" for s in sevs])
        ax.set_yticks(range(len(mods)), [MOD_LABEL[m] for m in mods])
        ax.grid(False)
        for i in range(len(mods)):
            for j in range(len(sevs)):
                if np.isfinite(mtx[i, j]):
                    ax.text(j, i, f"{mtx[i, j]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if mtx[i, j] > 0.6 * np.nanmax(mtx) else INK)
        ax.set_title(f"{title}: macro-F1 drop", color=INK)
        ax.set_xlabel("Severity")
    fig.colorbar(im, ax=axes, shrink=0.8, label="ΔF1 vs. clean")
    return fig


def fig_attackers(summary):
    att = summary["hypotheses"]["secondary"].get("attackers", [])
    if not att:
        return None
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    x = np.arange(len(att))
    w = 0.36
    for k, (key, color, label) in enumerate((("fixed_f1", FIXED, "Fixed fusion"), ("def_f1", DEFENDED, "Reliability-aware"))):
        means = [a[key]["mean"] for a in att]
        err = [[a[key]["mean"] - a[key]["lo"] for a in att], [a[key]["hi"] - a[key]["mean"] for a in att]]
        ax.bar(x + (k - 0.5) * w, means, w * 0.92, color=color, label=label, yerr=err, error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(x, [ATTACKER_LABEL.get(a["attacker"], a["attacker"]) for a in att])
    ax.set_ylabel("Macro-F1 under attack")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_title("Attacker capability (one modality suppressed per window)", color=INK)
    return fig


def fig_roc(summary, detection):
    mods = summary["modalities"]
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], color="#c3c7cf", lw=1, ls="--")
    for m in mods:
        key = next((k for k in detection["curves"] if k.startswith("mask|") and f"modality={m}|" in k
                    and "fill=zero" in k and k.endswith("severity=1")), None)
        if key:
            c = detection["curves"][key]
            ax.plot(c["fpr"], c["tpr"], color=MOD_COLORS[m], label=f"{MOD_LABEL[m]} (AUROC {c['auroc']:.2f})")
    ax.set_xlabel("False-rejection rate (clean)")
    ax.set_ylabel("Detection rate (suppressed)")
    ax.set_title("Suppression detection · full masking", color=INK)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    return fig


def fig_forest(summary):
    rows = summary.get("forest") or []
    if not rows:
        return None
    mods = summary["modalities"]
    subjects = sorted({r["subject"] for r in rows}, key=lambda s: int(s[1:]))
    fig, ax = plt.subplots(figsize=(4.4, 0.22 * len(subjects) + 1.0))
    for k, m in enumerate(mods):
        pts = {r["subject"]: r["gain"] for r in rows if r["modality"] == m}
        y = [i + (k - (len(mods) - 1) / 2) * 0.16 for i in range(len(subjects))]
        ax.scatter([pts.get(s, np.nan) for s in subjects], y, s=14, color=MOD_COLORS[m], label=MOD_LABEL[m],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.axvline(0, color="#9ca3af", lw=1)
    ax.set_yticks(range(len(subjects)), subjects)
    ax.invert_yaxis()
    ax.set_xlabel("Defense gain in macro-F1 (defended − fixed)")
    ax.set_title("Per-subject defense gain · full masking", color=INK)
    ax.legend(fontsize=7, ncol=len(mods), loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / max(len(subjects), 1)))
    return fig


def fig_bursts(summary, conditions):
    mods = summary["modalities"]
    conds = [c for c in _conds(conditions) if c["family"] == "burst" and c.get("scope") == "full"]
    if not conds:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for m in mods:
        rows = sorted([c for c in conds if c["modality"] == m], key=lambda c: c["burst"])
        b = [c["burst"] for c in rows]
        axes[0].plot(b, [1 - _m(c, "fixed_acc")[0] for c in rows], color=MOD_COLORS[m], ls="--", marker="o", ms=3)
        axes[0].plot(b, [1 - _m(c, "def_acc")[0] for c in rows], color=MOD_COLORS[m], marker="o", ms=3, label=MOD_LABEL[m])
        axes[1].plot(b, [_m(c, "flag_rate")[0] for c in rows], color=MOD_COLORS[m], marker="o", ms=3, label=MOD_LABEL[m])
    axes[0].set_xlabel("Burst length (windows)")
    axes[0].set_ylabel("Error rate in burst")
    axes[0].set_title("Errors (dashed = fixed, solid = defended)", color=INK)
    axes[1].set_xlabel("Burst length (windows)")
    axes[1].set_ylabel("Share of burst windows flagged")
    axes[1].set_title("Detection during bursts", color=INK)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10], ["1", "2", "5", "10"])
    return fig


FIGURES = {
    "fig_severity_zero": lambda s, c, d: fig_severity(s, c, "zero"),
    "fig_severity_hold": lambda s, c, d: fig_severity(s, c, "hold"),
    "fig_vulnerability_heatmap": lambda s, c, d: fig_heatmap(s, c, "zero"),
    "fig_attackers": lambda s, c, d: fig_attackers(s),
    "fig_detection_roc": lambda s, c, d: fig_roc(s, d),
    "fig_forest_defense_gain": lambda s, c, d: fig_forest(s),
    "fig_bursts": lambda s, c, d: fig_bursts(s, c),
}


def render_all(d: Path, out_dir: Path, fmt: str = "pdf") -> list[Path]:
    summary, conditions, detection = _load(d)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        try:
            fig = fn(summary, conditions, detection)
        except (StopIteration, KeyError, ValueError):
            fig = None
        if fig is None:
            continue
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def figures_zip(d: Path, fmt: str = "pdf") -> bytes:
    import tempfile
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        files = render_all(d, Path(tmp), fmt)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(f, f.name)
            z.writestr("tables.tex", latex_tables(d))
    return buf.getvalue()


def _ci(v: dict | None, nd: int = 3) -> str:
    if not v or v.get("mean") is None:
        return "--"
    return f"{v['mean']:.{nd}f} [{v['lo']:.{nd}f}, {v['hi']:.{nd}f}]"


def latex_tables(d: Path) -> str:
    summary, conditions, detection = _load(d)
    h = summary["hypotheses"]
    names = {"fusion": "Fixed fusion", "fusion+defense": "Reliability-aware fusion", "moddrop": "ModDrop fusion",
             "moddrop+defense": "ModDrop + reliability-aware", **{f"uni_{m}": f"Unimodal {MOD_LABEL[m]}" for m in MOD_LABEL}}
    lines = ["% Generated by TrustFuse from result set " + summary["run_id"],
             "\\begin{table}[t]\\centering\\small",
             "\\caption{Clean subject-independent performance (" + summary["title"].replace("·", "--")
             + f", {summary['n_subjects']} LOSO folds; mean [95\\% CI]).}}",
             "\\begin{tabular}{lcc}\\toprule Model & Macro-F1 & Balanced acc. \\\\ \\midrule"]
    for b in summary["baselines"]:
        lines.append(f"{names.get(b['variant'], b['variant'])} & {_ci(b['macro_f1'])} & {_ci(b['balanced_accuracy'])} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", "",
              "\\begin{table}[t]\\centering\\small",
              "\\caption{Full single-modality suppression (" + h["primary_condition"] + "): fixed vs.\\ reliability-aware fusion.}",
              "\\begin{tabular}{lcccc}\\toprule Modality & Fixed F1 & Defended F1 & Gain & $p_{\\mathrm{Holm}}$ \\\\ \\midrule"]
    for r in h["H3"]["per_modality"]:
        lines.append(f"{MOD_LABEL[r['modality']]} & {_ci(r['fixed_f1'])} & {_ci(r['def_f1'])} & {_ci(r['gain'])} & "
                     f"{r['p_holm']:.3g} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}", "",
              "\\begin{table}[t]\\centering\\small\\caption{Hypothesis tests.}",
              "\\begin{tabular}{llc}\\toprule & Hypothesis & Supported \\\\ \\midrule"]
    for k in ("H1", "H2", "H3", "H4"):
        lines.append(f"{k} & {h[k]['title']} & {'yes' if h[k]['supported'] else 'no'} \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table}"]
    return "\n".join(lines) + "\n"
