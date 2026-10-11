"""docs/RESULTS.md + docs/figures/ from the canonical result sets. Every number is read from the
result files; nothing is typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from .figures import MOD_LABEL, render_all

VARIANT = {"fusion": "Fixed fusion", "fusion+defense": "Reliability-aware fusion", "moddrop": "ModDrop fusion",
           "moddrop+defense": "ModDrop + reliability-aware", **{f"uni_{m}": f"Unimodal {MOD_LABEL[m]}" for m in MOD_LABEL}}
ATTACKER = {"random_drop": "Random", "targeted": "Targeted", "oracle": "Oracle", "adaptive": "Adaptive (defense-aware)"}


def _ci(v, nd=3, pct=False):
    if not v or v.get("mean") is None:
        return "—"
    f = (lambda x: f"{100 * x:.1f}%") if pct else (lambda x: f"{x:.{nd}f}")
    if v.get("lo") is None or v["lo"] == v["hi"]:
        return f(v["mean"])
    return f"{f(v['mean'])} [{f(v['lo'])}, {f(v['hi'])}]"


def _p(p):
    return "n/a" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def section(d: Path, fig_rel: str) -> list[str]:
    s = json.loads((d / "summary.json").read_text())
    run = json.loads((d / "run.json").read_text())
    h = s["hypotheses"]
    L = [f"## {s['title']}", "",
         f"Result set `{s['run_id']}` · config `{run['config_hash']}` · {s['n_subjects']} LOSO folds · "
         f"{s['n_windows']:,} test windows · seeds {', '.join(map(str, run['seeds']))} · {run['n_rows']:,} measurements.", ""]
    L += ["### Clean performance", "", "| Model | Macro-F1 [95% CI] | Balanced accuracy |", "|---|---|---|"]
    for b in s["baselines"]:
        L.append(f"| {VARIANT.get(b['variant'], b['variant'])} | {_ci(b['macro_f1'])} | {_ci(b['balanced_accuracy'])} |")
    L += ["", "### Hypotheses", "", "| | Hypothesis | Outcome | Key evidence |", "|---|---|---|---|"]
    best = max(h["H1"]["per_modality"], key=lambda r: r["drop"]["mean"] or 0)
    L.append(f"| H1 | {h['H1']['title']} | {'supported' if h['H1']['supported'] else 'not supported'} | "
             f"significant for {', '.join(MOD_LABEL[m] for m in h['H1']['supported_modalities']) or 'none'}; largest drop "
             f"{MOD_LABEL[best['modality']]} {_ci(best['drop'])} |")
    fr = h["H2"]["friedman"]
    L.append(f"| H2 | {h['H2']['title']} | {'supported' if h['H2']['supported'] else 'not supported'} | "
             f"ranking {' > '.join(MOD_LABEL[r['modality']] for r in h['H2']['ranking'])}; p {_p(fr.get('p'))}"
             + (f", Kendall's W {fr['kendall_w']:.2f}" if fr.get("kendall_w") is not None else "") + " |")
    L.append(f"| H3 | {h['H3']['title']} | {'supported' if h['H3']['supported'] else 'not supported'} | "
             f"clean cost {_ci(h['H3']['clean_cost'])} (margin {h['H3']['margin']}); mitigated: "
             f"{', '.join(MOD_LABEL[m] for m in h['H3']['mitigated_modalities']) or 'none'}; mean gain over all attacks "
             f"{_ci(h['H3']['mean_gain_all_attacks'])} |")
    L.append(f"| H4 | {h['H4']['title']} | {'supported' if h['H4']['supported'] else 'not supported'} | "
             f"clean FRR {_ci(h['H4']['clean_frr'], pct=True)} (limit {100 * h['H4']['frr_max']:.0f}%); detectable: "
             f"{', '.join(MOD_LABEL[m] for m in h['H4']['detectable_modalities']) or 'none'} |")
    L += ["", f"### Full suppression of one modality ({h['primary_condition']})", "",
          "| Suppressed | Fixed macro-F1 | Defended macro-F1 | Gain | p (Holm) | Detection AUROC |", "|---|---|---|---|---|---|"]
    auc = {r["modality"]: r["auroc"] for r in h["H4"]["per_modality"]}
    for r in h["H3"]["per_modality"]:
        L.append(f"| {MOD_LABEL[r['modality']]} | {_ci(r['fixed_f1'])} | {_ci(r['def_f1'])} | {_ci(r['gain'])} | "
                 f"{_p(r['p_holm'])} | {_ci(auc.get(r['modality']))} |")
    att = h["secondary"].get("attackers", [])
    if att:
        L += ["", "### Attacker capability", "", "| Attacker | Fixed macro-F1 | Defended macro-F1 | Flagged | Evasive defended flips |",
              "|---|---|---|---|---|"]
        for a in att:
            L.append(f"| {ATTACKER.get(a['attacker'], a['attacker'])} | {_ci(a['fixed_f1'])} | {_ci(a['def_f1'])} | "
                     f"{_ci(a['flag_rate'], pct=True)} | {_ci(a['evasive_def_flip'], pct=True)} |")
        tvr = h["secondary"].get("targeted_vs_random")
        if tvr:
            L += ["", f"Targeted versus random suppression: macro-F1 difference {_ci(tvr['difference'])}, "
                      f"one-sided Wilcoxon p {_p(tvr['p'])}."]
    if s.get("moddrop_comparison"):
        L += ["", "### Robustness training (ModDrop) versus the defense", "",
              "| Attack family | Fusion fixed | Fusion defended | ModDrop fixed | ModDrop defended |", "|---|---|---|---|---|"]
        for r in s["moddrop_comparison"]:
            L.append(f"| {r['family']} | {_ci(r['fusion_fixed'])} | {_ci(r['fusion_defended'])} | {_ci(r['moddrop_fixed'])} | "
                     f"{_ci(r['moddrop_defended'])} |")
    L += ["", "### Figures", ""]
    for name in ("fig_severity_zero", "fig_vulnerability_heatmap", "fig_attackers", "fig_detection_roc", "fig_forest_defense_gain", "fig_bursts"):
        L.append(f"![{name}]({fig_rel}/{name}.png)")
    L.append("")
    return L


def write_results_md(out: Path | None = None) -> Path:
    out = out or paths.REPO_ROOT / "docs" / "RESULTS.md"
    lines = ["# Results", "",
             "Generated by `python -m trustfuse report` from the canonical result sets in `artifacts/results/`. "
             "Values are subject-level means over leave-one-subject-out folds with 95 % subject-bootstrap confidence "
             "intervals. A hypothesis that the data do not support is reported as not supported.", ""]
    for task in ("wesad_binary", "wesad_3class", "dalia_activity"):
        d = paths.ARTIFACTS_DIR / "results" / task
        if not (d / "summary.json").exists():
            continue
        fig_dir = out.parent / "figures" / task
        render_all(d, fig_dir, "png")
        render_all(d, fig_dir, "pdf")
        lines += section(d, f"figures/{task}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out
