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
CALIBRATION = {"validation": "calibrated on the clean validation subjects only (report design)",
               "pooled": "signal-quality reference pooled over all non-test subjects (extension)"}
CALIBRATION_SHORT = {"validation": "validation subjects (report)", "pooled": "all non-test subjects (extension)"}


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
    d_ = s.get("defense")
    if d_:
        L += [f"Defense evaluated: reliability from {' + '.join(d_['components'])}, {CALIBRATION[d_['calibration']]}, "
              f"flag level α = {d_['alpha']:g}.", ""]
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
    ops = s.get("operating_points") or []
    if len(ops) > 1:
        L += ["", "### Evidence sets and calibration reference", "",
              "The first row is the defense used in every table above. The other rows change one thing at a time: "
              "the evidence set, the fusion strategy, or the calibration reference. \"Suppressed\" columns average over "
              f"the target modalities under {h['primary_condition']}.", "",
              "| Evidence | Calibration reference | Fusion | Clean false flags | Clean macro-F1 | Suppressed: fixed | Suppressed: defended | Suppressed: flagged |",
              "|---|---|---|---|---|---|---|---|"]
        for o in ops:
            name = " + ".join(o["components"])
            L.append(f"| {'**' + name + '**' if o['primary'] else name} | {CALIBRATION_SHORT[o['calibration']]} | "
                     f"{'product of experts' if o['strategy'] == 'poe' else 'embedding gate'} | {_ci(o['clean_frr'], pct=True)} | "
                     f"{_ci(o['clean_def_f1'])} | {_ci(o['full_fixed_f1'])} | {_ci(o['full_def_f1'])} | "
                     f"{_ci(o['full_flag_rate'], pct=True)} |")
    L += ["", "### Figures", ""]
    for name in ("fig_severity_zero", "fig_vulnerability_heatmap", "fig_attackers", "fig_detection_roc", "fig_forest_defense_gain", "fig_bursts"):
        L.append(f"![{name}]({fig_rel}/{name}.png)")
    L.append("")
    return L


HEADLINE_START, HEADLINE_END = "<!-- HEADLINE:START -->", "<!-- HEADLINE:END -->"
TASK_LABEL = {"wesad_binary": "WESAD stress vs. non-stress", "wesad_3class": "WESAD 3-class",
              "dalia_activity": "PPG-DaLiA activity (BVP+ACC)"}


def _m(v, nd=3, pct=False) -> str:
    if not v or v.get("mean") is None:
        return "—"
    return f"{100 * v['mean']:.1f}%" if pct else f"{v['mean']:.{nd}f}"


def _op(s: dict, components: str, calibration: str, strategy: str = "embedding_gate") -> dict | None:
    for o in s.get("operating_points") or []:
        if o["components"] == components and o["calibration"] == calibration and o["strategy"] == strategy:
            return o
    return None


def headline_lines() -> list[str]:
    """README 'Key findings' block. Every number comes from the canonical result sets."""
    sums = {}
    for task in TASK_LABEL:
        p = paths.ARTIFACTS_DIR / "results" / task / "summary.json"
        if p.exists():
            sums[task] = json.loads(p.read_text())
    if not sums:
        return []
    first = next(iter(sums.values()))
    d = first.get("defense") or {"components": ["Q", "D", "U"], "calibration": "validation"}
    design = f"{' + '.join(d['components'])}, {CALIBRATION_SHORT[d['calibration']].replace(' (report)', '')}"
    L = [f"## Key findings (measured, {first['n_subjects']} held-out subjects per dataset)", "",
         f"The defense evaluated is the one specified in the project report: reliability from {design}.", "",
         "| Task | Clean macro-F1 | Targeted suppression: fixed → defended | Clean false flags | Same, pooled-reference extension | Hypotheses |",
         "|---|---|---|---|---|---|"]
    for task, s in sums.items():
        h = s["hypotheses"]
        base = next(b for b in s["baselines"] if b["variant"] == "fusion")
        tg = next((a for a in h["secondary"]["attackers"] if a["attacker"] == "targeted"), None)
        ext = _op(s, "Q", "pooled")
        marks = " ".join(f"{k}{'✓' if h[k]['supported'] else '✗'}" for k in ("H1", "H2", "H3", "H4"))
        L.append(f"| {TASK_LABEL[task]} | {_m(base['macro_f1'])} | "
                 + (f"{_m(tg['fixed_f1'])} → {_m(tg['def_f1'])}" if tg else "—")
                 + f" | {_m(h['H4']['clean_frr'], pct=True)} | {_m(ext['clean_frr'], pct=True) if ext else '—'} | {marks} |")
    s = sums.get("wesad_binary") or first
    h = s["hypotheses"]
    base = next(b for b in s["baselines"] if b["variant"] == "fusion")
    att = {a["attacker"]: a for a in h["secondary"]["attackers"]}
    L.append("")
    tvr = h["secondary"].get("targeted_vs_random")
    if tvr and "targeted" in att and "random_drop" in att:
        line = ("* **Strategic suppression is a security problem, not just missing data.** On WESAD, removing one "
                f"modality chosen by a label-free attacker drops fixed-fusion macro-F1 from {_m(base['macro_f1'])} to "
                f"**{_m(att['targeted']['fixed_f1'])}**. Random modality loss with the same budget only reaches "
                f"{_m(att['random_drop']['fixed_f1'])}. The difference is {_ci(tvr['difference'], nd=2)}, one-sided Wilcoxon "
                f"p {_p(tvr['p'])}" + (f", rank-biserial r = {tvr['rank_biserial']:.2f}" if tvr.get("rank_biserial") is not None else "") + ".")
        md = next((r for r in s.get("moddrop_comparison") or [] if r["family"] == "targeted"), None)
        if md:
            line += (f" A model trained for generic missingness (ModDrop) still falls to {_m(md['moddrop_fixed'])} "
                     "under the same attacker.")
        L.append(line)
    h3 = h["H3"]
    mit = [r for r in h3["per_modality"] if r["modality"] in h3["mitigated_modalities"]]
    if mit:
        parts = "; ".join(f"{MOD_LABEL[r['modality']]} suppression {_m(r['fixed_f1'])} → **{_m(r['def_f1'])}** (p = {_p(r['p_holm'])}, Holm)"
                          for r in mit)
        L.append("* **Reliability-aware fusion mitigates it on the same frozen model.** " + parts + ". "
                 + (f"Targeted attacker: {_m(att['targeted']['fixed_f1'])} → {_m(att['targeted']['def_f1'])}; " if "targeted" in att else "")
                 + (f"oracle attacker: {_m(att['oracle']['fixed_f1'])} → {_m(att['oracle']['def_f1'])}. " if "oracle" in att else "")
                 + f"Clean macro-F1 cost {_ci(h3['clean_cost'])} (margin {h3['margin']}).")
    h4 = h["H4"]
    ext = _op(s, "Q", "pooled")
    prim = _op(s, "".join(d["components"]), d["calibration"])
    lead = ("False alarms on people the calibration has not seen stay within the limit." if h4["frr_ok"]
            else "The price is false alarms on people the calibration has not seen.")
    line = (f"* **{lead}** With thresholds from the clean "
            f"validation subjects of each fold, as the report specifies, the low-confidence flag fires on "
            f"{_ci(h4['clean_frr'], pct=True)} of clean windows of held-out people (pre-registered limit "
            f"{100 * h4['frr_max']:.0f}%), so H4 is {'supported' if h4['supported'] else 'not supported'} on WESAD.")
    if ext and prim:
        line += (f" Extension: referencing the model-free signal-quality evidence to all non-test subjects brings this to "
                 f"{_m(ext['clean_frr'], pct=True)} with the same protection under full suppression "
                 f"(defended macro-F1 {_m(ext['full_def_f1'])} versus {_m(prim['full_def_f1'])}).")
    L.append(line)
    limits = ["* **Honest limits.**"]
    if "adaptive" in att:
        limits.append(f"  * A defense-aware attacker still flips {_m(att['adaptive']['evasive_def_flip'], pct=True)} of decisions "
                      "without being flagged.")
    unmit = [MOD_LABEL[r["modality"]] for r in h3["per_modality"]
             if r["modality"] in h["H1"]["supported_modalities"] and r["modality"] not in h3["mitigated_modalities"]]
    if unmit:
        limits.append(f"  * Gating discards a whole stream, so it cannot recover a modality the model depends on most "
                      f"({', '.join(unmit)} on WESAD).")
    dal = sums.get("dalia_activity")
    if dal:
        g = dal["hypotheses"]["H3"]["mean_gain_all_attacks"]
        limits.append(f"  * On two-modality PPG-DaLiA the remaining stream carries too little information: averaged over all "
                      f"window-level attacks the defense changes macro-F1 by {_ci(g)}.")
    L += limits
    L += ["", "Full tables and figures: [docs/RESULTS.md](docs/RESULTS.md)."]
    return L


def write_readme_headline(readme: Path | None = None) -> Path | None:
    """Rewrite the block between the HEADLINE markers of README.md from the canonical results."""
    readme = readme or paths.REPO_ROOT / "README.md"
    lines = headline_lines()
    if not readme.exists() or not lines:
        return None
    text = readme.read_bytes().decode("utf-8")
    a, b = text.find(HEADLINE_START), text.find(HEADLINE_END)
    if a < 0 or b < 0:
        return None
    nl = "\r\n" if "\r\n" in text else "\n"      # keep the checkout's line endings
    text = text[:a + len(HEADLINE_START)] + nl + nl.join(lines) + nl + text[b:]
    readme.write_bytes(text.encode("utf-8"))
    return readme


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
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return out
