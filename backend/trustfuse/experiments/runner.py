"""Experiment runner: train -> calibrate -> attack matrix -> aggregate -> statistics -> report.

A result set is a directory ``<results root>/<run_id>/`` with:

    run.json          provenance (config hash, versions, timing, git commit, status)
    config.yaml       the exact experiment configuration
    rows.csv.gz       one row per variant x seed x subject x condition (raw measurements)
    subjects.csv.gz   seed-averaged subject-level table (unit of statistical analysis)
    conditions.json   per variant/condition aggregates with subject-bootstrap 95% CIs (UI)
    summary.json      baselines, hypotheses H1-H4, secondary analyses, headline numbers
    detection.json    pooled ROC curves and the detectability matrix
    manifest.npz      traceable per-window decisions for every attacked window (manifest seed)
    calibration.json  frozen reliability calibration of every fold (primary model)
"""

from __future__ import annotations

import gzip
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__, paths
from ..evaluation.evaluator import Evaluator
from ..evaluation.hypotheses import test_hypotheses
from ..evaluation.metrics import auroc, roc_curve
from ..evaluation.stats import bootstrap_ci
from ..service.bundle import load_bundle
from ..specs import get_task
from .config import ExperimentConfig
from .training import test_subjects, train_phase

KEY_METRICS = ("fixed_f1", "def_f1", "fixed_acc", "def_acc", "fixed_bacc", "def_bacc", "fixed_flip", "def_flip",
               "fixed_asr", "def_asr", "fixed_conf", "def_conf", "fixed_ptrue", "def_ptrue", "fixed_cwrong",
               "def_cwrong", "fixed_ece", "def_ece", "flag_rate", "sel_acc", "coverage", "evasive_def_flip",
               "auroc", "auroc_Q", "auroc_D", "auroc_U", "r_target", "w_target", "latency_s", "detected_bursts")
META = ("cond", "family", "attack", "modality", "fill", "severity", "snr_db", "burst", "scope", "components", "strategy")


def results_root(cfg: ExperimentConfig) -> Path:
    return paths.ARTIFACTS_DIR / "results" if cfg.results_root == "artifacts" else paths.RUNTIME_DIR / "results"


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def eval_jobs(cfg: ExperimentConfig) -> list[tuple[str, int, str]]:
    jobs = []
    for subject in test_subjects(cfg):
        for variant in cfg.variants:
            seeds = cfg.seeds_for(variant)
            if variant in cfg.eval_variants:
                seeds = [s for s in seeds if s in cfg.eval_seeds] or seeds[:1]
            elif not variant.startswith("uni_"):
                continue
            else:
                seeds = seeds[:1]
            for seed in seeds:
                jobs.append((variant, seed, subject))
    return jobs


def run_eval_job(cfg: ExperimentConfig, variant: str, seed: int, subject: str,
                 log: Callable[[str], None] | None = None, should_stop: Callable[[], bool] | None = None) -> dict:
    t0 = time.time()
    bundle = load_bundle(cfg.task, variant, seed, subject, mc_samples=cfg.reliability.mc_samples)
    if bundle.engine is not None:
        bundle.engine.alpha = cfg.reliability.alpha
        bundle.engine.gate_span = cfg.reliability.gate_span
        bundle.engine._dist_cache.clear()
    ev = Evaluator(bundle, cfg, log=log, should_stop=should_stop)
    if bundle.is_fusion:
        res = ev.evaluate(with_manifest=(variant == "fusion" and seed == cfg.manifest_seed))
        calib = bundle.engine.calibration_summary(tuple(cfg.reliability.components)) \
            if variant == "fusion" and seed == cfg.manifest_seed else None
    else:
        res = ev.evaluate_unimodal()
        calib = None
    return {"variant": variant, "seed": seed, "subject": subject, "rows": res.rows, "manifest": res.manifest,
            "windows": res.windows, "scores": res.scores, "calibration": calib, "seconds": time.time() - t0}


def _eval_worker(args) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    cfg = ExperimentConfig.model_validate(args[0])
    return run_eval_job(cfg, *args[1:])


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=1, ensure_ascii=False))


def evaluate_experiment(cfg: ExperimentConfig, workers: int = 1, log: Callable[[str], None] = print,
                        run_id: str | None = None, should_stop: Callable[[], bool] | None = None,
                        progress: Callable[[float, str], None] | None = None) -> Path:
    t_start = time.time()
    run_id = run_id or (cfg.task if cfg.results_root == "artifacts" else f"{cfg.task}-{time.strftime('%Y%m%d-%H%M%S')}")
    out = results_root(cfg) / run_id
    out.mkdir(parents=True, exist_ok=True)
    jobs = eval_jobs(cfg)
    log(f"[eval] {cfg.task}: {len(jobs)} fold evaluations with {workers} worker(s) -> {out}")
    results = []
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            if should_stop and should_stop():
                raise InterruptedError("cancelled")
            results.append(run_eval_job(cfg, *job, log=log, should_stop=should_stop))
            log(f"[eval] {i}/{len(jobs)} {job[0]} seed{job[1]} {job[2]} ({results[-1]['seconds']:.0f}s)")
            if progress:
                progress(i / len(jobs), f"evaluated {job[0]} seed{job[1]} {job[2]}")
    else:
        payload = cfg.model_dump()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_eval_worker, (payload, *j)): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                j = futs[fut]
                log(f"[eval] {i}/{len(jobs)} {j[0]} seed{j[1]} {j[2]} ({results[-1]['seconds']:.0f}s)")
    aggregate(cfg, results, out, run_id, time.time() - t_start, log)
    return out


# ------------------------------------------------------------------------------------------
def aggregate(cfg: ExperimentConfig, results: list[dict], out: Path, run_id: str, seconds: float,
              log: Callable[[str], None] = print) -> None:
    task = get_task(cfg.task)
    rows = pd.DataFrame([r for res in results for r in res["rows"]])
    for c in META:
        if c not in rows:
            rows[c] = None
    with gzip.open(out / "rows.csv.gz", "wt") as f:
        rows.to_csv(f, index=False)

    # subject-level table: average over seeds within subject
    group_cols = ["variant", "subject", "cond"]
    num_cols = [c for c in KEY_METRICS if c in rows] + ["n"]
    meta_first = rows.groupby(group_cols, dropna=False)[[c for c in META if c != "cond"]].first()
    subj = rows.groupby(group_cols, dropna=False)[num_cols].mean().join(meta_first).reset_index()
    with gzip.open(out / "subjects.csv.gz", "wt") as f:
        subj.to_csv(f, index=False)

    # seed variability of the subject-mean (primary metrics only)
    seed_sd = (rows.groupby(["variant", "cond", "seed"])[["fixed_f1", "def_f1"]].mean()
               .groupby(["variant", "cond"]).std(ddof=1))

    boot = dict(n_boot=cfg.stats.n_boot, seed=cfg.stats.seed)
    conditions = []
    for (variant, cond), g in subj.groupby(["variant", "cond"], dropna=False):
        first = g.iloc[0]
        entry = {"variant": variant, "cond": cond, "n_subjects": int(g.subject.nunique()),
                 **{k: (None if pd.isna(first[k]) else first[k]) for k in META if k != "cond"}}
        for m in KEY_METRICS:
            if m in g and g[m].notna().any():
                ci = bootstrap_ci(g[m].to_numpy(float), **boot)
                entry[m] = {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]}
        if (variant, cond) in seed_sd.index:
            sd = seed_sd.loc[(variant, cond)]
            entry["seed_sd"] = {"fixed_f1": sd.get("fixed_f1"), "def_f1": sd.get("def_f1")}
        conditions.append(entry)
    write_json(out / "conditions.json", {"run_id": run_id, "task": cfg.task, "conditions": conditions})

    # baselines (clean)
    clean = subj[subj.family == "clean"]
    baselines = []
    for variant, g in clean.groupby("variant"):
        baselines.append({"variant": variant, "macro_f1": bootstrap_ci(g.fixed_f1, **boot),
                          "accuracy": bootstrap_ci(g.fixed_acc, **boot),
                          "balanced_accuracy": bootstrap_ci(g.fixed_bacc, **boot)})
        if variant in ("fusion", "moddrop"):
            baselines.append({"variant": f"{variant}+defense", "macro_f1": bootstrap_ci(g.def_f1, **boot),
                              "accuracy": bootstrap_ci(g.def_acc, **boot),
                              "balanced_accuracy": bootstrap_ci(g.def_bacc, **boot),
                              "flag_rate": bootstrap_ci(g.flag_rate, **boot)})

    primary = subj[subj.variant == "fusion"]
    hyp = test_hypotheses(primary, cfg, task.modalities)
    moddrop_cmp = None
    if (subj.variant == "moddrop").any():
        md = subj[subj.variant == "moddrop"]
        moddrop_cmp = []
        for fam in ("mask", "attenuate", "targeted", "oracle", "adaptive", "random_drop"):
            a, b = primary[primary.family == fam], md[md.family == fam]
            if len(a) and len(b):
                moddrop_cmp.append({"family": fam,
                                    "fusion_fixed": bootstrap_ci(a.groupby("subject").fixed_f1.mean(), **boot),
                                    "fusion_defended": bootstrap_ci(a.groupby("subject").def_f1.mean(), **boot),
                                    "moddrop_fixed": bootstrap_ci(b.groupby("subject").fixed_f1.mean(), **boot),
                                    "moddrop_defended": bootstrap_ci(b.groupby("subject").def_f1.mean(), **boot)})

    # forest plot: per-subject defense gain under the primary condition, per modality
    forest = []
    for m in task.modalities:
        pr = primary[(primary.family == "mask") & (primary.fill == cfg.stats.primary_fill)
                     & np.isclose(primary.severity.astype(float), cfg.stats.primary_severity) & (primary.modality == m)]
        for _, r in pr.iterrows():
            forest.append({"modality": m, "subject": r.subject, "fixed_f1": r.fixed_f1, "def_f1": r.def_f1,
                           "gain": r.def_f1 - r.fixed_f1})

    summary = {
        "run_id": run_id, "task": cfg.task, "title": task.title, "classes": list(task.classes),
        "modalities": list(task.modalities), "n_subjects": int(subj.subject.nunique()),
        "n_windows": int(clean[clean.variant == "fusion"].n.sum()) if len(clean) else 0,
        "baselines": baselines, "hypotheses": hyp, "moddrop_comparison": moddrop_cmp, "forest": forest,
        "svi": {r["modality"]: r["svi"] for r in hyp["H2"]["ranking"]},
    }
    write_json(out / "summary.json", summary)

    # detection: pooled ROC (manifest seed, fusion) + detectability matrix
    det_scores: dict[str, list[np.ndarray]] = {}
    for res in results:
        if res["variant"] == "fusion" and res["seed"] == cfg.manifest_seed:
            for cid, sc in res["scores"].items():
                det_scores.setdefault(cid, []).append(sc)
    curves = {}
    if "clean" in det_scores:
        neg = np.concatenate(det_scores["clean"])
        for cid, arrs in det_scores.items():
            if cid == "clean":
                continue
            pos = np.concatenate(arrs)
            curves[cid] = {"auroc": auroc(neg, pos), **roc_curve(neg, pos, max_points=60)}
    matrix = []
    fam_rows = primary[primary.family.isin(["mask", "attenuate", "noise", "random_drop", "targeted", "oracle", "adaptive"])]
    for (fam, m), g in fam_rows.groupby(["family", "modality"], dropna=False):
        entry = {"family": fam, "modality": None if pd.isna(m) else m}
        for c in ("auroc", "auroc_Q", "auroc_D", "auroc_U"):
            if c in g:
                entry[c] = float(g[c].mean())
        matrix.append(entry)
    write_json(out / "detection.json", {"curves": curves, "matrix": matrix,
                                        "clean_frr": hyp["H4"]["clean_frr"], "alpha": cfg.reliability.alpha})

    # manifest: per-window decisions of every attacked window (traceability)
    man = {}
    meta_cols = {"subject": [], "start_s": [], "y": []}
    order = sorted([r for r in results if r["manifest"]], key=lambda r: r["subject"])
    if order:
        for r in order:
            meta_cols["subject"].append(np.full(len(r["windows"]["y"]), r["subject"]))
            meta_cols["start_s"].append(r["windows"]["start_s"])
            meta_cols["y"].append(r["windows"]["y"])
        keys = sorted(set().union(*[r["manifest"].keys() for r in order]))
        for k in keys:
            if all(k in r["manifest"] for r in order):
                man[k] = np.concatenate([r["manifest"][k] for r in order], axis=-1)
        np.savez_compressed(out / "manifest.npz", conditions=np.array(keys),
                            fields=np.array(["fixed_pred", "def_pred", "fixed_conf", "def_conf", "r_target",
                                             "w_target", "flag"]),
                            subject=np.concatenate(meta_cols["subject"]).astype("U4"),
                            start_s=np.concatenate(meta_cols["start_s"]), y=np.concatenate(meta_cols["y"]),
                            **{f"c{i}": man[k] for i, k in enumerate(keys) if k in man})

    write_json(out / "calibration.json", {r["subject"]: r["calibration"] for r in results if r["calibration"]})
    (out / "config.yaml").write_text(cfg.to_yaml())
    import torch
    write_json(out / "run.json", {
        "run_id": run_id, "name": cfg.name, "description": cfg.description, "task": cfg.task,
        "config_hash": cfg.config_hash(), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": seconds, "status": "complete", "trustfuse": __version__, "git_commit": _git_commit(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__},
        "subjects": sorted(subj.subject.unique().tolist()), "variants": sorted(subj.variant.unique().tolist()),
        "seeds": cfg.eval_seeds, "n_rows": int(len(rows)), "canonical": cfg.results_root == "artifacts",
    })
    log(f"[eval] aggregated {len(rows)} rows -> {out}")


def run_experiment(cfg: ExperimentConfig, log: Callable[[str], None] = print,
                   should_stop: Callable[[], bool] | None = None,
                   on_epoch: Callable | None = None, progress: Callable[[float, str], None] | None = None,
                   run_id: str | None = None) -> Path:
    train_phase(cfg, workers=1, log=log, should_stop=should_stop, on_epoch=on_epoch)
    return evaluate_experiment(cfg, workers=1, log=log, should_stop=should_stop, progress=progress, run_id=run_id)
