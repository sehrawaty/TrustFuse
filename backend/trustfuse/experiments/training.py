"""Training phase: one job per (held-out subject, variant, seed); resumable and parallelizable."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .. import paths
from ..data.loader import DatasetLoader
from ..data.preprocess import Preprocessor
from ..data.windows import gather
from ..models.checkpoint import build_model, checkpoint_path, save_checkpoint
from ..models.train import TrainConfig, predict_logits, train_model
from ..evaluation.metrics import macro_f1
from ..specs import get_task
from .config import ExperimentConfig


@dataclass(frozen=True)
class TrainJob:
    task: str
    variant: str
    seed: int
    subject: str


def models_root(cfg: ExperimentConfig) -> Path:
    return paths.ARTIFACTS_DIR / "models" if cfg.models_root == "artifacts" else paths.RUNTIME_DIR / "models"


def test_subjects(cfg: ExperimentConfig) -> list[str]:
    loader = DatasetLoader(get_task(cfg.task).dataset)
    return list(cfg.test_subjects or loader.spec.subjects)


def enumerate_jobs(cfg: ExperimentConfig) -> list[TrainJob]:
    jobs = []
    for subject in test_subjects(cfg):
        for variant in cfg.variants:
            for seed in cfg.seeds_for(variant):
                jobs.append(TrainJob(cfg.task, variant, seed, subject))
    return jobs


def run_train_job(cfg: ExperimentConfig, job: TrainJob, threads: int = 1,
                  on_epoch: Callable[[dict], None] | None = None,
                  should_stop: Callable[[], bool] | None = None) -> dict:
    task = get_task(job.task)
    loader = DatasetLoader(task.dataset)
    split = loader.split_data(job.subject, n_val=cfg.n_val)
    train_ws = gather(job.task, split.train)
    val_ws = gather(job.task, split.val)
    pp = Preprocessor(task)
    norm = pp.fit_normalizer(train_ws.phys)              # training-fold statistics only
    x_train = pp.normalize(train_ws.phys, norm)
    x_val = pp.normalize(val_ws.phys, norm)
    if job.variant.startswith("uni_"):
        m = job.variant[4:]
        x_train, x_val = {m: x_train[m]}, {m: x_val[m]}
    tc = TrainConfig(**cfg.train.model_dump(exclude={"moddrop_p"}), seed=job.seed, threads=threads,
                     moddrop_p=cfg.train.moddrop_p if job.variant == "moddrop" else 0.0)
    model = build_model(job.task, job.variant)
    t0 = time.time()
    model, history = train_model(model, (x_train, train_ws.y), (x_val, val_ws.y), task.n_classes, tc,
                                 on_epoch=on_epoch, should_stop=should_stop)
    val_f1 = macro_f1(val_ws.y, predict_logits(model, x_val).argmax(1), task.n_classes)
    path = checkpoint_path(models_root(cfg), job.task, job.variant, job.seed, job.subject)
    save_checkpoint(path, model, task=job.task, variant=job.variant, seed=job.seed, subject=job.subject,
                    split={"train": list(split.train), "val": list(split.val), "test": list(split.test)},
                    norm=norm, train_cfg=tc.to_dict(), history=history, val_f1=val_f1,
                    config_hash=cfg.config_hash())
    return {"job": job.__dict__, "val_f1": val_f1, "epochs": len(history), "seconds": time.time() - t0,
            "path": str(path)}


def _worker(args: tuple[dict, TrainJob]) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cfg = ExperimentConfig.model_validate(args[0])
    return run_train_job(cfg, args[1], threads=1)


def train_phase(cfg: ExperimentConfig, workers: int = 1, log: Callable[[str], None] = print,
                should_stop: Callable[[], bool] | None = None,
                on_epoch: Callable[[TrainJob, dict], None] | None = None) -> list[dict]:
    jobs = enumerate_jobs(cfg)
    root = models_root(cfg)
    todo = [j for j in jobs if not checkpoint_path(root, j.task, j.variant, j.seed, j.subject).exists()]
    log(f"[train] {cfg.task}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    results = []
    if workers <= 1:
        for i, job in enumerate(todo, 1):
            r = run_train_job(cfg, job, threads=int(os.environ.get("TRUSTFUSE_THREADS", "2")),
                              should_stop=should_stop,
                              on_epoch=(lambda rec, j=job: on_epoch(j, rec)) if on_epoch else None)
            results.append(r)
            log(f"[train] {i}/{len(todo)} {job.variant} seed{job.seed} {job.subject}: "
                f"val F1 {r['val_f1']:.3f} ({r['epochs']} ep, {r['seconds']:.0f}s)")
        return results
    payload = cfg.model_dump()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (payload, j)): j for j in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            r = fut.result()
            results.append(r)
            log(f"[train] {i}/{len(todo)} {job.variant} seed{job.seed} {job.subject}: "
                f"val F1 {r['val_f1']:.3f} ({r['epochs']} ep, {r['seconds']:.0f}s)")
    return results


def _priority(job: TrainJob) -> tuple:
    variant_rank = {"fusion": 0, "moddrop": 1}.get(job.variant, 2)
    task_rank = {"wesad_binary": 0, "wesad_3class": 1, "dalia_activity": 2}.get(job.task, 3)
    return (job.seed > 0, variant_rank, task_rank, job.seed)


def train_many(cfgs: list[ExperimentConfig], workers: int, log: Callable[[str], None] = print) -> None:
    """Train the jobs of several configs in one process pool, seed-0 fusion models first."""
    queue = []
    for cfg in cfgs:
        root = models_root(cfg)
        queue += [(cfg, j) for j in enumerate_jobs(cfg)
                  if not checkpoint_path(root, j.task, j.variant, j.seed, j.subject).exists()]
    queue.sort(key=lambda cj: _priority(cj[1]))
    log(f"[train] {len(queue)} jobs queued across {len(cfgs)} configs with {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, (cfg.model_dump(), j)): j for cfg, j in queue}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                r = fut.result()
                log(f"[train] {i}/{len(queue)} {job.task} {job.variant} seed{job.seed} {job.subject}: "
                    f"val F1 {r['val_f1']:.3f} ({r['epochs']} ep, {r['seconds']:.0f}s) "
                    f"elapsed {(time.time() - t0) / 60:.1f} min")
            except Exception as exc:  # keep the remaining jobs running
                log(f"[train] FAILED {job}: {exc!r}")
