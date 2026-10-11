from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...models.checkpoint import find_checkpoint, load_checkpoint
from ...specs import DATASETS, TASKS, get_task
from ..state import service

router = APIRouter(tags=["data"])


def _task(task: str):
    if task not in TASKS:
        raise HTTPException(404, f"unknown task {task}")
    return get_task(task)


def _subject(task: str, subject: str) -> None:
    if subject not in DATASETS[_task(task).dataset].subjects:
        raise HTTPException(404, f"unknown subject {subject}")


@router.get("/tasks")
def tasks() -> list[dict]:
    return service().tasks()


@router.get("/tasks/{task}/subjects")
def subjects(task: str) -> list[dict]:
    _task(task)
    return service().subjects(task)


@router.get("/tasks/{task}/subjects/{subject}/signals")
def signals(task: str, subject: str, t0: float | None = None, t1: float | None = None,
            points: int = Query(1500, ge=50, le=6000)) -> dict:
    _subject(task, subject)
    return service().signals(task, subject, t0, t1, points)


@router.get("/tasks/{task}/subjects/{subject}/windows")
def windows(task: str, subject: str) -> dict:
    _subject(task, subject)
    return service().windows(task, subject)


@router.get("/models/{task}/{variant}/{seed}/{subject}")
def model_info(task: str, variant: str, seed: int, subject: str) -> dict:
    _subject(task, subject)
    path = find_checkpoint(task, variant, seed, subject)
    if path is None:
        raise HTTPException(404, "checkpoint not found")
    model, ckpt = load_checkpoint(path)
    return {"task": task, "variant": variant, "seed": seed, "subject": subject, "split": ckpt["split"],
            "val_f1": ckpt["val_f1"], "history": ckpt["history"], "train_cfg": ckpt["train_cfg"],
            "created": ckpt["created"], "versions": ckpt.get("versions"), "config_hash": ckpt["config_hash"],
            "parameters": int(sum(p.numel() for p in model.parameters())), "bytes": path.stat().st_size}
