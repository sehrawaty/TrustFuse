from __future__ import annotations

import asyncio
import json

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ... import paths
from ...experiments.config import ExperimentConfig
from ...experiments.training import enumerate_jobs
from ..schemas import JobIn
from ..state import jobs

router = APIRouter(tags=["experiments"])

# Presets offered in the UI. "Evaluate frozen checkpoints" re-runs the attack matrix on the shipped
# models without retraining; user runs never overwrite the shipped canonical results.
PRESETS = {
    "quick_demo": {"file": "quick_demo.yaml", "label": "Quick demo", "minutes": 2,
                   "summary": "Train a fusion model for one held-out subject and run a compact attack matrix."},
    "evaluate_frozen": {"file": "wesad_binary.yaml", "label": "Evaluate frozen checkpoints", "minutes": 2,
                        "summary": "Re-run the attack matrix on shipped LOSO models for selected subjects.",
                        "overrides": {"name": "Frozen checkpoints — re-evaluation", "test_subjects": ["S2", "S3", "S4"],
                                      "variants": ["fusion"], "seeds": [0], "eval_seeds": [0], "models_root": "artifacts",
                                      "reliability": {"ablations": [], "calibration_ablations": []},
                                      "attacks": {"strategies": ["random_drop", "targeted", "oracle"],
                                                  "burst_placements_per_subject": 4}}},
    "wesad_binary": {"file": "wesad_binary.yaml", "label": "Full LOSO · WESAD binary", "minutes": 240,
                     "summary": "The complete headline experiment (15 folds, 3 seeds). Hours on a laptop."},
    "wesad_3class": {"file": "wesad_3class.yaml", "label": "Full LOSO · WESAD 3-class", "minutes": 150,
                     "summary": "Supplementary three-class experiment."},
    "dalia_activity": {"file": "dalia_activity.yaml", "label": "Full LOSO · PPG-DaLiA", "minutes": 200,
                       "summary": "Focused BVP + ACC validation on daily-life recordings."},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _preset_config(key: str, overrides: dict | None = None) -> ExperimentConfig:
    if key not in PRESETS:
        raise HTTPException(404, f"unknown preset {key}")
    p = PRESETS[key]
    raw = yaml.safe_load((paths.CONFIGS_DIR / p["file"]).read_text())
    raw = _merge(raw, p.get("overrides", {}))
    raw = _merge(raw, overrides or {})
    # user-launched runs always write to the runtime area
    raw["results_root"] = "runtime"
    if raw.get("models_root") != "artifacts":
        raw["models_root"] = "runtime"
    return ExperimentConfig.model_validate(raw)


@router.get("/experiments/presets")
def presets() -> list[dict]:
    out = []
    for key, p in PRESETS.items():
        cfg = _preset_config(key)
        out.append({"key": key, "label": p["label"], "summary": p["summary"], "minutes": p["minutes"],
                    "config": cfg.model_dump(), "config_hash": cfg.config_hash(),
                    "train_jobs": len(enumerate_jobs(cfg)), "test_subjects": cfg.test_subjects})
    return out


@router.post("/experiments/validate")
def validate(body: JobIn) -> dict:
    cfg = _build(body)
    return {"ok": True, "config": cfg.model_dump(), "config_hash": cfg.config_hash(),
            "train_jobs": len(enumerate_jobs(cfg)), "yaml": cfg.to_yaml()}


def _build(body: JobIn) -> ExperimentConfig:
    try:
        if body.preset:
            return _preset_config(body.preset, body.overrides)
        if body.config:
            raw = dict(body.config)
            raw["results_root"] = "runtime"
            raw.setdefault("models_root", "runtime")
            return ExperimentConfig.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(422, json.loads(exc.json())) from exc
    raise HTTPException(422, "preset or config required")


@router.post("/jobs")
def submit(body: JobIn) -> dict:
    cfg = _build(body)
    try:
        return jobs().submit(cfg).public()
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc


@router.get("/jobs")
def list_jobs() -> list[dict]:
    return jobs().list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        job = jobs().get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc
    return {**job.public(), "config": job.config, "events": job.events[-400:]}


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict:
    try:
        return jobs().cancel(job_id).public()
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc


@router.get("/jobs/{job_id}/events")
async def stream(job_id: str, request: Request, since: int = 0) -> StreamingResponse:
    try:
        jobs().get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "job not found") from exc

    async def gen():
        seq = since
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            job = jobs().get(job_id)
            batch = job.events[seq:]
            for ev in batch:
                yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps({**ev, 'job': job.public()})}\n\n"
            seq += len(batch)
            if job.status in ("completed", "failed", "cancelled") and not batch:
                yield f"event: end\ndata: {json.dumps({'job': job.public()})}\n\n"
                break
            idle = 0 if batch else idle + 1
            if idle and idle % 30 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
