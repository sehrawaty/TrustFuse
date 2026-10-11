from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..schemas import InferIn, TimelineIn, WindowIn
from ..state import service

router = APIRouter(tags=["inference"])


async def _call(fn, *args, **kw):
    try:
        return await run_in_threadpool(fn, *args, **kw)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (IndexError, KeyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/infer")
async def infer(req: InferIn) -> dict:
    attack = req.attack.model_dump()
    if attack["type"] != "none" and not attack.get("modality"):
        raise HTTPException(422, "attack.modality is required")
    return await _call(service().infer, req.task, req.subject, req.window, attack, req.variant, req.seed,
                       req.components, req.strategy, req.calibration)


@router.post("/attack/targeted")
async def targeted(req: WindowIn) -> dict:
    return await _call(service().worst_modality, req.task, req.subject, req.window, req.fill, req.variant, req.seed)


@router.post("/attack/adaptive")
async def adaptive(req: WindowIn) -> dict:
    return await _call(service().adaptive, req.task, req.subject, req.window, req.variant, req.seed)


@router.post("/timeline")
async def timeline(req: TimelineIn) -> dict:
    burst = req.burst.model_dump() if req.burst else None
    return await _call(service().timeline, req.task, req.subject, burst, req.variant, req.seed)


@router.get("/calibration/{task}/{subject}")
async def calibration(task: str, subject: str, variant: str = "fusion", seed: int = 0,
                      components: str = "QDU", calibration: str = "validation") -> dict:
    comps = [c for c in components.upper() if c in "QDU"]
    if calibration not in ("validation", "pooled"):
        raise HTTPException(422, "calibration must be 'validation' or 'pooled'")
    return await _call(service().calibration, task, subject, variant, seed, comps or None, calibration)
