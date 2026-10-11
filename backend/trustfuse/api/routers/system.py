from __future__ import annotations

import os
import platform
import time
from pathlib import Path

from fastapi import APIRouter

from ... import __version__, paths
from ...data.loader import DatasetLoader
from ...specs import DATASETS, MODALITY_SPECS
from ..state import jobs, service

router = APIRouter(tags=["system"])
STARTED = time.time()


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def memory() -> dict:
    rss = None
    status = _read("/proc/self/status") or ""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    limit = _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read("/sys/fs/cgroup/memory.current") or _read("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    lim = int(limit) if limit and limit.isdigit() and int(limit) < 1 << 50 else None
    return {"rss_bytes": rss, "cgroup_limit_bytes": lim, "cgroup_used_bytes": int(used) if used and used.isdigit() else None}


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - STARTED, 1)}


@router.get("/system")
def system() -> dict:
    import torch
    running = [j for j in jobs().list() if j["status"] in ("queued", "running")]
    return {
        "version": __version__, "python": platform.python_version(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "cpus": os.cpu_count(), "memory": memory(),
        "uptime_s": round(time.time() - STARTED, 1), "in_docker": Path("/.dockerenv").exists(),
        "paths": {"data": str(paths.DATA_DIR), "artifacts": str(paths.ARTIFACTS_DIR), "runtime": str(paths.RUNTIME_DIR)},
        "jobs_active": running, "cached_models": len(service().bundles._items),
    }


@router.get("/overview")
def overview() -> dict:
    from ...experiments.results import list_results
    datasets = []
    for key, spec in DATASETS.items():
        loader = DatasetLoader(key)
        man = loader.manifest()
        datasets.append({"key": key, "name": spec.name, "description": spec.description, "citation": spec.citation,
                         "license": spec.license, "url": spec.url, "subjects_total": len(spec.subjects),
                         "subjects_ready": len(loader.available_subjects()),
                         "hours": round(sum(s["duration_s"] for s in man.get("subjects", [])) / 3600, 2),
                         "size_mb": round(sum(s["bytes"] for s in man.get("subjects", [])) / 1e6, 1)})
    return {"tasks": service().tasks(), "datasets": datasets,
            "modalities": {k: {"label": v.label, "unit": v.unit, "native_hz": v.native_hz, "channels": v.channels}
                           for k, v in MODALITY_SPECS.items()},
            "results": list_results()}
