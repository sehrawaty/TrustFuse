"""FoldBundle: a frozen fold model together with its normalization and frozen calibration."""

from __future__ import annotations

import io
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.preprocess import NormStats
from ..data.windows import gather
from ..models.checkpoint import find_checkpoint, load_checkpoint
from ..reliability.engine import ReliabilityEngine
from ..specs import TaskSpec, get_task

CALIB_VERSION = 2


def release_memory() -> None:
    """Return freed heap memory to the OS (keeps the API inside its container memory limit)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def calibration_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + ".calib.npz")


@dataclass
class FoldBundle:
    task: TaskSpec
    variant: str
    seed: int
    subject: str
    path: Path
    model: object
    ckpt: dict
    norm: NormStats
    engine: ReliabilityEngine | None

    @property
    def is_fusion(self) -> bool:
        return self.engine is not None

    def meta(self) -> dict:
        c = self.ckpt
        return {"task": self.task.key, "variant": self.variant, "seed": self.seed, "subject": self.subject,
                "split": c["split"], "val_f1": c["val_f1"], "epochs": len(c["history"]),
                "created": c["created"], "config_hash": c["config_hash"], "path": str(self.path),
                "versions": c.get("versions", {})}


def _save_calibration(path: Path, engine: ReliabilityEngine) -> None:
    state = engine.state()
    val_z = state.pop("val_z")
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=np.frombuffer(json.dumps({"version": CALIB_VERSION, **state}).encode(), dtype=np.uint8),
                        **{f"val_z__{m}": np.asarray(v, dtype=np.float32) for m, v in val_z.items()})
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def _load_calibration(path: Path) -> dict | None:
    try:
        with np.load(path) as z:
            state = json.loads(bytes(z["meta"]).decode())
            if state.get("version") != CALIB_VERSION:
                return None
            state["val_z"] = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("val_z__")}
        return state
    except (OSError, KeyError, ValueError):
        return None


def load_bundle(task_key: str, variant: str, seed: int, subject: str, mc_samples: int = 8,
                calibrate: bool = True, persist: bool = True, path: Path | None = None) -> FoldBundle:
    task = get_task(task_key)
    path = path or find_checkpoint(task_key, variant, seed, subject)
    if path is None:
        raise FileNotFoundError(f"no checkpoint for {task_key}/{variant}/seed{seed}/{subject}")
    model, ckpt = load_checkpoint(path)
    norm = NormStats.from_dict(ckpt["norm"])
    engine = None
    if not variant.startswith("uni_"):
        engine = ReliabilityEngine(model, task, norm, mc_samples=mc_samples)
        if calibrate:
            cpath = calibration_path(path)
            state = _load_calibration(cpath) if cpath.exists() else None
            if state is not None and state.get("mc_samples") == mc_samples:
                engine.load_state(state)
            else:
                # clean data of this fold's non-test subjects only; never the test subject
                val = gather(task_key, ckpt["split"]["val"])
                ref = gather(task_key, ckpt["split"]["train"] + ckpt["split"]["val"], max_per_subject=300)
                engine.calibrate(val.phys, ref_phys=ref.phys)
                del ref, val
                release_memory()
                if persist:
                    try:
                        _save_calibration(cpath, engine)
                    except OSError:
                        pass   # read-only model directory (e.g. shipped artifacts in Docker)
    return FoldBundle(task, variant, seed, subject, path, model, ckpt, norm, engine)


class BundleCache:
    """Small LRU cache so the API keeps only a handful of fold models in memory."""

    def __init__(self, capacity: int = 6):
        self.capacity = capacity
        self._items: OrderedDict[tuple, FoldBundle] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, task: str, variant: str, seed: int, subject: str) -> FoldBundle:
        key = (task, variant, seed, subject)
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        bundle = load_bundle(task, variant, seed, subject)
        with self._lock:
            self._items[key] = bundle
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return bundle

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
