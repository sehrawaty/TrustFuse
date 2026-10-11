"""Conversion of the original dataset archives into the compact processed format.

Each subject becomes one ``.npz`` file that keeps the wrist streams at their *native*
sampling rates (no information is discarded):

* ``bvp``   float32 (1, N)   64 Hz
* ``acc``   int8    (3, N)   32 Hz, counts of 1/64 g (the E4's native resolution)
* ``eda``   float32 (1, N)   4 Hz, µS
* ``temp``  float32 (1, N)   4 Hz, °C
* ``label`` int8    (N,)     4 Hz (WESAD protocol condition / DaLiA activity)
* ``hr``    float32 (K,)     DaLiA only: ECG reference heart rate, 8 s windows, 2 s shift

Subjects are read one at a time straight from the zip archive so peak memory stays near
the size of a single subject pickle.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..specs import DATASETS, LABEL_HZ

Progress = Callable[[str], None]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _acc_counts(acc: np.ndarray, unit_scale: float) -> np.ndarray:
    counts = np.rint(np.asarray(acc, dtype=np.float64) * unit_scale)
    if np.abs(counts - np.asarray(acc) * unit_scale).max() > 1e-6:
        raise ValueError("accelerometer values are not on the 1/64 g grid")
    return np.clip(counts, -128, 127).astype(np.int8).T.copy()


def _wrist_arrays(wrist: dict, acc_scale: float) -> dict[str, np.ndarray]:
    return {
        "bvp": np.asarray(wrist["BVP"], dtype=np.float32).reshape(1, -1),
        "acc": _acc_counts(wrist["ACC"], acc_scale),
        "eda": np.asarray(wrist["EDA"], dtype=np.float32).reshape(1, -1),
        "temp": np.asarray(wrist["TEMP"], dtype=np.float32).reshape(1, -1),
    }


def _crop_to_common_duration(arrays: dict[str, np.ndarray], label: np.ndarray) -> tuple[dict, np.ndarray, float]:
    rates = {"bvp": 64, "acc": 32, "eda": 4, "temp": 4}
    seconds = min(min(a.shape[-1] / rates[k] for k, a in arrays.items()), len(label) / LABEL_HZ)
    n4 = int(np.floor(seconds * LABEL_HZ))
    duration = n4 / LABEL_HZ
    cropped = {k: a[..., : int(round(duration * rates[k]))] for k, a in arrays.items()}
    return cropped, label[:n4], duration


def _write_subject(out_dir: Path, subject: str, arrays: dict, label: np.ndarray, extra: dict) -> dict:
    path = out_dir / f"{subject}.npz"
    np.savez_compressed(path, label=label.astype(np.int8), **arrays, **extra)
    values, counts = np.unique(label, return_counts=True)
    return {
        "subject": subject,
        "file": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "duration_s": float(label.shape[0] / LABEL_HZ),
        "label_seconds": {int(v): float(c / LABEL_HZ) for v, c in zip(values, counts)},
    }


def convert_wesad(zip_path: Path, out_dir: Path, subjects: list[str] | None = None,
                  progress: Progress = print) -> list[dict]:
    """WESAD.zip (official archive) -> data/processed/wesad/S*.npz."""
    spec = DATASETS["wesad"]
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(zip_path) as zf:
        for subject in subjects or spec.subjects:
            t0 = time.time()
            with zf.open(f"WESAD/{subject}/{subject}.pkl") as f:
                raw = pickle.load(f, encoding="latin1")
            arrays = _wrist_arrays(raw["signal"]["wrist"], acc_scale=1.0)  # already 1/64 g counts
            label700 = np.asarray(raw["label"]).astype(np.int16)
            del raw
            # 700 Hz -> 4 Hz: take the label at the centre of every 0.25 s bin.
            n4 = int(len(label700) // 175)
            centres = np.arange(n4) * 175 + 87
            label = label700[centres].astype(np.int8)
            arrays, label, _ = _crop_to_common_duration(arrays, label)
            entries.append(_write_subject(out_dir, subject, arrays, label, {}))
            progress(f"wesad {subject}: {entries[-1]['duration_s']:.0f}s -> "
                     f"{entries[-1]['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
    return entries


def convert_dalia(zip_path: Path, out_dir: Path, subjects: list[str] | None = None,
                  progress: Progress = print) -> list[dict]:
    """PPG-DaLiA ``data.zip`` (inner archive of the UCI download) -> data/processed/dalia/S*.npz."""
    spec = DATASETS["dalia"]
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(zip_path) as zf:
        for subject in subjects or spec.subjects:
            t0 = time.time()
            with zf.open(f"PPG_FieldStudy/{subject}/{subject}.pkl") as f:
                raw = pickle.load(f, encoding="latin1")
            arrays = _wrist_arrays(raw["signal"]["wrist"], acc_scale=64.0)  # stored in g
            label = np.asarray(raw["activity"]).reshape(-1).astype(np.int8)
            hr = np.asarray(raw["label"], dtype=np.float32).reshape(-1)
            del raw
            arrays, label, _ = _crop_to_common_duration(arrays, label)
            entries.append(_write_subject(out_dir, subject, arrays, label, {"hr": hr}))
            progress(f"dalia {subject}: {entries[-1]['duration_s']:.0f}s -> "
                     f"{entries[-1]['bytes'] / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
    return entries


def write_manifest(dataset: str, out_dir: Path, entries: list[dict], source: Path) -> Path:
    spec = DATASETS[dataset]
    manifest = {
        "dataset": spec.name,
        "key": spec.key,
        "format": "trustfuse-processed-v1",
        "source_archive": source.name,
        "source_url": spec.url,
        "license": spec.license,
        "citation": spec.citation,
        "streams": {
            "bvp": {"hz": 64, "dtype": "float32", "unit": "a.u."},
            "acc": {"hz": 32, "dtype": "int8", "unit": "1/64 g"},
            "eda": {"hz": 4, "dtype": "float32", "unit": "µS"},
            "temp": {"hz": 4, "dtype": "float32", "unit": "°C"},
            "label": {"hz": LABEL_HZ, "dtype": "int8", "names": spec.label_names},
        },
        "subjects": sorted(entries, key=lambda e: int(e["subject"][1:])),
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return path


def verify_manifest(out_dir: Path) -> list[str]:
    """Return a list of problems (empty when every processed file matches its checksum)."""
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    problems = []
    for entry in json.loads(manifest_path.read_text())["subjects"]:
        path = out_dir / entry["file"]
        if not path.exists():
            problems.append(f"missing {path}")
        elif _sha256(path) != entry["sha256"]:
            problems.append(f"checksum mismatch {path}")
    return problems
