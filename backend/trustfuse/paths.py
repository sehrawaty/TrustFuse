"""Filesystem locations. Every path can be overridden with an environment variable so the
same code runs from a checkout and inside the Docker image."""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).resolve() if value else default


REPO_ROOT = _env_path("TRUSTFUSE_ROOT", Path(__file__).resolve().parents[2])
DATA_DIR = _env_path("TRUSTFUSE_DATA_DIR", REPO_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = _env_path("TRUSTFUSE_ARTIFACTS_DIR", REPO_ROOT / "artifacts")
CONFIGS_DIR = _env_path("TRUSTFUSE_CONFIGS_DIR", REPO_ROOT / "configs")
# Writable state of the running application (user-launched runs, job database, caches).
RUNTIME_DIR = _env_path("TRUSTFUSE_RUNTIME_DIR", REPO_ROOT / ".runtime")


def models_dirs() -> list[Path]:
    """Model roots searched in order: shipped (canonical) first, then runtime (user runs)."""
    return [ARTIFACTS_DIR / "models", RUNTIME_DIR / "models"]


def results_dirs() -> list[Path]:
    return [ARTIFACTS_DIR / "results", RUNTIME_DIR / "results"]
