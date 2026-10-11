"""Process-wide singletons (created lazily so importing the app stays cheap)."""

from __future__ import annotations

from functools import lru_cache

from ..experiments.jobs import JobManager
from ..service.inference import InferenceService


@lru_cache(maxsize=1)
def service() -> InferenceService:
    return InferenceService(capacity=6)


@lru_cache(maxsize=1)
def jobs() -> JobManager:
    return JobManager()
