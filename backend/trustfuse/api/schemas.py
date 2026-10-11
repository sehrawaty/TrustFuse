from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AttackIn(BaseModel):
    type: Literal["none", "mask", "attenuate", "noise"] = "none"
    modality: str | None = None
    severity: float = Field(1.0, ge=0.0, le=1.0)
    fill: Literal["zero", "missing", "hold"] = "zero"
    snr_db: float = 10.0
    placement: Literal["random", "start", "center", "end"] = "center"
    seed: int = 0


class InferIn(BaseModel):
    task: str
    subject: str
    window: int = 0
    variant: Literal["fusion", "moddrop"] = "fusion"
    seed: int = 0
    attack: AttackIn = Field(default_factory=AttackIn)
    components: list[Literal["Q", "D", "U"]] = ["Q", "D", "U"]
    calibration: Literal["validation", "pooled"] = "validation"
    strategy: Literal["embedding_gate", "poe"] = "embedding_gate"


class WindowIn(BaseModel):
    task: str
    subject: str
    window: int = 0
    variant: Literal["fusion", "moddrop"] = "fusion"
    seed: int = 0
    fill: Literal["zero", "missing", "hold"] = "zero"


class BurstIn(BaseModel):
    modality: str
    start_index: int = 0
    length: int = Field(5, ge=1, le=200)
    fill: Literal["zero", "missing", "hold"] = "zero"


class TimelineIn(BaseModel):
    task: str
    subject: str
    variant: Literal["fusion", "moddrop"] = "fusion"
    seed: int = 0
    burst: BurstIn | None = None


class JobIn(BaseModel):
    preset: str | None = None
    config: dict | None = None
    overrides: dict | None = None
