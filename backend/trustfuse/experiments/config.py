"""ExperimentConfig: versioned, hashable experiment definition (YAML <-> pydantic)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from ..specs import MODALITIES, TASKS

FUSION_VARIANTS = ("fusion", "moddrop")
UNIMODAL_VARIANTS = tuple(f"uni_{m}" for m in MODALITIES)
ALL_VARIANTS = FUSION_VARIANTS + UNIMODAL_VARIANTS


class TrainSettings(BaseModel):
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 7
    aux_weight: float = 0.5
    moddrop_p: float = 0.5


class AttackGrid(BaseModel):
    """Suppression matrix. Final settings are fixed here *before* test evaluation."""

    severities: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    mask_fills: list[Literal["zero", "missing", "hold"]] = ["zero", "missing", "hold"]
    attenuations: list[float] = [0.1, 0.2, 0.4, 0.6, 0.8]
    noise_snr_db: list[float] = [20.0, 10.0, 5.0, 0.0]
    burst_windows: list[int] = [1, 2, 5, 10]
    burst_fill: Literal["zero", "missing", "hold"] = "zero"
    burst_placements_per_subject: int = 12
    strategies: list[Literal["random_drop", "targeted", "oracle", "adaptive"]] = [
        "random_drop", "targeted", "oracle", "adaptive"]
    strategy_fill: Literal["zero", "missing", "hold"] = "zero"
    adaptive_severities: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    placement: Literal["random", "start", "center", "end"] = "random"
    seed: int = 1234


class ReliabilitySettings(BaseModel):
    components: list[Literal["Q", "D", "U"]] = ["Q", "D", "U"]
    mc_samples: int = 8
    alpha: float = 0.05            # target clean false-rejection rate of the low-confidence flag
    gate_span: float = 1.0         # gate closes when anomaly exceeds the threshold by span x (thr - median)
    strategy: Literal["embedding_gate", "poe"] = "embedding_gate"
    ablations: list[list[Literal["Q", "D", "U"]]] = [["Q"], ["D"], ["U"], ["Q", "D"], ["Q", "U"], ["D", "U"]]


class StatsSettings(BaseModel):
    alpha: float = 0.05
    n_boot: int = 2000
    noninferiority_margin: float = 0.02   # H3: max acceptable clean macro-F1 loss of the defense
    frr_max: float = 0.10                 # H4: max acceptable clean false-rejection rate of the flag
    primary_fill: Literal["zero", "missing", "hold"] = "zero"
    primary_severity: float = 1.0
    seed: int = 7


class ExperimentConfig(BaseModel):
    name: str
    task: str
    description: str = ""
    split: Literal["loso", "holdout"] = "loso"
    test_subjects: list[str] | None = None     # None -> every subject (full LOSO)
    n_val: int = 2
    variants: list[str] = list(ALL_VARIANTS)
    seeds: list[int] = [0, 1, 2]               # seeds for the fusion model
    aux_seeds: list[int] = [0]                 # seeds for ModDrop / unimodal baselines
    eval_variants: list[str] = list(FUSION_VARIANTS)
    eval_seeds: list[int] = [0, 1, 2]
    manifest_seed: int = 0                     # attacked-window manifest written for this seed
    train: TrainSettings = Field(default_factory=TrainSettings)
    attacks: AttackGrid = Field(default_factory=AttackGrid)
    reliability: ReliabilitySettings = Field(default_factory=ReliabilitySettings)
    stats: StatsSettings = Field(default_factory=StatsSettings)
    models_root: Literal["artifacts", "runtime"] = "artifacts"
    results_root: Literal["artifacts", "runtime"] = "artifacts"

    @field_validator("task")
    @classmethod
    def _known_task(cls, v: str) -> str:
        if v not in TASKS:
            raise ValueError(f"unknown task {v}; expected one of {sorted(TASKS)}")
        return v

    @field_validator("variants", "eval_variants")
    @classmethod
    def _known_variants(cls, v: list[str]) -> list[str]:
        bad = [x for x in v if x not in ALL_VARIANTS]
        if bad:
            raise ValueError(f"unknown variants {bad}")
        return v

    # -- setters named in the class diagram -------------------------------------------------
    def set_attack_rates(self, severities: list[float], attenuations: list[float] | None = None) -> "ExperimentConfig":
        self.attacks.severities = sorted(severities)
        if attenuations is not None:
            self.attacks.attenuations = sorted(attenuations)
        return self

    def set_durations(self, burst_windows: list[int]) -> "ExperimentConfig":
        self.attacks.burst_windows = sorted(burst_windows)
        return self

    def set_seeds(self, seeds: list[int], aux_seeds: list[int] | None = None) -> "ExperimentConfig":
        self.seeds = list(seeds)
        if aux_seeds is not None:
            self.aux_seeds = list(aux_seeds)
        return self

    def set_splits(self, split: str, test_subjects: list[str] | None = None, n_val: int | None = None) -> "ExperimentConfig":
        self.split = split  # type: ignore[assignment]
        self.test_subjects = test_subjects
        if n_val is not None:
            self.n_val = n_val
        return self

    # -- identity ------------------------------------------------------------------------------
    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(exclude={"name", "description"}), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def seeds_for(self, variant: str) -> list[int]:
        return self.seeds if variant == "fusion" else self.aux_seeds

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)
