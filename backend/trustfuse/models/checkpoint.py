"""Checkpoint format: one file per (task, variant, seed, held-out subject)."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import numpy as np
import torch

from .. import paths
from ..data.preprocess import NormStats
from ..specs import get_task
from .fusion import FusionModel, UnimodalModel

FORMAT = "trustfuse-checkpoint-v1"


def checkpoint_path(root: Path, task: str, variant: str, seed: int, subject: str) -> Path:
    return root / task / variant / f"seed{seed}" / f"{subject}.pt"


def find_checkpoint(task: str, variant: str, seed: int, subject: str) -> Path | None:
    for root in paths.models_dirs():
        p = checkpoint_path(root, task, variant, seed, subject)
        if p.exists():
            return p
    return None


def build_model(task_key: str, variant: str) -> torch.nn.Module:
    task = get_task(task_key)
    if variant.startswith("uni_"):
        m = variant[4:]
        return UnimodalModel(m, task.input_shapes()[m], task.n_classes)
    return FusionModel(task.modalities, task.input_shapes(), task.n_classes)


def save_checkpoint(path: Path, model: torch.nn.Module, *, task: str, variant: str, seed: int, subject: str,
                    split: dict, norm: NormStats, train_cfg: dict, history: list[dict], val_f1: float,
                    config_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({
        "format": FORMAT, "task": task, "variant": variant, "seed": seed, "subject": subject,
        "split": split, "norm": norm.to_dict(), "train_cfg": train_cfg, "history": history,
        "val_f1": val_f1, "config_hash": config_hash, "state_dict": state,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
    }, path)


def load_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported checkpoint format {ckpt.get('format')}")
    model = build_model(ckpt["task"], ckpt["variant"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt
