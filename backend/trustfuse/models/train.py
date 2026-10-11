"""Deterministic training with early stopping on validation macro-F1."""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from ..evaluation.metrics import accuracy, macro_f1


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 7
    aux_weight: float = 0.5
    moddrop_p: float = 0.0      # > 0: modality-dropout training (ModDrop robustness baseline)
    seed: int = 0
    threads: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class Cancelled(Exception):
    pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(x: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {m: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for m, v in x.items()}


@torch.no_grad()
def predict_logits(model: nn.Module, x: dict[str, np.ndarray], batch_size: int = 512) -> np.ndarray:
    model.eval()
    n = next(iter(x.values())).shape[0]
    out = []
    for i in range(0, n, batch_size):
        xb = to_tensors({m: v[i:i + batch_size] for m, v in x.items()})
        out.append(model(xb)[0].numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def class_weights(y: np.ndarray, k: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=k).astype(float)
    w = np.where(counts > 0, counts.sum() / (k * np.maximum(counts, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


def train_model(model: nn.Module, train: tuple[dict, np.ndarray], val: tuple[dict, np.ndarray], n_classes: int,
                cfg: TrainConfig, on_epoch: Callable[[dict], None] | None = None,
                should_stop: Callable[[], bool] | None = None) -> tuple[nn.Module, list[dict]]:
    torch.set_num_threads(max(1, cfg.threads))
    set_seed(cfg.seed)
    x_train, y_train = train
    x_val, y_val = val
    modalities = list(x_train)
    xt = to_tensors(x_train)
    yt = torch.from_numpy(y_train.astype(np.int64))
    criterion = nn.CrossEntropyLoss(weight=class_weights(y_train, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    gen = torch.Generator().manual_seed(cfg.seed)
    best_state, best_f1, best_epoch, history = None, -1.0, -1, []
    n = yt.shape[0]
    for epoch in range(cfg.epochs):
        if should_stop and should_stop():
            raise Cancelled()
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        total, batches = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if idx.numel() < 2:
                continue
            xb = {m: xt[m][idx] for m in modalities}
            if cfg.moddrop_p > 0 and len(modalities) > 1:
                drop = torch.rand(idx.numel(), generator=gen) < cfg.moddrop_p
                which = torch.randint(0, len(modalities), (idx.numel(),), generator=gen)
                for k, m in enumerate(modalities):
                    sel = drop & (which == k)
                    if sel.any():
                        xb[m] = xb[m].clone()
                        xb[m][sel] = 0.0   # normalized zero == "missing" (training-mean) input
            logits, aux, _ = model(xb)
            yb = yt[idx]
            loss = criterion(logits, yb)
            if aux and cfg.aux_weight > 0:
                loss = loss + cfg.aux_weight * torch.stack([criterion(a, yb) for a in aux.values()]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach())
            batches += 1
        sched.step()
        pred = predict_logits(model, x_val).argmax(1)
        f1 = macro_f1(y_val, pred, n_classes)
        record = {"epoch": epoch + 1, "loss": total / max(batches, 1), "val_f1": f1,
                  "val_acc": accuracy(y_val, pred), "seconds": time.time() - t0}
        history.append(record)
        if on_epoch:
            on_epoch(record)
        if f1 > best_f1 + 1e-6:
            best_f1, best_epoch = f1, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history
