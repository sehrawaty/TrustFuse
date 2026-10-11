"""FusionModel (standard concatenation fusion) and unimodal baselines.

Each modality has a compact temporal encoder E_m producing an embedding h_m. The standard
(fixed) fusion path concatenates [h_1, ..., h_M] and applies a light classifier. Every modality
also has an auxiliary unimodal head g_m(h_m) trained jointly (deep supervision); these heads
expose per-modality evidence for the reliability engine (disagreement D_m and uncertainty U_m)
without changing the fixed-fusion decision path.

The reliability-aware defense re-uses the very same frozen checkpoint: it only replaces h_m by a
reliability-gated embedding  h~_m = w_m h_m + (1 - w_m) mu_m  before the classifier.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class ModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, length: int, emb_dim: int = 32,
                 widths: tuple[int, ...] = (16, 32, 48, 48, 48), dropout: float = 0.1):
        super().__init__()
        n_blocks = int(min(len(widths), max(2, math.ceil(math.log2(max(length, 2) / 12)))))
        layers: list[nn.Module] = []
        c = in_channels
        for i in range(n_blocks):
            k = 7 if i == 0 else 5
            layers += [nn.Conv1d(c, widths[i], kernel_size=k, stride=2, padding=k // 2, bias=False),
                       nn.BatchNorm1d(widths[i]), nn.ReLU(inplace=True)]
            c = widths[i]
        self.conv = nn.Sequential(*layers)
        self.feature_dropout = nn.Dropout(dropout)
        self.n_stats = 5 * in_channels
        self.proj = nn.Sequential(nn.Linear(2 * c + self.n_stats, emb_dim), nn.ReLU(inplace=True))
        self.emb_dim = emb_dim

    @staticmethod
    def window_stats(x: torch.Tensor) -> torch.Tensor:
        """Mean, std, min, max and linear slope of every channel (level information that global
        pooling of convolution features would otherwise blur, e.g. for EDA / temperature)."""
        t = torch.linspace(-1.0, 1.0, x.shape[-1], device=x.device)
        mean = x.mean(-1)
        std = x.std(-1, unbiased=False)
        slope = ((x - mean[..., None]) * t).mean(-1) / (t.pow(2).mean())
        return torch.cat([mean, std, x.amin(-1), x.amax(-1), slope], dim=-1)

    def pooled_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic part of the encoder (convolutions + pooling + window statistics)."""
        f = self.conv(x)
        return torch.cat([f.mean(-1), f.amax(-1)], dim=-1), self.window_stats(x)

    def embed(self, pooled: torch.Tensor, stats: torch.Tensor,
              dropout_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Stochastic part. ``dropout_mask`` lets MC-dropout use explicit, reproducible masks."""
        pooled = pooled * dropout_mask if dropout_mask is not None else self.feature_dropout(pooled)
        return self.proj(torch.cat([pooled, stats], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(*self.pooled_features(x))


class FusionModel(nn.Module):
    def __init__(self, modalities: tuple[str, ...], input_shapes: dict[str, tuple[int, int]],
                 n_classes: int, emb_dim: int = 32, hidden: int = 64, dropout: float = 0.3,
                 head_dropout: float = 0.2):
        super().__init__()
        self.modalities = tuple(modalities)
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.encoders = nn.ModuleDict({m: ModalityEncoder(input_shapes[m][0], input_shapes[m][1], emb_dim)
                                       for m in self.modalities})
        self.aux_heads = nn.ModuleDict({m: nn.Sequential(nn.Dropout(head_dropout), nn.Linear(emb_dim, n_classes))
                                        for m in self.modalities})
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(emb_dim * len(self.modalities), hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    # -- the three operations named in the class diagram --------------------------------------
    def encode_modalities(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.encoders[m](x[m]) for m in self.modalities}

    def fuse(self, h: dict[str, torch.Tensor], weights: dict[str, torch.Tensor] | None = None,
             mu: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        parts = []
        for m in self.modalities:
            hm = h[m]
            if weights is not None and m in weights:
                w = weights[m].reshape(-1, 1)
                ref = mu[m].reshape(1, -1) if mu is not None else torch.zeros_like(hm)
                hm = w * hm + (1.0 - w) * ref
            parts.append(hm)
        return torch.cat(parts, dim=-1)

    def predict(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.fuse(self.encode_modalities(x)))

    def forward(self, x: dict[str, torch.Tensor]):
        h = self.encode_modalities(x)
        logits = self.classifier(self.fuse(h))
        aux = {m: self.aux_heads[m](h[m]) for m in self.modalities}
        return logits, aux, h

    def classify(self, fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(fused)


class UnimodalModel(nn.Module):
    """Single-modality baseline used to quantify each sensor's stand-alone contribution."""

    def __init__(self, modality: str, input_shape: tuple[int, int], n_classes: int, emb_dim: int = 32,
                 hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.modality = modality
        self.modalities = (modality,)
        self.encoder = ModalityEncoder(input_shape[0], input_shape[1], emb_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: dict[str, torch.Tensor]):
        return self.head(self.encoder(x[self.modality])), {}, {}

    def predict(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(x)[0]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
