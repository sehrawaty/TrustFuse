"""AttackSimulator: the Selective Modality Suppression Attack (SMSA).

For a target modality m the transform A_m modifies only x_m; every untargeted stream x_j
(j != m) is passed through *by identity* (same array object), which the test-suite checks.

Attacks operate on synchronized windows in physical units, i.e. at the attack surface between
pre-processing and the modality encoders (Fig. 2 of the report).

Window-level primitives
    mask       contiguous segment covering ``severity`` x window length is replaced by a fill
               (``zero`` = sensor reports zeros, ``missing`` = NaN / dropped samples,
               ``hold`` = sample-and-hold of the last valid value, i.e. a frozen sensor).
               severity = 1.0 is full masking.
    attenuate  amplitude scaling by (1 - severity), i.e. partial suppression.
    noise      additive Gaussian corruption at a given SNR (dB) — extension.
Stream-level primitive
    burst      the target stream is suppressed for a contiguous interval spanning B consecutive
               windows and then restored.
Attacker strategies that need model access (random / targeted / oracle / adaptive) live in
``attacks.strategies``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

WINDOW_ATTACKS = ("mask", "attenuate", "noise")
STRATEGY_ATTACKS = ("random_drop", "targeted", "oracle", "adaptive")
ATTACK_TYPES = ("none",) + WINDOW_ATTACKS + ("burst",) + STRATEGY_ATTACKS
FILLS = ("zero", "missing", "hold")
PLACEMENTS = ("random", "start", "center", "end")


@dataclass
class AttackSpec:
    type: str = "none"
    modality: str | None = None
    severity: float = 1.0
    fill: str = "zero"
    snr_db: float = 10.0
    burst_windows: int = 1
    placement: str = "random"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.type not in ATTACK_TYPES:
            raise ValueError(f"unknown attack type {self.type}")
        if self.fill not in FILLS:
            raise ValueError(f"unknown fill {self.fill}")
        if self.placement not in PLACEMENTS:
            raise ValueError(f"unknown placement {self.placement}")
        if not 0.0 <= float(self.severity) <= 1.0:
            raise ValueError("severity must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        if self.type == "none":
            return "clean"
        if self.type == "mask":
            return f"mask[{self.fill}] {self.modality} {self.severity:.0%}"
        if self.type == "attenuate":
            return f"attenuate {self.modality} {self.severity:.0%}"
        if self.type == "noise":
            return f"noise {self.modality} {self.snr_db:g} dB"
        if self.type == "burst":
            return f"burst[{self.fill}] {self.modality} x{self.burst_windows}"
        return f"{self.type}"


class AttackSimulator:
    def __init__(self, modalities: tuple[str, ...], seed: int = 0):
        self.modalities = tuple(modalities)
        self.rng = np.random.default_rng(seed)

    # -- primitives on one modality: x has shape (B, C, L) ---------------------------------
    @staticmethod
    def _segment(length: int, severity: float, batch: int, placement: str,
                 rng: np.random.Generator) -> tuple[np.ndarray, int]:
        n = int(round(severity * length))
        if severity > 0:
            n = max(n, 1)
        n = min(n, length)
        if placement == "start" or n == length:
            off = np.zeros(batch, dtype=np.int64)
        elif placement == "end":
            off = np.full(batch, length - n, dtype=np.int64)
        elif placement == "center":
            off = np.full(batch, (length - n) // 2, dtype=np.int64)
        else:
            off = rng.integers(0, length - n + 1, size=batch)
        return off, n

    def mask(self, x: np.ndarray, severity: float, fill: str = "zero", placement: str = "random",
             rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        out = np.array(x, dtype=np.float32, copy=True)
        b, _, length = out.shape
        off, n = self._segment(length, severity, b, placement, rng)
        if n == 0:
            return out
        idx = off[:, None] + np.arange(n)[None, :]                    # (B, n)
        rows = np.arange(b)[:, None]
        if fill == "zero":
            values = np.zeros((b, out.shape[1], 1), dtype=np.float32)
        elif fill == "missing":
            values = np.full((b, out.shape[1], 1), np.nan, dtype=np.float32)
        else:  # hold: freeze at the last valid sample before the segment (or after it)
            before = np.clip(off - 1, 0, length - 1)
            after = np.clip(off + n, 0, length - 1)
            src = np.where(off > 0, before, after)
            values = x[np.arange(b), :, src][:, :, None].astype(np.float32)
        seg = np.broadcast_to(values, (b, out.shape[1], n))
        out_t = out.transpose(0, 2, 1)                                  # (B, L, C) view
        out_t[rows, idx, :] = seg.transpose(0, 2, 1)
        return out

    @staticmethod
    def attenuate(x: np.ndarray, severity: float) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) * np.float32(1.0 - severity)).astype(np.float32)

    def add_noise(self, x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        x = np.asarray(x, dtype=np.float32)
        sig = np.std(x, axis=-1, keepdims=True)
        noise_std = sig / (10.0 ** (snr_db / 20.0))
        return (x + rng.standard_normal(x.shape).astype(np.float32) * noise_std).astype(np.float32)

    # -- batch-level (dict of modalities) -----------------------------------------------------
    def apply(self, batch: dict[str, np.ndarray], spec: AttackSpec,
              rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
        """Apply a single-target window-level attack. Untargeted modalities are returned as the
        very same array objects."""
        if spec.type == "none":
            return dict(batch)
        if spec.type not in WINDOW_ATTACKS:
            raise ValueError(f"{spec.type} is not a window-level attack; use attacks.strategies")
        if spec.modality not in batch:
            raise KeyError(f"target modality {spec.modality} not in batch {list(batch)}")
        rng = rng or np.random.default_rng(spec.seed)
        out = dict(batch)
        x = batch[spec.modality]
        if spec.type == "mask":
            out[spec.modality] = self.mask(x, spec.severity, spec.fill, spec.placement, rng)
        elif spec.type == "attenuate":
            out[spec.modality] = self.attenuate(x, spec.severity)
        else:
            out[spec.modality] = self.add_noise(x, spec.snr_db, rng)
        return out

    def drop_modality(self, batch: dict[str, np.ndarray], modality: str, fill: str = "zero") -> dict:
        return self.apply(batch, AttackSpec(type="mask", modality=modality, severity=1.0, fill=fill,
                                            placement="start"))

    def random_drop(self, batch: dict[str, np.ndarray], fill: str = "zero",
                    rng: np.random.Generator | None = None) -> tuple[dict, np.ndarray]:
        """Benign-like baseline: each window loses one uniformly chosen modality."""
        rng = rng or self.rng
        b = next(iter(batch.values())).shape[0]
        choice = rng.integers(0, len(self.modalities), size=b)
        out = dict(batch)
        for k, m in enumerate(self.modalities):
            sel = choice == k
            if sel.any():
                x = np.array(out[m], copy=True)
                x[sel] = self.mask(batch[m][sel], 1.0, fill, "start", rng)
                out[m] = x
        return out, np.asarray(self.modalities)[choice]

    # -- stream-level burst ---------------------------------------------------------------------
    @staticmethod
    def burst_drop(signals: dict[str, np.ndarray], modality: str, start_s: float, duration_s: float,
                   hz: int, fill: str = "zero") -> dict[str, np.ndarray]:
        """Suppress ``modality`` of a continuous stream on [start_s, start_s + duration_s) and
        restore it afterwards. Returns a new dict; untargeted streams are shared by identity."""
        out = dict(signals)
        x = np.array(signals[modality], dtype=np.float32, copy=True)
        a = max(0, int(round(start_s * hz)))
        b = min(x.shape[-1], int(round((start_s + duration_s) * hz)))
        if b > a:
            if fill == "zero":
                x[:, a:b] = 0.0
            elif fill == "missing":
                x[:, a:b] = np.nan
            else:
                ref = x[:, a - 1] if a > 0 else x[:, min(b, x.shape[-1] - 1)]
                x[:, a:b] = ref[:, None]
        out[modality] = x
        return out
