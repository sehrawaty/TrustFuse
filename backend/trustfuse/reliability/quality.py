"""Model-free signal-quality evidence Q_m.

Each modality window is summarised by interpretable sensor-quality indicators drawn from
wearable signal-quality practice. Indicator sets are fixed per modality from physical
reasoning (not tuned on test data):

    nan_frac   share of missing samples (dropped packets)                        higher = worse
    zero_frac  share of samples that are exactly zero (contact loss / zero fill) higher = worse
    flat       share of consecutive samples that do not change                    higher = worse
    max_run    longest run of identical samples, as a share of the window         higher = worse
    implaus    share of samples outside physical plausibility bounds              higher = worse
               (skin temperature 20-42 °C, EDA 0-100 µS, |ACC| 0.3-3.5 g)
    level      window mean (temperature °C, ACC magnitude g)                       two-sided
    log_std    log within-window standard deviation (BVP)                          lower = worse
    sqi        BVP spectral quality: share of power in the cardiac band 0.7-3.5 Hz lower = worse

Indicators are compared with a clean reference distribution through directional robust
z-scores (median / IQR-scaled). Because Q is model-free its reference pools every non-test
subject (training + validation), which captures inter-subject variability. The Q anomaly of a
window is its largest directional z-score.
"""

from __future__ import annotations

import numpy as np

FEATURES: dict[str, tuple[str, ...]] = {
    "bvp": ("nan_frac", "zero_frac", "flat", "max_run", "log_std", "sqi"),
    "acc": ("nan_frac", "zero_frac", "flat", "max_run", "implaus", "level"),
    "eda": ("nan_frac", "zero_frac", "flat", "max_run", "implaus"),
    "temp": ("nan_frac", "zero_frac", "flat", "max_run", "implaus", "level"),
}
BOUNDS: dict[str, tuple[float, float]] = {"acc": (0.3, 3.5), "eda": (0.0, 100.0), "temp": (20.0, 42.0)}
DIRECTION = {"nan_frac": 1, "zero_frac": 1, "flat": 1, "max_run": 1, "implaus": 1, "level": 0,
             "log_std": -1, "sqi": -1}
# Minimum robust scale per indicator, so degenerate clean distributions (e.g. nan_frac == 0
# everywhere) do not produce infinite z-scores.
SCALE_FLOOR = {"nan_frac": 0.02, "zero_frac": 0.02, "flat": 0.03, "max_run": 0.03, "implaus": 0.02,
               "level": 0.05, "log_std": 0.1, "sqi": 0.03}
Z_CAP = 50.0


def feature_names(modality: str) -> tuple[str, ...]:
    return FEATURES[modality]


def _series(modality: str, x: np.ndarray) -> np.ndarray:
    """Reduce (B, C, L) to the 1-D series the indicators are computed on: the ACC vector
    magnitude (orientation independent) or the single channel otherwise."""
    if modality == "acc":
        return np.sqrt(np.sum(np.square(x), axis=1))
    return x[:, 0, :]


def _max_run(same: np.ndarray) -> np.ndarray:
    """Longest run of True per row (vectorised run-length on a boolean matrix)."""
    b, n = same.shape
    if n == 0:
        return np.zeros(b)
    run = np.zeros(b)
    best = np.zeros(b)
    for j in range(n):           # n is at most a few hundred samples
        run = np.where(same[:, j], run + 1, 0)
        best = np.maximum(best, run)
    return best


def raw_features(modality: str, x: np.ndarray, hz: int) -> np.ndarray:
    s = _series(modality, np.asarray(x, dtype=np.float64))          # (B, L)
    nan = np.isnan(s)
    counts = (~nan).sum(axis=1, keepdims=True)
    row_mean = np.where(nan, 0.0, s).sum(axis=1, keepdims=True) / np.maximum(counts, 1)
    filled = np.where(nan, row_mean, s)
    length = s.shape[1]
    cols: dict[str, np.ndarray] = {"nan_frac": nan.mean(axis=1)}
    cols["zero_frac"] = np.where(nan, 0.0, np.abs(s) < 1e-9).mean(axis=1)
    same = np.abs(np.diff(filled, axis=1)) <= 1e-9
    same |= nan[:, 1:] | nan[:, :-1]
    cols["flat"] = same.mean(axis=1)
    cols["max_run"] = (_max_run(same) + 1) / length
    if modality in BOUNDS:
        lo, hi = BOUNDS[modality]
        cols["implaus"] = np.where(nan, 0.0, (filled < lo) | (filled > hi)).mean(axis=1)
    cols["level"] = filled.mean(axis=1)
    cols["log_std"] = np.log(filled.std(axis=1) + 1e-6)
    if modality == "bvp":
        centred = filled - filled.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(centred, axis=1)) ** 2
        freqs = np.fft.rfftfreq(length, d=1.0 / hz)
        band = (freqs >= 0.7) & (freqs <= 3.5)
        total = (freqs >= 0.1) & (freqs <= min(8.0, hz / 2))
        p_total = spec[:, total].sum(axis=1)
        cols["sqi"] = np.where(p_total > 1e-12, spec[:, band].sum(axis=1) / np.maximum(p_total, 1e-12), 0.0)
    return np.stack([cols[n] for n in FEATURES[modality]], axis=1)


class QualityCalibration:
    """Clean reference for one modality."""

    def __init__(self, modality: str, hz: int):
        self.modality = modality
        self.hz = hz
        self.names = feature_names(modality)
        self.median = np.zeros(len(self.names))
        self.scale = np.ones(len(self.names))

    def fit(self, x_clean: np.ndarray) -> "QualityCalibration":
        f = raw_features(self.modality, x_clean, self.hz)
        self.median = np.median(f, axis=0)
        iqr = np.subtract(*np.percentile(f, [75, 25], axis=0))
        floors = np.array([SCALE_FLOOR[n] for n in self.names])
        self.scale = np.maximum(iqr / 1.349, floors)
        return self

    def zscores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = raw_features(self.modality, x, self.hz)
        z = (f - self.median) / self.scale
        directional = np.empty_like(z)
        for i, n in enumerate(self.names):
            d = DIRECTION[n]
            directional[:, i] = np.abs(z[:, i]) if d == 0 else d * z[:, i]
        return f, np.clip(directional, -Z_CAP, Z_CAP)

    def anomaly(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f, z = self.zscores(x)
        return np.clip(z.max(axis=1), 0.0, Z_CAP), f, z

    def to_dict(self) -> dict:
        return {"modality": self.modality, "hz": self.hz, "names": list(self.names),
                "bounds": list(BOUNDS[self.modality]) if self.modality in BOUNDS else None,
                "median": self.median.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "QualityCalibration":
        q = cls(d["modality"], d["hz"])
        q.median = np.asarray(d["median"])
        q.scale = np.asarray(d["scale"])
        return q
