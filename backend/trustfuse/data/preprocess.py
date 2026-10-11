"""Preprocessor: alignment, resampling, windowing and train-fold normalization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.signal import resample_poly

from ..specs import LABEL_HZ, TaskSpec, get_task
from .loader import DatasetLoader, SubjectRecord


@dataclass
class WindowIndex:
    start_s: np.ndarray   # int32, window start time in seconds (on the stride grid)
    y: np.ndarray         # int64 class index
    segment: np.ndarray   # int32, id of the contiguous label segment the window lies in

    def __len__(self) -> int:
        return int(self.start_s.shape[0])

    def subset(self, idx: np.ndarray) -> "WindowIndex":
        return WindowIndex(self.start_s[idx], self.y[idx], self.segment[idx])


@dataclass
class SubjectStream:
    """A subject's synchronized streams at model sampling rates, in physical units."""

    dataset: str
    subject: str
    task: str
    signals: dict[str, np.ndarray]    # modality -> float32 (C, N) at task.model_hz
    labels: np.ndarray                # 4 Hz class index, -1 = excluded
    raw_label: np.ndarray             # 4 Hz raw dataset label
    duration_s: float
    windows: WindowIndex


@dataclass
class NormStats:
    mean: dict[str, np.ndarray]   # modality -> (C,)
    std: dict[str, np.ndarray]

    def to_dict(self) -> dict:
        return {"mean": {m: v.tolist() for m, v in self.mean.items()},
                "std": {m: v.tolist() for m, v in self.std.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "NormStats":
        return cls(mean={m: np.asarray(v, dtype=np.float32) for m, v in d["mean"].items()},
                   std={m: np.asarray(v, dtype=np.float32) for m, v in d["std"].items()})


class Preprocessor:
    def __init__(self, task: TaskSpec):
        self.task = task

    # -- alignment / resampling -----------------------------------------------------------
    def resample(self, x: np.ndarray, native_hz: int, target_hz: int) -> np.ndarray:
        if native_hz == target_hz:
            return x.astype(np.float32, copy=True)
        g = np.gcd(native_hz, target_hz)
        return resample_poly(x, target_hz // g, native_hz // g, axis=-1).astype(np.float32)

    def align_signals(self, record: SubjectRecord) -> dict[str, np.ndarray]:
        synced = DatasetLoader.sync_modalities(record, self.task.modalities)
        out = {}
        for m in self.task.modalities:
            hz = self.task.model_hz[m]
            x = self.resample(synced[m], record.rate(m), hz)
            out[m] = x[:, : int(round(record.duration_s * hz))]
        return out

    # -- windowing ------------------------------------------------------------------------
    def window(self, labels: np.ndarray) -> WindowIndex:
        """Label-pure windows on a global stride grid (every sample in the window has the
        same valid class). Excluded labels (-1) break segments."""
        w4 = self.task.window_s * LABEL_HZ
        s4 = self.task.stride_s * LABEL_HZ
        starts, ys, segs = [], [], []
        change = np.flatnonzero(np.diff(labels)) + 1
        bounds = np.concatenate([[0], change, [len(labels)]])
        seg_id = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            cls = int(labels[a])
            if cls < 0 or b - a < w4:
                continue
            first = -(-a // s4) * s4
            grid = np.arange(first, b - w4 + 1, s4)
            if grid.size:
                starts.append(grid // LABEL_HZ)
                ys.append(np.full(grid.size, cls))
                segs.append(np.full(grid.size, seg_id))
                seg_id += 1
        if not starts:
            return WindowIndex(np.zeros(0, np.int32), np.zeros(0, np.int64), np.zeros(0, np.int32))
        return WindowIndex(np.concatenate(starts).astype(np.int32), np.concatenate(ys).astype(np.int64),
                           np.concatenate(segs).astype(np.int32))

    def prepare_subject(self, record: SubjectRecord) -> SubjectStream:
        labels = DatasetLoader.get_labels(record, self.task)
        return SubjectStream(dataset=record.dataset, subject=record.subject, task=self.task.key,
                             signals=self.align_signals(record), labels=labels, raw_label=record.label,
                             duration_s=record.duration_s, windows=self.window(labels))

    def extract(self, signals: dict[str, np.ndarray], start_s: np.ndarray,
                modalities: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """Cut windows -> {modality: float32 (B, C, L)} in physical units."""
        out = {}
        starts = np.asarray(start_s, dtype=np.int64)
        for m in modalities or self.task.modalities:
            hz = self.task.model_hz[m]
            length = self.task.window_s * hz
            idx = starts[:, None] * hz + np.arange(length)[None, :]
            out[m] = np.ascontiguousarray(signals[m][:, idx].transpose(1, 0, 2))
        return out

    # -- normalization --------------------------------------------------------------------
    def fit_normalizer(self, windows: dict[str, np.ndarray]) -> NormStats:
        mean, std = {}, {}
        for m, x in windows.items():
            mean[m] = np.nanmean(x, axis=(0, 2)).astype(np.float32)
            std[m] = np.maximum(np.nanstd(x, axis=(0, 2)), 1e-6).astype(np.float32)
        return NormStats(mean, std)

    @staticmethod
    def normalize(windows: dict[str, np.ndarray], stats: NormStats) -> dict[str, np.ndarray]:
        out = {}
        for m, x in windows.items():
            z = (x - stats.mean[m][None, :, None]) / stats.std[m][None, :, None]
            # A "missing" representation (NaN) is imputed with the training mean, i.e. 0.
            out[m] = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return out


@lru_cache(maxsize=20)   # one task's subjects + a few others; bounded for the API memory budget
def load_stream(task_key: str, subject: str) -> SubjectStream:
    task = get_task(task_key)
    record = DatasetLoader(task.dataset).load_subject(subject)
    return Preprocessor(task).prepare_subject(record)
