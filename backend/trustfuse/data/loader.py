"""DatasetLoader: reads processed subject files and produces leakage-controlled splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import paths
from ..specs import DATASETS, LABEL_HZ, MODALITY_SPECS, TaskSpec


@dataclass
class SubjectRecord:
    """Synchronized wrist streams of one subject in physical units at native rates."""

    dataset: str
    subject: str
    signals: dict[str, np.ndarray]   # modality -> float32 (channels, samples)
    label: np.ndarray                # int8, 4 Hz raw dataset label
    duration_s: float

    def rate(self, modality: str) -> int:
        return MODALITY_SPECS[modality].native_hz


@dataclass(frozen=True)
class Split:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def assert_disjoint(self) -> None:
        a, b, c = set(self.train), set(self.val), set(self.test)
        if a & b or a & c or b & c:
            raise AssertionError(f"subject leakage between splits: {self}")


class DatasetLoader:
    def __init__(self, dataset: str, processed_dir: Path | None = None):
        if dataset not in DATASETS:
            raise KeyError(f"unknown dataset {dataset}")
        self.dataset = dataset
        self.spec = DATASETS[dataset]
        self.dir = (processed_dir or paths.PROCESSED_DIR) / dataset

    # -- discovery -------------------------------------------------------------------------
    def manifest(self) -> dict:
        path = self.dir / "manifest.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def available_subjects(self) -> list[str]:
        return [s for s in self.spec.subjects if (self.dir / f"{s}.npz").exists()]

    def is_ready(self) -> bool:
        return len(self.available_subjects()) == len(self.spec.subjects)

    # -- loading ---------------------------------------------------------------------------
    def load_subject(self, subject: str) -> SubjectRecord:
        return _load_subject_cached(str(self.dir), self.dataset, subject)

    def load_subjects(self, subjects: list[str] | None = None) -> dict[str, SubjectRecord]:
        return {s: self.load_subject(s) for s in (subjects or self.available_subjects())}

    @staticmethod
    def sync_modalities(record: SubjectRecord, modalities: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Return the requested streams cropped to a common duration (already synchronized by
        the dataset authors; this guards against off-by-one sample counts)."""
        n_seconds = record.duration_s
        out = {}
        for m in modalities:
            hz = record.rate(m)
            out[m] = record.signals[m][:, : int(round(n_seconds * hz))]
        return out

    @staticmethod
    def get_labels(record: SubjectRecord, task: TaskSpec) -> np.ndarray:
        """Map raw labels to task classes; -1 marks excluded samples (transients, other conditions)."""
        mapped = np.full(record.label.shape, -1, dtype=np.int16)
        for raw, cls in task.label_map.items():
            mapped[record.label == raw] = cls
        return mapped

    # -- splits ----------------------------------------------------------------------------
    def split_data(self, test_subject: str, n_val: int = 2, subjects: list[str] | None = None) -> Split:
        """Leave-one-subject-out split. Validation subjects are the next ``n_val`` subjects in the
        canonical order (cyclic), so every fold is deterministic and validation is subject-disjoint."""
        order = list(subjects or self.spec.subjects)
        if test_subject not in order:
            raise KeyError(f"{test_subject} not in {order}")
        i = order.index(test_subject)
        rest = order[i + 1:] + order[:i]
        val = tuple(rest[:n_val])
        train = tuple(s for s in order if s != test_subject and s not in val)
        split = Split(train=train, val=val, test=(test_subject,))
        split.assert_disjoint()
        return split

    def loso_splits(self, n_val: int = 2, subjects: list[str] | None = None) -> list[Split]:
        order = list(subjects or self.spec.subjects)
        return [self.split_data(s, n_val=n_val, subjects=order) for s in order]


@lru_cache(maxsize=40)
def _load_subject_cached(directory: str, dataset: str, subject: str) -> SubjectRecord:
    with np.load(Path(directory) / f"{subject}.npz") as z:
        signals = {
            "bvp": z["bvp"].astype(np.float32),
            "acc": (z["acc"].astype(np.float32) / 64.0),
            "eda": z["eda"].astype(np.float32),
            "temp": z["temp"].astype(np.float32),
        }
        label = z["label"].astype(np.int8)
    return SubjectRecord(dataset=dataset, subject=subject, signals=signals, label=label,
                         duration_s=label.shape[0] / LABEL_HZ)
