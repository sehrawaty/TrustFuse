"""Helpers that gather windows of several subjects into model-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..specs import get_task
from .preprocess import NormStats, Preprocessor, load_stream


@dataclass
class WindowSet:
    phys: dict[str, np.ndarray]     # modality -> (B, C, L) physical units
    y: np.ndarray
    subject: np.ndarray             # subject id per window
    start_s: np.ndarray
    segment: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])


def gather(task_key: str, subjects: list[str] | tuple[str, ...], max_per_subject: int | None = None) -> WindowSet:
    """Windows of several subjects. ``max_per_subject`` keeps an evenly spaced subset per subject
    (used for calibration references, so memory stays small)."""
    task = get_task(task_key)
    pp = Preprocessor(task)
    phys = {m: [] for m in task.modalities}
    ys, subs, starts, segs = [], [], [], []
    for s in subjects:
        st = load_stream(task_key, s)
        win = st.windows
        if max_per_subject and len(win) > max_per_subject:
            win = win.subset(np.linspace(0, len(win) - 1, max_per_subject).astype(int))
        w = pp.extract(st.signals, win.start_s)
        for m in phys:
            phys[m].append(w[m])
        ys.append(win.y)
        subs.append(np.full(len(win), s))
        starts.append(win.start_s)
        segs.append(win.segment)
    return WindowSet(phys={m: np.concatenate(v) for m, v in phys.items()}, y=np.concatenate(ys),
                     subject=np.concatenate(subs), start_s=np.concatenate(starts), segment=np.concatenate(segs))


def normalize(task_key: str, phys: dict[str, np.ndarray], norm: NormStats) -> dict[str, np.ndarray]:
    return Preprocessor(get_task(task_key)).normalize(phys, norm)
