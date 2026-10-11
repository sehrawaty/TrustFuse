"""Static descriptions of datasets, modalities and prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass, field

MODALITIES: tuple[str, ...] = ("bvp", "acc", "eda", "temp")


@dataclass(frozen=True)
class ModalitySpec:
    key: str
    label: str
    unit: str
    native_hz: int
    channels: int
    channel_names: tuple[str, ...]


MODALITY_SPECS: dict[str, ModalitySpec] = {
    "bvp": ModalitySpec("bvp", "BVP / PPG", "a.u.", 64, 1, ("bvp",)),
    "acc": ModalitySpec("acc", "Accelerometer", "g", 32, 3, ("x", "y", "z")),
    "eda": ModalitySpec("eda", "EDA", "µS", 4, 1, ("eda",)),
    "temp": ModalitySpec("temp", "Skin temperature", "°C", 4, 1, ("temp",)),
}

LABEL_HZ = 4  # all label tracks are stored at 4 Hz (the slowest wrist modality)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    subjects: tuple[str, ...]
    label_names: dict[int, str]
    citation: str
    url: str
    license: str
    description: str


DATASETS: dict[str, DatasetSpec] = {
    "wesad": DatasetSpec(
        key="wesad",
        name="WESAD",
        subjects=tuple(f"S{i}" for i in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17)),
        label_names={0: "transient", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation",
                     5: "ignore", 6: "ignore", 7: "ignore"},
        citation=("P. Schmidt, A. Reiss, R. Duerichen, C. Marberger, K. Van Laerhoven, "
                  "\"Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection\", "
                  "ICMI 2018. doi:10.1145/3242969.3242985"),
        url="https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
        license="Academic, non-commercial research use (see original WESAD readme)",
        description="Wrist-worn Empatica E4 (BVP, EDA, skin temperature, 3-axis ACC) recorded during a "
                    "laboratory protocol with baseline, stress (TSST) and amusement conditions.",
    ),
    "dalia": DatasetSpec(
        key="dalia",
        name="PPG-DaLiA",
        subjects=tuple(f"S{i}" for i in range(1, 16)),
        label_names={0: "transient", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling",
                     5: "driving", 6: "lunch", 7: "walking", 8: "working"},
        citation=("A. Reiss, I. Indlekofer, P. Schmidt, K. Van Laerhoven, \"Deep PPG: Large-Scale Heart "
                  "Rate Estimation with Convolutional Neural Networks\", Sensors 19(14):3079, 2019. "
                  "doi:10.3390/s19143079"),
        url="https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
        license="CC BY 4.0 (UCI Machine Learning Repository)",
        description="Wrist-worn Empatica E4 during daily-life activities; used here as a focused "
                    "PPG/BVP + accelerometer validation setting.",
    ),
}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    dataset: str
    title: str
    modalities: tuple[str, ...]
    classes: tuple[str, ...]
    label_map: dict[int, int]          # raw dataset label -> class index (others excluded)
    window_s: int
    stride_s: int
    model_hz: dict[str, int] = field(default_factory=dict)
    headline: bool = False

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def input_shapes(self) -> dict[str, tuple[int, int]]:
        return {m: (MODALITY_SPECS[m].channels, self.window_s * self.model_hz[m]) for m in self.modalities}


TASKS: dict[str, TaskSpec] = {
    "wesad_binary": TaskSpec(
        key="wesad_binary",
        dataset="wesad",
        title="WESAD · stress vs. non-stress",
        modalities=("bvp", "acc", "eda", "temp"),
        classes=("non-stress", "stress"),
        label_map={1: 0, 3: 0, 2: 1},
        window_s=30,
        stride_s=5,
        model_hz={"bvp": 32, "acc": 16, "eda": 4, "temp": 4},
        headline=True,
    ),
    "wesad_3class": TaskSpec(
        key="wesad_3class",
        dataset="wesad",
        title="WESAD · baseline / stress / amusement",
        modalities=("bvp", "acc", "eda", "temp"),
        classes=("baseline", "stress", "amusement"),
        label_map={1: 0, 2: 1, 3: 2},
        window_s=30,
        stride_s=5,
        model_hz={"bvp": 32, "acc": 16, "eda": 4, "temp": 4},
    ),
    "dalia_activity": TaskSpec(
        key="dalia_activity",
        dataset="dalia",
        title="PPG-DaLiA · activity (BVP + ACC)",
        modalities=("bvp", "acc"),
        classes=("sitting", "stairs", "table soccer", "cycling", "driving", "lunch", "walking", "working"),
        label_map={i: i - 1 for i in range(1, 9)},
        window_s=8,
        stride_s=4,
        model_hz={"bvp": 32, "acc": 32},
    ),
}


def get_task(key: str) -> TaskSpec:
    try:
        return TASKS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown task '{key}', expected one of {sorted(TASKS)}") from exc
