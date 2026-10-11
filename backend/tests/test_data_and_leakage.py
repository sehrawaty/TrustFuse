"""Data pipeline and leakage control on the real processed data."""

import numpy as np
import pytest

from trustfuse import paths
from trustfuse.data.loader import DatasetLoader
from trustfuse.data.preprocess import Preprocessor, load_stream
from trustfuse.data.raw import verify_manifest
from trustfuse.specs import TASKS, get_task

pytestmark = pytest.mark.skipif(not DatasetLoader("wesad").is_ready(), reason="processed data not present")


@pytest.mark.parametrize("ds", ["wesad", "dalia"])
def test_checksums(ds):
    assert verify_manifest(paths.PROCESSED_DIR / ds) == []


@pytest.mark.parametrize("ds", ["wesad", "dalia"])
def test_loso_splits_are_subject_disjoint(ds):
    loader = DatasetLoader(ds)
    splits = loader.loso_splits()
    assert len(splits) == 15
    for sp in splits:
        assert not set(sp.train) & set(sp.val)
        assert not set(sp.train) & set(sp.test)
        assert not set(sp.val) & set(sp.test)
        assert len(sp.train) + len(sp.val) + len(sp.test) == 15
    assert sorted(s.test[0] for s in splits) == sorted(loader.spec.subjects)


@pytest.mark.parametrize("task", list(TASKS))
def test_windows_are_label_pure_and_on_grid(task):
    t = get_task(task)
    st = load_stream(task, DatasetLoader(t.dataset).spec.subjects[0])
    w = st.windows
    assert len(w) > 50
    assert np.all(w.start_s % t.stride_s == 0)
    for s, y in zip(w.start_s[::7], w.y[::7]):
        seg = st.labels[s * 4:(s + t.window_s) * 4]
        assert np.all(seg == y)


def test_normalizer_uses_only_given_windows():
    t = get_task("wesad_binary")
    pp = Preprocessor(t)
    st = load_stream(t.key, "S2")
    phys = pp.extract(st.signals, st.windows.start_s[:50])
    norm = pp.fit_normalizer(phys)
    z = pp.normalize(phys, norm)
    for m in t.modalities:
        assert abs(float(z[m].mean())) < 1e-3
        assert np.isfinite(z[m]).all()


def test_missing_values_are_imputed_with_training_mean():
    t = get_task("wesad_binary")
    pp = Preprocessor(t)
    st = load_stream(t.key, "S2")
    phys = pp.extract(st.signals, st.windows.start_s[:4])
    norm = pp.fit_normalizer(phys)
    phys["eda"][:] = np.nan
    assert np.all(pp.normalize(phys, norm)["eda"] == 0)
