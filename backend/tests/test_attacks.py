"""SMSA invariants: only the target modality changes, severity semantics are exact."""

import numpy as np
import pytest

from trustfuse.attacks.simulator import AttackSimulator, AttackSpec

MODS = ("bvp", "acc", "eda", "temp")


def batch(b=6):
    rng = np.random.default_rng(0)
    return {"bvp": rng.normal(size=(b, 1, 960)).astype(np.float32),
            "acc": rng.normal(size=(b, 3, 480)).astype(np.float32),
            "eda": rng.uniform(0.5, 3, size=(b, 1, 120)).astype(np.float32),
            "temp": rng.uniform(30, 35, size=(b, 1, 120)).astype(np.float32)}


@pytest.mark.parametrize("kind", ["mask", "attenuate", "noise"])
@pytest.mark.parametrize("target", MODS)
def test_untargeted_streams_are_untouched(kind, target):
    x = batch()
    snapshot = {m: v.copy() for m, v in x.items()}
    out = AttackSimulator(MODS).apply(x, AttackSpec(kind, target, 0.6, "zero"))
    for m in MODS:
        if m == target:
            assert not np.array_equal(out[m], snapshot[m])
        else:
            assert out[m] is x[m]                       # passed through by identity
            assert np.array_equal(out[m], snapshot[m])  # and byte-identical
    for m in MODS:                                      # inputs are never mutated in place
        assert np.array_equal(x[m], snapshot[m])


@pytest.mark.parametrize("sev", [0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_mask_severity_is_exact_share(sev):
    x = batch()
    out = AttackSimulator(MODS).mask(x["eda"], sev, "missing", "random", np.random.default_rng(1))
    share = np.isnan(out).mean(axis=(1, 2))
    assert np.allclose(share, round(sev * 120) / 120)
    # the suppressed samples form one contiguous segment
    for row in np.isnan(out[:, 0]):
        idx = np.flatnonzero(row)
        assert np.all(np.diff(idx) == 1)


def test_fills():
    x = batch()["temp"]
    sim = AttackSimulator(MODS)
    assert np.all(sim.mask(x, 1.0, "zero", "start") == 0)
    held = sim.mask(x, 0.5, "hold", "end")
    assert np.allclose(held[:, :, 60:], held[:, :, 59:60])


def test_attenuation_scales():
    x = batch()["bvp"]
    assert np.allclose(AttackSimulator.attenuate(x, 0.8), x * 0.2, atol=1e-6)


def test_burst_restores_after_interval():
    sig = {"eda": np.ones((1, 400), np.float32), "bvp": np.ones((1, 3200), np.float32)}
    out = AttackSimulator.burst_drop(sig, "eda", 10.0, 20.0, 4, "zero")
    assert np.all(out["eda"][0, 40:120] == 0)
    assert np.all(out["eda"][0, :40] == 1) and np.all(out["eda"][0, 120:] == 1)
    assert out["bvp"] is sig["bvp"]


def test_random_drop_hits_exactly_one_modality_per_window():
    x = batch(20)
    out, chosen = AttackSimulator(MODS, seed=3).random_drop(x)
    for i, m in enumerate(chosen):
        for j in MODS:
            changed = not np.array_equal(out[j][i], x[j][i])
            assert changed == (j == m)


def test_spec_validation():
    with pytest.raises(ValueError):
        AttackSpec("mask", "eda", 1.5)
    with pytest.raises(ValueError):
        AttackSpec("explode", "eda")
