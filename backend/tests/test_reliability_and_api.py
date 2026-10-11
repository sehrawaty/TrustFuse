"""Reliability engine on a real fold model + API contract tests."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trustfuse.attacks.simulator import AttackSimulator
from trustfuse.data.windows import gather
from trustfuse.models.checkpoint import find_checkpoint
from trustfuse.service.bundle import load_bundle

HAS_MODEL = find_checkpoint("wesad_binary", "fusion", 0, "S2") is not None
pytestmark = pytest.mark.skipif(not HAS_MODEL, reason="fold model not present")


@pytest.fixture(scope="module")
def bundle():
    return load_bundle("wesad_binary", "fusion", 0, "S2", persist=False)


def test_conformal_reliability_on_calibration_data_is_valid(bundle):
    """On the clean validation windows themselves R is (super-)uniform: P(R <= t) <= t (+1/n)."""
    eng = bundle.engine
    val = gather("wesad_binary", bundle.ckpt["split"]["val"])
    dec = eng.run(val.phys)
    n = len(val)
    for m in eng.modalities:
        for t in (0.01, 0.05, 0.2):
            assert np.mean(dec.reliability[m] <= t) <= t + 2.0 / n
    assert dec.flag.mean() <= eng.alpha + 4.0 / n


def test_determinism(bundle):
    ws = gather("wesad_binary", ["S2"])
    phys = {m: v[:40] for m, v in ws.phys.items()}
    a, b = bundle.engine.run(phys), bundle.engine.run(phys)
    assert np.array_equal(a.defended_probs, b.defended_probs)
    for m in bundle.engine.modalities:
        assert np.array_equal(a.reliability[m], b.reliability[m])


def test_full_zero_mask_is_flagged_and_gated(bundle):
    ws = gather("wesad_binary", ["S2"])
    phys = {m: v[:60] for m, v in ws.phys.items()}
    att = AttackSimulator(bundle.engine.modalities).drop_modality(phys, "eda", "zero")
    dec = bundle.engine.run(att)
    assert dec.flag.mean() > 0.9
    assert dec.weights["eda"].mean() < 0.2


@pytest.fixture(scope="module")
def client():
    from trustfuse.api.app import app
    with TestClient(app) as c:
        yield c


def test_api_contract(client):
    assert client.get("/api/health").json()["status"] == "ok"
    ov = client.get("/api/overview").json()
    assert {t["key"] for t in ov["tasks"]} == {"wesad_binary", "wesad_3class", "dalia_activity"}
    r = client.post("/api/infer", json={"task": "wesad_binary", "subject": "S2", "window": 5,
                                        "attack": {"type": "mask", "modality": "eda", "severity": 1, "fill": "zero"}})
    assert r.status_code == 200
    body = r.json()
    assert set(body["modalities"]) == {"bvp", "acc", "eda", "temp"}
    assert len(body["trace"]) == 8 and body["flag"] in (True, False)
    assert client.post("/api/infer", json={"task": "nope", "subject": "S2"}).status_code in (404, 422)
    t = client.post("/api/timeline", json={"task": "wesad_binary", "subject": "S2",
                                           "burst": {"modality": "eda", "start_index": 100, "length": 5}}).json()
    assert sum(t["affected"]) >= 5
    assert client.get("/api/experiments/presets").status_code == 200
