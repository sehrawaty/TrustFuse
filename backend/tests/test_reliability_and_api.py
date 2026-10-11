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


def test_default_defense_is_the_report_design():
    """Report 4.3.3: R combines Q, D and U; scaling and thresholds come from clean validation data."""
    from trustfuse.experiments.config import ReliabilitySettings
    from trustfuse.reliability.engine import DEFAULT_CALIBRATION, DEFAULT_COMPONENTS
    assert DEFAULT_COMPONENTS == ("Q", "D", "U") and DEFAULT_CALIBRATION == "validation"
    r = ReliabilitySettings()
    assert r.components == ["Q", "D", "U"] and r.calibration == "validation"


def test_validation_calibration_uses_validation_subjects_only(bundle):
    """Every reference of the default scheme is refitted here from the validation subjects alone
    and must equal the frozen calibration; the pooled extension must differ from it."""
    from trustfuse.reliability.quality import QualityCalibration
    eng = bundle.engine
    split = bundle.ckpt["split"]
    assert not set(split["val"]) & (set(split["train"]) | set(split["test"]))
    val = gather("wesad_binary", split["val"])
    differs = False
    for m in eng.modalities:
        ref = QualityCalibration(m, eng.task.model_hz[m]).fit(val.phys[m])
        got = eng.quality["validation"][m]
        assert np.allclose(got.median, ref.median) and np.allclose(got.scale, ref.scale)
        assert eng.val_z["validation"][m].shape[0] == len(val)
        differs |= not np.allclose(eng.quality["pooled"][m].median, ref.median)
    assert differs
    dec = eng.run(val.phys)
    assert dec.components == ("Q", "D", "U") and dec.calibration == "validation"


def test_conformal_reliability_on_calibration_data_is_valid(bundle):
    """On its own reference data R is (super-)uniform: P(R <= t) <= t (+ finite-sample slack).
    The report scheme references every evidence set to the validation subjects; the pooled
    extension references Q to every non-test subject."""
    eng = bundle.engine
    split = bundle.ckpt["split"]
    val = gather("wesad_binary", split["val"])
    cases = [(("Q", "D", "U"), "validation", val), (("Q",), "validation", val),
             (("Q",), "pooled", gather("wesad_binary", split["train"] + split["val"], max_per_subject=300))]
    for comps, cal, ws in cases:
        dec = eng.run(ws.phys, components=comps, calibration=cal)
        n = len(ws)
        for m in eng.modalities:
            for t in (0.01, 0.05, 0.2):
                assert np.mean(dec.reliability[m] <= t) <= t + 3.0 / n + 0.005, (comps, cal, m, t)
        assert dec.flag.mean() <= eng.alpha + 4.0 / n + 0.005, (comps, cal)


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


def test_evaluator_traces_bursts_and_reports_calibration_ablations(bundle):
    """Every burst-affected window gets a traceability record, and the pooled-reference extension
    is evaluated next to the report design with unique condition ids."""
    from trustfuse.evaluation.evaluator import BURST_FIELDS, Evaluator
    from trustfuse.experiments.config import ExperimentConfig
    cfg = ExperimentConfig(
        name="test", task="wesad_binary", variants=["fusion"], seeds=[0], eval_seeds=[0],
        attacks={"severities": [1.0], "mask_fills": ["zero"], "attenuations": [0.8], "noise_snr_db": [],
                 "burst_windows": [2], "burst_placements_per_subject": 2, "strategies": ["targeted"]},
        reliability={"ablations": [["Q"]], "calibration_ablations": [["Q"]]})
    assert cfg.reliability.components == ["Q", "D", "U"] and cfg.reliability.calibration == "validation"
    res = Evaluator(bundle, cfg).evaluate(with_manifest=True)

    assert set(res.bursts) == set(BURST_FIELDS)
    n = len(res.bursts["start_s"])
    assert n > 0 and all(len(v) == n for v in res.bursts.values())
    assert set(np.unique(res.bursts["modality"])) == set(bundle.engine.modalities)
    assert (res.bursts["burst_windows"] == 2).all() and set(np.unique(res.bursts["placement"])) == {0, 1}
    # a fully suppressed window lies inside the suppressed interval
    full = res.bursts["fully_suppressed"]
    assert full.any() and (res.bursts["start_s"][full] >= res.bursts["burst_start_s"][full]).all()

    assert len({r["cond"] for r in res.rows}) == len(res.rows)
    abl = {(r["components"], r["calibration"], r["strategy"]) for r in res.rows if r["family"] == "ablation"}
    assert abl == {("Q", "validation", "embedding_gate"), ("QDU", "validation", "poe"), ("Q", "pooled", "embedding_gate")}
    assert "mask|modality=eda|fill=zero|severity=1" in res.manifest
    assert any(k.startswith("targeted") and k.endswith("#target") for k in res.manifest)


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
    assert body["components"] == ["Q", "D", "U"] and body["calibration"] == "validation"
    ext = client.post("/api/infer", json={"task": "wesad_binary", "subject": "S2", "window": 5, "components": ["Q"],
                                          "calibration": "pooled"})
    assert ext.status_code == 200 and ext.json()["calibration"] == "pooled"
    assert client.post("/api/infer", json={"task": "wesad_binary", "subject": "S2", "calibration": "test"}).status_code == 422
    cal = client.get("/api/calibration/wesad_binary/S2").json()
    assert cal["components"] == ["Q", "D", "U"] and cal["calibration"] == "validation"
    assert client.post("/api/infer", json={"task": "nope", "subject": "S2"}).status_code in (404, 422)
    t = client.post("/api/timeline", json={"task": "wesad_binary", "subject": "S2",
                                           "burst": {"modality": "eda", "start_index": 100, "length": 5}}).json()
    assert sum(t["affected"]) >= 5
    assert client.get("/api/experiments/presets").status_code == 200
