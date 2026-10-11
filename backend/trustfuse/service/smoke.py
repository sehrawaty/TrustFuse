"""End-to-end self test run by ``config-everything`` inside the API container."""

from __future__ import annotations

import time

from .. import paths
from ..data.raw import verify_manifest
from ..experiments.results import list_results
from ..specs import TASKS


def run_smoke() -> dict:
    checks = []

    def check(name: str, fn):
        t0 = time.time()
        try:
            detail = fn()
            checks.append({"check": name, "ok": True, "detail": detail, "seconds": round(time.time() - t0, 2)})
        except Exception as exc:  # noqa: BLE001
            checks.append({"check": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}",
                           "seconds": round(time.time() - t0, 2)})

    def data():
        problems = verify_manifest(paths.PROCESSED_DIR / "wesad") + verify_manifest(paths.PROCESSED_DIR / "dalia")
        if problems:
            raise RuntimeError("; ".join(problems[:5]))
        return "WESAD + PPG-DaLiA processed files match their checksums"

    def models():
        from ..models.checkpoint import find_checkpoint
        from ..specs import DATASETS
        counts = {}
        for key, t in TASKS.items():
            counts[key] = sum(1 for s in DATASETS[t.dataset].subjects if find_checkpoint(key, "fusion", 0, s))
        if counts["wesad_binary"] == 0:
            raise RuntimeError("no WESAD fold models found")
        return counts

    def inference():
        from .inference import InferenceService
        svc = InferenceService(capacity=1)
        r = svc.infer("wesad_binary", "S2", 0, {"type": "mask", "modality": "eda", "severity": 1.0, "fill": "zero"})
        return {"label": r["window"]["label_name"], "fixed": r["fixed"]["pred"], "defended": r["defended"]["pred"],
                "flag": r["flag"], "trace_ms": round(sum(s["ms"] for s in r["trace"]), 1)}

    def results():
        found = [r["run_id"] for r in list_results() if r.get("canonical")]
        if not found:
            raise RuntimeError("no canonical result sets found")
        return found

    check("processed data", data)
    check("fold models", models)
    check("attacked inference", inference)
    check("result sets", results)
    return {"ok": all(c["ok"] for c in checks), "checks": checks}
