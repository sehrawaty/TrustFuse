"""Read-only access to result sets (canonical shipped results and user runs)."""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path

import numpy as np

from .. import paths

JSON_FILES = ("summary", "conditions", "detection", "calibration", "run")


def list_results() -> list[dict]:
    out = []
    for root in paths.results_dirs():
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            run = d / "run.json"
            if d.is_dir() and run.exists():
                try:
                    meta = json.loads(run.read_text())
                except json.JSONDecodeError:
                    continue
                meta["canonical"] = root == paths.ARTIFACTS_DIR / "results"
                meta["files"] = sorted(p.name for p in d.iterdir() if p.is_file())
                out.append(meta)
    out.sort(key=lambda m: (not m["canonical"], m.get("task", ""), m.get("created", "")))
    return out


def result_dir(run_id: str) -> Path:
    if "/" in run_id or ".." in run_id:
        raise KeyError(run_id)
    for root in paths.results_dirs():
        d = root / run_id
        if (d / "run.json").exists():
            return d
    raise KeyError(run_id)


def read_json(run_id: str, name: str) -> dict:
    if name not in JSON_FILES:
        raise KeyError(name)
    p = result_dir(run_id) / f"{name}.json"
    if not p.exists():
        raise KeyError(name)
    return json.loads(p.read_text())


def read_csv_rows(run_id: str, name: str = "subjects") -> list[dict]:
    with gzip.open(result_dir(run_id) / f"{name}.csv.gz", "rt") as f:
        return list(csv.DictReader(f))


def manifest(run_id: str, cond: str | None = None, limit: int = 2000, offset: int = 0) -> dict:
    p = result_dir(run_id) / "manifest.npz"
    if not p.exists():
        raise KeyError("manifest")
    with np.load(p, allow_pickle=False) as z:
        conds = z["conditions"].tolist()
        fields = z["fields"].tolist()
        if cond is None:
            return {"conditions": [c for c in conds if not c.endswith("#target")], "fields": fields,
                    "n_windows": int(z["y"].shape[0])}
        if cond not in conds:
            raise KeyError(cond)
        arr = z[f"c{conds.index(cond)}"].astype(float)
        tgt_key = cond + "#target"
        target = z[f"c{conds.index(tgt_key)}"] if tgt_key in conds else None
        n = arr.shape[-1]
        sl = slice(offset, min(n, offset + limit))
        rows = []
        for i in range(sl.start, sl.stop):
            r = {"subject": str(z["subject"][i]), "start_s": int(z["start_s"][i]), "y": int(z["y"][i])}
            r.update({f: (int(arr[k, i]) if f in ("fixed_pred", "def_pred", "flag") else round(float(arr[k, i]), 4))
                      for k, f in enumerate(fields)})
            if target is not None:
                r["target_index"] = int(target[i])
            rows.append(r)
        return {"condition": cond, "total": n, "offset": offset, "rows": rows}


def manifest_csv(run_id: str, cond: str) -> str:
    data = manifest(run_id, cond, limit=10**9)
    buf = io.StringIO()
    if data["rows"]:
        w = csv.DictWriter(buf, fieldnames=list(data["rows"][0]))
        w.writeheader()
        w.writerows(data["rows"])
    return buf.getvalue()
