from __future__ import annotations

import io
import zipfile

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, PlainTextResponse, Response

from ...experiments import results as store

router = APIRouter(tags=["results"])
DOWNLOADABLE = {"summary.json", "conditions.json", "detection.json", "calibration.json", "run.json", "config.yaml",
                "rows.csv.gz", "subjects.csv.gz", "manifest.npz"}


def _guard(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as exc:
        raise HTTPException(404, f"not found: {exc}") from exc


@router.get("/results")
def list_results() -> list[dict]:
    return store.list_results()


@router.get("/results/{run_id}/manifest/conditions")
def manifest_conditions(run_id: str) -> dict:
    return _guard(store.manifest, run_id)


@router.get("/results/{run_id}/manifest/rows")
def manifest_rows(run_id: str, cond: str, limit: int = Query(500, ge=1, le=5000), offset: int = 0) -> dict:
    return _guard(store.manifest, run_id, cond, limit, offset)


@router.get("/results/{run_id}/manifest/csv")
def manifest_csv(run_id: str, cond: str) -> Response:
    text = _guard(store.manifest_csv, run_id, cond)
    safe = cond.replace("|", "_").replace("=", "-")
    return PlainTextResponse(text, media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{run_id}_{safe}.csv"'})


@router.get("/results/{run_id}/files/{name}")
def download(run_id: str, name: str):
    if name not in DOWNLOADABLE:
        raise HTTPException(404, "file not downloadable")
    path = _guard(store.result_dir, run_id) / name
    if not path.exists():
        raise HTTPException(404, "file missing")
    return FileResponse(path, filename=f"{run_id}_{name}")


@router.get("/results/{run_id}/bundle.zip")
def bundle(run_id: str) -> Response:
    d = _guard(store.result_dir, run_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            if p.is_file() and p.name in DOWNLOADABLE:
                z.write(p, f"{run_id}/{p.name}")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}_results.zip"'})


@router.post("/reports/{run_id}/figures")
async def figures(run_id: str, fmt: str = Query("pdf", pattern="^(pdf|png|svg)$")) -> Response:
    from ...evaluation.figures import figures_zip
    d = _guard(store.result_dir, run_id)
    data = await run_in_threadpool(figures_zip, d, fmt)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{run_id}_figures_{fmt}.zip"'})


@router.get("/reports/{run_id}/latex")
async def latex(run_id: str) -> PlainTextResponse:
    from ...evaluation.figures import latex_tables
    d = _guard(store.result_dir, run_id)
    return PlainTextResponse(await run_in_threadpool(latex_tables, d), media_type="text/plain")


# generic JSON route last so the specific routes above take precedence
@router.get("/results/{run_id}/{name}")
def result_json(run_id: str, name: str) -> dict:
    return _guard(store.read_json, run_id, name)
