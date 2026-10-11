"""FastAPI application. All routes live under /api; the web container's nginx proxies /api here."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .. import __version__
from .routers import data, experiments, inference, results, system

torch.set_num_threads(int(os.environ.get("TRUSTFUSE_THREADS", "2")))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from .state import jobs
    jobs()   # start the job worker and restore the job registry
    yield


app = FastAPI(title="TrustFuse API", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json",
              lifespan=lifespan,
              description="Security evaluation of intentional modality suppression in wearable health AI.")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("TRUSTFUSE_CORS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])
for r in (system.router, data.router, inference.router, results.router, experiments.router):
    app.include_router(r, prefix="/api")

