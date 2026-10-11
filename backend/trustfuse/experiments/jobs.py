"""Background experiment jobs: one worker thread (fits the 1.5 GB budget), SQLite registry, and an
in-memory event log streamed to the UI with Server-Sent Events."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

from .. import paths
from .config import ExperimentConfig
from .training import enumerate_jobs, models_root
from ..models.checkpoint import checkpoint_path

TERMINAL = {"completed", "failed", "cancelled"}


@dataclass
class Job:
    id: str
    name: str
    kind: str
    config: dict
    status: str = "queued"
    stage: str = "queued"
    progress: float = 0.0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    run_id: str | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind, "status": self.status, "stage": self.stage,
                "progress": round(self.progress, 4), "created": self.created, "started": self.started,
                "finished": self.finished, "run_id": self.run_id, "error": self.error,
                "task": self.config.get("task"), "n_events": len(self.events)}


class JobManager:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (paths.RUNTIME_DIR / "trustfuse.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: Queue[str] = Queue()
        self.lock = threading.Lock()
        self._init_db()
        self.worker = threading.Thread(target=self._loop, name="trustfuse-jobs", daemon=True)
        self.worker.start()

    # -- persistence ------------------------------------------------------------------------------
    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=5)

    def _init_db(self) -> None:
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, name TEXT, kind TEXT, task TEXT,
                          status TEXT, created REAL, started REAL, finished REAL, run_id TEXT, error TEXT, config TEXT)""")
            # jobs that were running when the process stopped cannot be resumed
            db.execute("UPDATE jobs SET status='failed', error='interrupted by restart' WHERE status IN ('queued','running')")
            for row in db.execute("SELECT id,name,kind,status,created,started,finished,run_id,error,config FROM jobs"):
                j = Job(id=row[0], name=row[1], kind=row[2], config=json.loads(row[9]), status=row[3], stage=row[3],
                        progress=1.0 if row[3] == "completed" else 0.0, created=row[4], started=row[5],
                        finished=row[6], run_id=row[7], error=row[8])
                self.jobs[j.id] = j

    def _save(self, job: Job) -> None:
        with self._db() as db:
            db.execute("INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                       (job.id, job.name, job.kind, job.config.get("task"), job.status, job.created, job.started,
                        job.finished, job.run_id, job.error, json.dumps(job.config)))

    # -- API --------------------------------------------------------------------------------------
    def submit(self, cfg: ExperimentConfig, kind: str = "experiment") -> Job:
        busy = [j for j in self.jobs.values() if j.status in ("queued", "running")]
        if len(busy) >= 3:
            raise RuntimeError("too many queued jobs; wait for one to finish")
        job = Job(id=uuid.uuid4().hex[:10], name=cfg.name, kind=kind, config=cfg.model_dump())
        with self.lock:
            self.jobs[job.id] = job
        self._event(job, "status", {"status": "queued"})
        self._save(job)
        self.queue.put(job.id)
        return job

    def list(self) -> list[dict]:
        return [j.public() for j in sorted(self.jobs.values(), key=lambda j: -j.created)]

    def get(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(job_id)
        return self.jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.status not in TERMINAL:
            job.cancel.set()
            self._event(job, "log", {"message": "cancellation requested"})
            if job.status == "queued":
                job.status = job.stage = "cancelled"
                job.finished = time.time()
                self._event(job, "status", {"status": "cancelled"})
                self._save(job)
        return job

    def events_since(self, job_id: str, seq: int) -> list[dict]:
        job = self.get(job_id)
        return job.events[seq:]

    # -- worker -----------------------------------------------------------------------------------
    def _event(self, job: Job, kind: str, data: dict) -> None:
        with self.lock:
            job.events.append({"seq": len(job.events), "t": time.time(), "type": kind, "data": data})
            if len(job.events) > 5000:          # keep memory bounded for very long jobs
                job.events = job.events[:100] + job.events[-4000:]

    def _loop(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        from .runner import run_experiment

        cfg = ExperimentConfig.model_validate(job.config)
        job.status = "running"
        job.started = time.time()
        self._save(job)
        self._event(job, "status", {"status": "running"})
        root = models_root(cfg)
        train_jobs = enumerate_jobs(cfg)
        todo = [j for j in train_jobs if not checkpoint_path(root, j.task, j.variant, j.seed, j.subject).exists()]
        n_train = max(len(todo), 0)
        train_share = 0.65 if n_train else 0.0
        state = {"done": 0, "current": None}

        def log(msg: str) -> None:
            self._event(job, "log", {"message": msg})
            if msg.startswith("[train]") and "/" in msg and "val F1" in msg:
                state["done"] += 1
                job.progress = train_share * state["done"] / max(n_train, 1)
            if msg.startswith("[eval]") and "fold evaluations" in msg:
                job.stage = "evaluating"
                self._event(job, "stage", {"stage": "evaluating"})

        def on_epoch(tj, rec: dict) -> None:
            if state["current"] != tj:
                state["current"] = tj
                job.stage = "training"
                self._event(job, "stage", {"stage": "training", "model": f"{tj.variant} seed{tj.seed} {tj.subject}"})
            frac = rec["epoch"] / max(cfg.train.epochs, 1)
            job.progress = train_share * (state["done"] + min(frac, 1.0)) / max(n_train, 1)
            self._event(job, "epoch", {"model": f"{tj.variant} seed{tj.seed} {tj.subject}", **rec})

        def progress(frac: float, msg: str) -> None:
            job.progress = train_share + (1 - train_share) * frac
            self._event(job, "progress", {"progress": job.progress, "message": msg})

        try:
            job.stage = "training" if n_train else "evaluating"
            self._event(job, "stage", {"stage": job.stage, "train_jobs": n_train})
            out = run_experiment(cfg, log=log, should_stop=job.cancel.is_set, on_epoch=on_epoch, progress=progress,
                                 run_id=f"{cfg.task}-{time.strftime('%Y%m%d-%H%M%S')}-{job.id[:4]}")
            job.run_id = out.name
            job.status = job.stage = "completed"
            job.progress = 1.0
        except Exception as exc:   # noqa: BLE001 - surface every failure to the UI
            cancelled = job.cancel.is_set()
            job.status = job.stage = "cancelled" if cancelled else "failed"
            job.error = None if cancelled else f"{type(exc).__name__}: {exc}"
            if not cancelled:
                self._event(job, "log", {"message": traceback.format_exc(limit=6)})
        finally:
            from ..service.bundle import release_memory
            release_memory()
            job.finished = time.time()
            self._save(job)
            self._event(job, "status", {"status": job.status, "run_id": job.run_id, "error": job.error})
