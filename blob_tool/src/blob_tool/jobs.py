"""Job store: one directory per job, processed one at a time by a background worker.

Models hold GPU memory and are not thread-safe, so jobs run strictly sequentially.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import pipeline

WORKSPACE = Path(os.environ.get("BLOB_WORKSPACE", Path(__file__).resolve().parents[2] / "workspace"))

# files a job may serve; anything else in the job dir is private
PUBLIC_FILES = {"original.mp4", "original.wav", "altered.wav", "processed.mp4", "tracks.json",
                "poses.json", "pose.mp4", "blobpose.mp4"}
ORIGINAL_FILES = ("original.mp4", "original.wav")


@dataclass
class Job:
    id: str
    filename: str
    created: float
    params: dict
    status: str = "queued"  # queued | running | done | error
    stage: str = ""
    stage_progress: float = 0.0
    progress: float = 0.0
    message: str = ""
    error: str | None = None
    result: dict | None = None
    original_discarded: bool = False
    files: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return WORKSPACE / self.id

    def save(self) -> None:
        self.files = sorted(p.name for p in self.dir.iterdir() if p.name in PUBLIC_FILES) if self.dir.exists() else []
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(self.dir / "job.json")


# share of total progress per stage; tracking and pose dominate
STAGE_WEIGHTS = {"prepare": 0.05, "track": 0.5, "pose": 0.25, "render": 0.08, "voice": 0.05, "mux": 0.07}


class JobManager:
    def __init__(self) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[str] = queue.Queue()
        self._load_existing()
        threading.Thread(target=self._worker, daemon=True, name="blob-worker").start()

    def _load_existing(self) -> None:
        for meta in WORKSPACE.glob("*/job.json"):
            try:
                job = Job(**json.loads(meta.read_text()))
            except Exception:
                continue
            if job.status in ("queued", "running"):  # server restarted mid-job
                job.status, job.error = "error", "interrupted by server restart"
                job.save()
            self.jobs[job.id] = job

    def create(self, upload_path: Path, filename: str, params: dict) -> Job:
        pipeline.PipelineParams.from_dict(params)  # validate early
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, created=time.time(), params=params)
        job.dir.mkdir(parents=True)
        shutil.move(upload_path, job.dir / ("input" + Path(filename).suffix.lower()))
        job.save()
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job.id)
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status == "running":
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job_id, None)
        return True

    def discard_original(self, job_id: str) -> Job | None:
        """Privacy mode: delete every file that contains original pixels or voice."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "done":
            return None
        for name in ORIGINAL_FILES:
            (job.dir / name).unlink(missing_ok=True)
        job.original_discarded = True
        job.save()
        return job

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        job.status = "running"
        job.save()
        last_save = 0.0

        def progress(stage: str, frac: float, message: str) -> None:
            nonlocal last_save
            done = sum(w for s, w in STAGE_WEIGHTS.items() if pipeline.STAGES.index(s) < pipeline.STAGES.index(stage))
            job.stage, job.stage_progress, job.message = stage, min(frac, 1.0), message
            job.progress = min(done + STAGE_WEIGHTS[stage] * min(frac, 1.0), 1.0)
            now = time.time()
            if now - last_save > 1.0:
                last_save = now
                job.save()

        src = next(job.dir.glob("input.*"))
        try:
            params = pipeline.PipelineParams.from_dict(job.params)
            job.result = pipeline.run(src, job.dir, params, progress)
            job.status, job.progress, job.message = "done", 1.0, "done"
            if job.params.get("discard_original"):
                for name in ORIGINAL_FILES:
                    (job.dir / name).unlink(missing_ok=True)
                job.original_discarded = True
        except Exception as e:
            traceback.print_exc()
            job.status, job.error = "error", f"{type(e).__name__}: {e}"
        finally:
            src.unlink(missing_ok=True)  # the raw upload is never kept
            job.save()
