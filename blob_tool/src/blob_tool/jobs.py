"""Job store: one directory per job, processed one at a time by a background worker.

Models hold GPU memory and are not thread-safe, so jobs run strictly sequentially. Follow-up
tasks on a finished job (the opt-in blurred videos) go through the same queue.
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
from .blur import LAYERS as BLUR_FILES
from .blur import BlurParams

WORKSPACE = Path(os.environ.get("BLOB_WORKSPACE", Path(__file__).resolve().parents[2] / "workspace"))

# files a job may serve; anything else in the job dir is private
PUBLIC_FILES = {"original.mp4", "original.wav", "altered.wav", "processed.mp4", "tracks.json",
                "poses.json", "pose.mp4", "blobpose.mp4", *BLUR_FILES}
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
    tasks: dict[str, dict] = field(default_factory=dict)  # "blur" → {status, progress, message, error}

    @property
    def dir(self) -> Path:
        return WORKSPACE / self.id

    def save(self) -> None:
        self.files = sorted(p.name for p in self.dir.iterdir() if p.name in PUBLIC_FILES) if self.dir.exists() else []
        tmp = self.dir / "job.json.tmp"
        tmp.write_text(json.dumps(asdict(self)))
        tmp.replace(self.dir / "job.json")


# share of total progress per stage; tracking and pose dominate
STAGE_WEIGHTS = {"prepare": 0.05, "track": 0.5, "pose": 0.22, "render": 0.08, "voice": 0.05, "mux": 0.05, "blur": 0.05}


class JobManager:
    def __init__(self) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.queue: queue.Queue[tuple[str, str, dict | None]] = queue.Queue()  # (kind, job id, params)
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
            for task in job.tasks.values():
                if task["status"] in ("queued", "running"):
                    task.update(status="error", error="interrupted by server restart")
            if job.dir.exists():
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
        self.queue.put(("process", job.id, None))
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    @staticmethod
    def _busy(job: Job) -> bool:
        return job.status == "running" or any(t["status"] in ("queued", "running") for t in job.tasks.values())

    def delete(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or self._busy(job):
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        with self.lock:
            self.jobs.pop(job_id, None)
        return True

    def discard_original(self, job_id: str) -> Job | None:
        """Privacy mode: delete the full-quality original video and voice. Blurred versions stay: they are outputs."""
        job = self.jobs.get(job_id)
        if job is None or job.status != "done" or self._busy(job):  # a blur may be reading the original
            return None
        for name in ORIGINAL_FILES:
            (job.dir / name).unlink(missing_ok=True)
        job.original_discarded = True
        job.save()
        return job

    def blur(self, job_id: str, params: dict) -> Job:
        """Queue (re-)making the blurred videos of a finished job. Raises ValueError when not possible."""
        job = self.jobs.get(job_id)
        if job is None:
            raise ValueError("job not found")
        params = {**params, "enabled": True}
        BlurParams(**params)  # validate
        if job.status != "done":
            raise ValueError("job is not finished")
        if job.original_discarded or not (job.dir / "original.mp4").exists():
            raise ValueError("the original was discarded, so there is nothing to blur")
        if job.tasks.get("blur", {}).get("status") in ("queued", "running"):
            raise ValueError("a blur is already queued")
        job.tasks["blur"] = {"status": "queued", "progress": 0.0, "message": "", "error": None}
        job.save()
        self.queue.put(("blur", job.id, params))
        return job

    def _worker(self) -> None:
        while True:
            kind, job_id, params = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            if kind == "blur":
                self._blur(job, BlurParams(**params))
            else:
                self._run(job)

    def _blur(self, job: Job, params: BlurParams) -> None:
        task = job.tasks["blur"]
        task["status"] = "running"
        job.save()
        last_save = 0.0

        def progress(frac: float, message: str) -> None:
            nonlocal last_save
            task["progress"], task["message"] = min(frac, 1.0), message
            if time.time() - last_save > 1.0:
                last_save = time.time()
                job.save()

        try:
            min_score = pipeline.PipelineParams.from_dict(job.params).pose.min_score
            pipeline.blur(job.dir, params, min_score, progress)
            job.params["blur"] = asdict(params)
            task.update(status="done", progress=1.0, message="done", error=None)
        except Exception as e:
            traceback.print_exc()
            task.update(status="error", error=f"{type(e).__name__}: {e}")
        finally:
            if job.dir.exists():
                job.save()

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
            if params.seed is None:  # kept only in this local job.json, never in a served file
                params.seed = pipeline.new_seed()
                job.params["seed"] = params.seed
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
