"""Ask a video LLM (Gemini) the same question about each video variant and keep the answers side by side.

Each variant is its own stateless request: a fresh client, a fresh upload under a neutral file name,
one `generate_content` call and no chat history, so no answer can see another variant. The upload is
deleted right after. Runs are stored as workspace/<job>/llm/<run>.json and processed one at a time.
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

VARIANTS = {  # variant → file in the job dir
    "original": "original.mp4",
    "blobs": "processed.mp4",
    "pose": "pose.mp4",
    "blobpose": "blobpose.mp4",
    "blurred": "blurred.mp4",
    "faceblur": "faceblur.mp4",
}
SENSITIVE = ("original", "blurred", "faceblur")  # contain original pixels: never pre-selected, confirmed per run
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]  # free tier, video input
DEFAULT_PROMPT = "Describe what happens in this video and who said what."
MAX_PROMPT = 4000


def status() -> dict:
    installed = importlib.util.find_spec("google.genai") is not None
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    hint = None
    if not installed:
        hint = "run `make install-llm` to enable"
    elif not has_key:
        hint = "put GEMINI_API_KEY=… in blob_tool/.env and restart `make run`"
    return {"available": installed and has_key, "hint": hint, "models": MODELS, "variants": list(VARIANTS),
            "sensitive": list(SENSITIVE), "default_prompt": DEFAULT_PROMPT}


def gemini_client():
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def ask(client, video: Path, prompt: str, model: str) -> str:
    """One isolated question about one video."""
    with open(video, "rb") as f:  # a stream, so the local file name is never sent
        up = client.files.upload(file=f, config={"mime_type": "video/mp4", "display_name": "video.mp4"})
    try:
        deadline = time.time() + 300
        while getattr(up.state, "name", up.state) == "PROCESSING":
            if time.time() > deadline:
                raise TimeoutError("upload still processing after 5 min")
            time.sleep(2)
            up = client.files.get(name=up.name)
        if getattr(up.state, "name", up.state) != "ACTIVE":
            raise RuntimeError(f"upload failed: {up.state}")
        resp = client.models.generate_content(model=model, contents=[up, prompt], config={"temperature": 0.0})
        return resp.text or ""
    finally:
        try:
            client.files.delete(name=up.name)
        except Exception:
            traceback.print_exc()


@dataclass
class Run:
    id: str
    job_id: str
    created: float
    prompt: str
    model: str
    variants: list[str]
    status: str = "queued"  # queued | running | done
    answers: dict[str, dict] = field(default_factory=dict)  # variant → {status, text, error, seconds}


class LlmManager:
    def __init__(self, jobs, client_factory: Callable = gemini_client, start_worker: bool = True) -> None:
        self.jobs = jobs
        self.client_factory = client_factory
        self.queue: queue.Queue[Run] = queue.Queue()
        self.active: dict[str, Run] = {}
        if start_worker:
            threading.Thread(target=self._worker, daemon=True, name="llm-worker").start()

    def _path(self, job, run_id: str) -> Path:
        return job.dir / "llm" / f"{run_id}.json"

    def _save(self, job, run: Run) -> None:
        if not job.dir.exists():  # job deleted while the run was going
            return
        path = self._path(job, run.id)
        path.parent.mkdir(exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(run)))
        tmp.replace(path)

    def submit(self, job, prompt: str, variants: list[str], model: str, confirm_original: bool) -> Run:
        prompt = prompt.strip()
        if job.status != "done":
            raise ValueError("job is not finished")
        if not prompt or len(prompt) > MAX_PROMPT:
            raise ValueError(f"prompt must be 1–{MAX_PROMPT} characters")
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}")
        if not variants or any(v not in VARIANTS for v in variants):
            raise ValueError(f"variants must be a non-empty subset of {list(VARIANTS)}")
        sensitive = [v for v in variants if v in SENSITIVE]
        if sensitive and not confirm_original:
            raise ValueError(f"sending {sensitive} needs confirm_original: these versions contain original pixels")
        missing = [v for v in variants if not (job.dir / VARIANTS[v]).exists()]
        if missing:
            raise ValueError(f"not available for this job: {missing}")
        run = Run(uuid.uuid4().hex[:10], job.id, time.time(), prompt, model, list(dict.fromkeys(variants)),
                  answers={v: {"status": "queued"} for v in variants})
        self._save(job, run)
        self.active[run.id] = run
        self.queue.put(run)
        return run

    def list(self, job) -> list[dict]:
        runs = []
        for p in (job.dir / "llm").glob("*.json"):
            try:
                runs.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        return sorted(runs, key=lambda r: r["created"], reverse=True)

    def get(self, job, run_id: str) -> dict | None:
        if not run_id.isalnum():
            return None
        if run_id in self.active:
            return asdict(self.active[run_id])
        path = self._path(job, run_id)
        return json.loads(path.read_text()) if path.is_file() else None

    def _worker(self) -> None:
        while True:
            self.process(self.queue.get())

    def process(self, run: Run) -> None:
        job = self.jobs.get(run.job_id)
        run.status = "running"
        for v in run.variants:
            ans = run.answers[v]
            if job is None or not (job.dir / VARIANTS[v]).exists():  # job or original deleted meanwhile
                ans.update(status="error", error="video no longer available")
                continue
            ans["status"] = "running"
            self._save(job, run)
            t = time.time()
            try:
                ans.update(status="done", text=ask(self.client_factory(), job.dir / VARIANTS[v], run.prompt, run.model))
            except Exception as e:
                traceback.print_exc()
                ans.update(status="error", error=f"{type(e).__name__}: {e}")
            ans["seconds"] = round(time.time() - t, 1)
        run.status = "done"
        if job is not None:
            self._save(job, run)
        self.active.pop(run.id, None)
