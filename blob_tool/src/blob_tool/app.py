"""FastAPI server: upload → background processing → serve results + static UI."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .jobs import PUBLIC_FILES, JobManager
from .render import STYLES
from .tracking import available_engines

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="Blob Tool", version=__version__)
manager = JobManager()


def _job_or_404(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/config")
def config() -> dict:
    from .tracking.yolo import class_names

    return {
        "version": __version__,
        "engines": available_engines(),
        "classes": class_names(),
        "styles": list(STYLES),
    }


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [asdict(j) for j in manager.list()]


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), params: str = Form("{}")) -> dict:
    try:
        parsed = json.loads(params)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"params is not valid JSON: {e}") from e
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        job = manager.create(Path(tmp.name), file.filename or "video.mp4", parsed)
    except (TypeError, ValueError) as e:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(400, f"invalid params: {e}") from e
    return asdict(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return asdict(_job_or_404(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not manager.delete(job_id):
        raise HTTPException(409, "job not found or still running")
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/discard-original")
def discard_original(job_id: str) -> dict:
    job = manager.discard_original(job_id)
    if job is None:
        raise HTTPException(409, "job not found or not finished")
    return asdict(job)


@app.get("/api/jobs/{job_id}/files/{name}")
def get_file(job_id: str, name: str) -> FileResponse:
    job = _job_or_404(job_id)
    path = job.dir / name
    if name not in PUBLIC_FILES or not path.exists():
        raise HTTPException(404, "file not found")
    download = f"{Path(job.filename).stem}_{name}"
    # FileResponse supports HTTP Range, which <video> seeking needs
    return FileResponse(path, filename=download, content_disposition_type="inline")


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
