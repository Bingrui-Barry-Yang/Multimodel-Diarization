# CLAUDE.md

## Project

This project does privacy-preserving classroom analysis. Raw classroom video becomes **tracked colour blobs + altered voice**, so that nothing identifies the children. Local multimodal models and traditional diarization then run on that processed data. The research question is whether models can still extract speaker and turn-taking features from blobs alone.

- **Stage 1 (done):** `blob_tool/`, a localhost "filter" app. Upload a video, then compare before/after, inspect objects, and download the blob video, altered audio and `tracks.json`.
- **Stage 2 (next):** diarization on the outputs (`tracks.json` + `altered.wav`). The AMI sample with ground-truth `.rttm` is the evaluation starting point.
- **End goal:** the same tool on a phone, recording live and saving **only** the processed output, never the original.

## Commands

All targets work from the repo root (the root `Makefile` delegates to `blob_tool/Makefile`).

```bash
make install       # uv sync --inexact
make run           # http://127.0.0.1:8000 (host venv, uses MPS/CUDA). PORT=… HOST=… to override
make dev           # same, with auto-reload on src/ changes
make test          # pytest; no model downloads needed
make sample-data   # openly licensed test clips into blob_tool/samples/
make install-sam   # optional EdgeTAM / SAM 3 engines (transformers, accelerate, timm)
make docker-build / docker-run / docker-stop   # CPU-only container
```

Run a pipeline without the UI:
```bash
cd blob_tool && uv run python -c "from pathlib import Path; from blob_tool import pipeline; \
print(pipeline.run(Path('samples/x.mp4'), Path('workspace/x'), pipeline.PipelineParams(), lambda *a: None))"
```

## Layout (`blob_tool/`)

- `src/blob_tool/pipeline.py`: the stages are prepare (normalise to CFR H.264, ≤1080p), track, render, voice, then mux. Progress is reported as `(stage, fraction, message)`.
- `src/blob_tool/tracking/`: engines behind the `Tracker` protocol (`base.py`). They yield `FrameResult` with per-object bool masks.
  - `yolo.py`: the default, YOLO26-seg with BoT-SORT using `botsort_classroom.yaml` (track_buffer 150, ReID on, `conf=0.1` on purpose).
  - `edgetam.py`: YOLO decides the IDs and EdgeTAM decides the masks, re-seeded every `chunk_frames`.
  - `sam3.py`: SAM 3 text prompt in streaming mode.
  - `__init__.py`: the engine registry. Heavy imports are lazy.
- `src/blob_tool/render.py`:
  - `mask_to_polygons` smooths on the bbox crop at ≤480 px (`WORK_SIZE`), which matters for speed.
  - `TrackStore` stitches broken tracks, drops short ones and renumbers IDs.
  - It also writes `tracks.json` and draws the blob video **from polygons only**.
- `src/blob_tool/voice.py`: McAdams formant shift (LPC pole angle^α) + optional pitch shift.
- `src/blob_tool/jobs.py`: job dirs in `workspace/<id>/`, one sequential worker thread, `PUBLIC_FILES` allow-list.
- `src/blob_tool/app.py`: FastAPI. `/api/config`, `/api/jobs` (CRUD), `/api/jobs/{id}/files/{name}` (Range-capable), `/api/jobs/{id}/discard-original`, plus the static `web/`.
- `web/`: vanilla JS/CSS with no build step. A canvas overlay draws polygons from `tracks.json`. Deep links take the form `#job=<id>&mode=overlay&t=4&sel=3`.
- `scripts/fetch_samples.py`: sample downloader. It writes `samples/SOURCES.md` with licenses.

## Invariants — do not break

- **No original pixels in outputs.** Blob frames are drawn only from simplified polygons on `BACKGROUND_BGR`. `tests/test_render.py` asserts this. `STAGE_BG` in `web/app.js` and `--stage-bg` in `style.css` must match `BACKGROUND_BGR`.
- **One fixed α per recording** for the voice transform. This keeps speakers distinguishable for diarization, so do not randomise it per utterance.
- **Originals are deletable.** The raw upload is always deleted after processing. `original.mp4`/`original.wav` exist only for before/after comparison and are removed by "discard original".
- **Never commit media or weights.** `workspace/`, `samples/`, `models/`, `*.pt` are gitignored. Footage of children must never enter git.
- Keep the phone path in mind: prefer CoreML-exportable models and streaming-friendly (frame/buffer-based) processing.

## Gotchas (learned the hard way)

- `make install` must stay `uv sync --inexact`. A plain `uv sync` uninstalls the optional `sam` extra.
- `pyproject.toml` pins `torch`/`torchvision` to the **CPU index on Linux** (`[tool.uv.sources]`). Without it the Docker image is about 20 GB of CUDA libraries instead of about 4.7 GB. macOS uses the PyPI wheel (MPS).
- Docker on macOS cannot use the Apple GPU, which is why `make run` is native uv and Docker is CPU-only.
- The EdgeTAM transformers checkpoint is `yonigozlan/EdgeTAM-hf` (`facebook/EdgeTAM` is the raw `.pt`), and it needs `timm`.
- `facebook/sam3` is gated. It needs license acceptance and `uv run hf auth login`. The official Meta repo is CUDA-only, so use the transformers implementation.
- The repo is **AGPL-3.0** (`LICENSE`) because Ultralytics YOLO is AGPL-3.0. A closed-source app would need an enterprise license or a swap to RF-DETR/RT-DETR.
- Any new model, dataset, or adapted code must be added to the *Third-party components* table and *References* in `blob_tool/README.md`, with its license verified at the source. Sample media are never committed (the Pexels License forbids redistribution).
- Headless Chrome screenshots do not paint `<video>` layers, and seek events do not fire under `--virtual-time-budget`. Check playback in a real browser.
- AMI `ES2002a` is a 352×288 fisheye. Track fragmentation there (~24 IDs for 4 people) comes from the source quality, not a pipeline bug.

## Conventions

- Python ≥3.11, managed by uv. Use `uv run …`, never bare pip.
- Match the existing style: dataclasses for params (`TrackParams`, `BlobParams`, `VoiceParams`, built by `PipelineParams.from_dict`), short module docstrings, few comments.
- New tracking engines implement `Tracker.run(...)` and register in `tracking/__init__.py` with a lazy import.
- Tests use a fake tracker (`tests/test_pipeline_smoke.py`), so they stay download-free.
