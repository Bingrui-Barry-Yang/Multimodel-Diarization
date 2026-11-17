# CLAUDE.md

## Project

This project does privacy-preserving classroom analysis. Raw classroom video becomes **tracked colour blobs + altered voice**, so that nothing identifies the children. Local multimodal models and traditional diarization then run on that processed data. The research question is whether models can still extract speaker and turn-taking features from blobs alone.

- **Stage 1 (done):** `blob_tool/`, a localhost "filter" app. Upload a video, then compare before/after, inspect objects, and download the blob video, altered audio and `tracks.json`.
- **Stage 2 (next):** diarization on the outputs (`tracks.json` + `altered.wav`). The AMI sample with ground-truth `.rttm` is the evaluation starting point; ClassBank lessons (`.rttm` + `.transcript.tsv`) give real-classroom ground truth.
- **End goal:** the same tool on a phone, recording live and saving **only** the processed output, never the original.

## Commands

All targets work from the repo root (the root `Makefile` delegates to `blob_tool/Makefile`).

```bash
make install       # uv sync --inexact
make run           # http://127.0.0.1:8000 (host venv, uses MPS/CUDA). PORT=… HOST=… to override
make dev           # same, with auto-reload on src/ changes
make test          # pytest; no model downloads needed
make sample-data   # one busiest-30 s clip per dataset (AMI + each ClassBank corpus) → blob_tool/samples/sample/
make all-data      # every recording in samples/raw/, full length → blob_tool/samples/all/
make install-sam   # optional EdgeTAM / SAM 3 engines (transformers, accelerate, timm)
make install-llm   # optional Gemini client; key goes in blob_tool/.env (GEMINI_API_KEY=…), loaded by the Makefile
make docker-build / docker-run / docker-stop   # CPU-only container
```

Run a pipeline without the UI:
```bash
cd blob_tool && uv run python -c "from pathlib import Path; from blob_tool import pipeline; \
print(pipeline.run(Path('samples/x.mp4'), Path('workspace/x'), pipeline.PipelineParams(), lambda *a: None))"
```

## Layout (`blob_tool/`)

- `src/blob_tool/pipeline.py`: the stages are prepare (normalise to CFR H.264, ≤1080p), track, pose, render, voice, mux, then blur (only when enabled). Render writes `processed.mp4`, plus `pose.mp4` and `blobpose.mp4` when pose is on. Progress is reported as `(stage, fraction, message)`.
- `src/blob_tool/tracking/`: engines behind the `Tracker` protocol (`base.py`). They yield `FrameResult` with per-object bool masks.
  - `rfdetr.py`: the default, RF-DETR-Seg + BoT-SORT from Roboflow `trackers` (Apache-2.0, CoreML-exportable). Weights go to `models/` via `RF_HOME`. Tracker thresholds were tuned on the samples (ByteTrack gave far more IDs and false positives).
  - `yolo.py`: YOLO26-seg with BoT-SORT using `botsort_classroom.yaml` (track_buffer 150, ReID on, `conf=0.1` on purpose). AGPL.
  - `edgetam.py`: YOLO decides the IDs and EdgeTAM decides the masks, re-seeded every `chunk_frames`.
  - `sam3.py`: SAM 3 text prompt in streaming mode.
  - `__init__.py`: the engine registry. Heavy imports are lazy.
- `src/blob_tool/render.py`:
  - `mask_to_polygons` smooths on the bbox crop at ≤480 px (`WORK_SIZE`), which matters for speed.
  - `TrackStore` stitches broken tracks, drops short ones and renumbers IDs.
  - It also writes `tracks.json` and draws the blob video **from polygons only**.
- `src/blob_tool/pose.py`: top-down RTMPose-m (rtmlib/ONNX) on each tracked blob's box, so skeletons share the blob ID and colour. Writes `poses.json`. Weights are in `models/rtmlib/` (via `TORCH_HOME`).
- `src/blob_tool/blur.py`: opt-in light blur of the original. Whole frame area-downsampled by `strength` (default 2.5) and scaled back up; optional head-sized Gaussian ellipses from the pose head keypoints (`faces`); layers `blurred.mp4`, `blurred_blobs.mp4` (45 % blobs + outlines), `blurred_pose.mp4`, `blurred_blobpose.mp4` written in one pass. Runs as the last pipeline stage when `PipelineParams.blur.enabled`, or later via `POST /api/jobs/{id}/blur` (a `blur` task in `jobs.py`, `pipeline.blur` reads `tracks.json`/`poses.json` back). Detection/pose always use the clear video.
- `src/blob_tool/voice.py`: McAdams formant shift (LPC pole angle^α). By default α ∈ [0.75, 0.90] is drawn from the job seed (`voice.draw`) with no pitch shift, plus a −45 dB noise floor; `randomize=False` uses fixed α/pitch slider values.
- `src/blob_tool/jobs.py`: job dirs in `workspace/<id>/`, one sequential worker thread for both `process` and `blur` tasks (`job.tasks` tracks blur status), `PUBLIC_FILES` allow-list.
- `src/blob_tool/llm.py`: Gemini comparison. One isolated request per version (new client, upload streamed as `video.mp4`, `generate_content`, delete). `LlmManager` has its own worker thread and stores runs in `workspace/<id>/llm/<run>.json`.
- `src/blob_tool/app.py`: FastAPI. `/api/config` (includes `llm` status), `/api/jobs` (CRUD), `/api/jobs/{id}/files/{name}` (Range-capable), `/api/jobs/{id}/discard-original`, `/api/jobs/{id}/blur`, `/api/jobs/{id}/llm[/{run}]`, plus the static `web/`.
- `web/`: vanilla JS/CSS with no build step. A canvas overlay draws polygons from `tracks.json` (and skeletons from `poses.json` in Overlay mode). The Pose / Blobs+Pose modes swap the processed `<video>` source (`PROC_FILE`); the separate **Blur** switch (`state.blur`, key `b`) maps each mode to its blurred layer (`BLUR_FILE`), and Overlay is disabled while it is on. Mode buttons, downloads and the switch follow `job.files` via `refreshFiles`, so they update when a blur task finishes. Deep links take the form `#job=<id>&mode=overlay&t=4&sel=3`.
- `scripts/build_samples.py`: downloads AMI into `samples/raw/ami/`, pairs ClassBank videos with their `.cha` in `samples/raw/classbank/<corpus>/`, and writes `samples/<sample|all>/<dataset>/[<corpus>/]<recording>/` with `.mp4` + `.rttm` (+ `.transcript.tsv`) and a `SOURCES.md` per mode.

## Invariants — do not break

- **No original pixels in blob/pose outputs.** Blob and pose frames are drawn only from simplified polygons and keypoints on `BACKGROUND_BGR`. `tests/test_render.py` asserts this. The blurred videos are the only opt-in exception: they must stay a function of the area-downsampled frame (`tests/test_blur.py` checks fine detail cannot change them), and all four count as sensitive in `llm.SENSITIVE` (never pre-selected, `confirm_original` required). `STAGE_BG` in `web/app.js` and `--stage-bg` in `style.css` must match `BACKGROUND_BGR`.
- **One fixed α per recording** for the voice transform. This keeps speakers distinguishable for diarization, so do not randomise it per utterance. The seed picks that one α per recording.
- **The seed is secret.** `PipelineParams.seed` (drawn with `secrets` when empty) lives only in the local `job.json`. Never write it, or the α it draws, into a served file, `tracks.json`, `poses.json`, a job `result` or an LLM request (`test_pipeline_smoke.py` checks the JSON outputs).
- **Originals are deletable.** The raw upload is always deleted after processing. `original.mp4`/`original.wav` exist only for before/after comparison and are removed by "discard original" (blurred videos stay; discard is refused while a blur task runs).
- **Never commit media or weights.** `workspace/`, `samples/`, `models/`, `*.pt` are gitignored. Footage of children must never enter git.
- Keep the phone path in mind: prefer CoreML-exportable models and streaming-friendly (frame/buffer-based) processing.

## Gotchas (learned the hard way)

- `make install` must stay `uv sync --inexact`. A plain `uv sync` uninstalls the optional `sam` and `llm` extras, and so does `uv sync --extra X` without `--inexact`.
- **`blob_tool/.env` holds the Gemini key. It is gitignored and dockerignored and must never be committed.** Only `.env.example` is tracked.
- Gemini free tier: Google may use the content to improve its products, and human reviewers may see it. The "original" version must stay opt-in per run (UI tick plus `confirm_original` enforced server-side). Never pre-select it, and never send it by default.
- LLM tests use a fake client (`tests/test_llm.py`). They check that versions are isolated, so keep each version as a separate request.
- `pyproject.toml` pins `torch`/`torchvision` to the **CPU index on Linux** (`[tool.uv.sources]`). Without it the Docker image is about 20 GB of CUDA libraries instead of about 4.7 GB. macOS uses the PyPI wheel (MPS).
- Docker on macOS cannot use the Apple GPU, which is why `make run` is native uv and Docker is CPU-only.
- The EdgeTAM transformers checkpoint is `yonigozlan/EdgeTAM-hf` (`facebook/EdgeTAM` is the raw `.pt`), and it needs `timm`.
- `facebook/sam3` is gated. It needs license acceptance and `uv run hf auth login`. The official Meta repo is CUDA-only, so use the transformers implementation.
- The repo is **AGPL-3.0** (`LICENSE`) because Ultralytics YOLO is AGPL-3.0. The phone/app path must use only the `rfdetr` engine. Removing AGPL from the repo would mean moving `ultralytics` to an extra and reseeding `edgetam` from RF-DETR (not done yet).
- `/api/config` class names come from `rfdetr.assets.coco_classes`, so starting the UI does not load any model. RF-DETR class ids are COCO category ids (person = 1).
- Any new model, dataset, or adapted code must be added to the *Third-party components* table and *References* in `blob_tool/README.md`, with its license verified at the source. Sample media are never committed (footage of people; most licenses forbid redistribution).
- Browsers may keep a cached `web/` after an update (StaticFiles sends no `Cache-Control`). Tell the user to hard-reload (`Cmd+Shift+R`) when new UI does not show up.
- Headless Chrome screenshots do not paint `<video>` layers, and seek events do not fire under `--virtual-time-budget`. Check playback in a real browser.
- ClassBank media need a free TalkBank login, so they are downloaded by hand, not by `make sample-data`. The license is CC BY-NC-SA 4.0 and every use must cite the corpus reference on its page (`CORPORA` in `build_samples.py`).
- Pexels clips were dropped: they have no usable classroom audio and no ground truth.
- Size the face blur from head keypoints only (about 1.3× head width). Joining it with the top of each blob and filling with mosaic blocks covered 40–60 % of a 320×240 ClassBank frame and hid everything.
- Do not shift pitch in the seeded voice path: the overlap-add `pitch_shift` (no phase alignment) makes speech grainy and unintelligible on classroom audio, and a secret pitch adds little against inversion.
- One-stage RTMO (rtmlib) misses most small students in low-res clips, and its default `score_thr` of 0.7 is far too high. That is why pose is top-down on the tracker boxes.
- `rtmlib` pulls in `opencv-python` and `opencv-contrib-python` next to `opencv-python-headless`. They are all the same version, so `cv2` works, but keep them in step if you pin OpenCV.
- AMI `ES2002a` is a 352×288 fisheye. Track fragmentation there (~24 IDs for 4 people) comes from the source quality, not a pipeline bug.

## Conventions

- Python ≥3.11, managed by uv. Use `uv run …`, never bare pip.
- Match the existing style: dataclasses for params (`TrackParams`, `BlobParams`, `VoiceParams`, built by `PipelineParams.from_dict`), short module docstrings, few comments.
- New tracking engines implement `Tracker.run(...)` and register in `tracking/__init__.py` with a lazy import.
- Tests use a fake tracker and a fake pose estimator (`tests/test_pipeline_smoke.py` patches `pipeline.get_tracker` and `pipeline.estimate_poses`), so they stay download-free.

## Workflow

- Before suggesting a commit, update this `CLAUDE.md` so it reflects the change (commands, layout, gotchas, stage status).
- Before a push, update `blob_tool/README.md` so it matches what is being pushed (usage, sample data, third-party table, references).
