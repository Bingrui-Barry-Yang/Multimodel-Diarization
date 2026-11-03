# Blob Tool

A privacy filter for classroom video. It turns every person (or any other object class you choose) into a **coloured blob that keeps its colour across time**, and it **alters the voices**. The output contains no original pixels and no original voice, but it keeps *who is where, who moves, and who speaks when*. That is the signal a downstream diarization model needs.

```
raw video ──► normalise (CFR H.264) ──► detect + segment + track ──► smooth & simplify outlines ──► processed.mp4
          └─► audio 16 kHz mono ──► McAdams formant shift (+ optional pitch) ───────────────────────┘  (blobs + altered voice)
                                                                                tracks.json (per-frame polygons, IDs)
```

## Setup

Tested on macOS (Apple Silicon, M3 Max). Linux works the same way, and on Linux `make run` uses CUDA if it is available.

### 1. Prerequisites

| Tool | Why | Install (macOS) | Check |
|---|---|---|---|
| `make` | runs every command | ships with Xcode Command Line Tools: `xcode-select --install` | `make --version` |
| `uv` | Python + dependency manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | `uv --version` |
| `ffmpeg` / `ffprobe` | decode, normalise, mux video and audio | `brew install ffmpeg` | `ffmpeg -version` |
| Docker Desktop | *optional*, only for `make docker-*` | <https://www.docker.com/products/docker-desktop/> | `docker --version` |

You do not need to install Python yourself. uv picks a compatible interpreter (3.11–3.13) or downloads one automatically.

### 2. Get the code

```bash
git clone https://github.com/Bingrui-Barry-Yang/Multimodel-Dirazation.git
cd Multimodel-Dirazation
```

Every `make` target works from the repo root (it delegates to `blob_tool/`) or from inside `blob_tool/`.

### 3. Install dependencies

```bash
make install        # uv sync → creates blob_tool/.venv with torch, ultralytics, fastapi, opencv, …
```

The first install takes a minute or two and about 1.2 GB on disk (mostly PyTorch). On macOS this is the normal PyPI torch wheel, which includes Apple GPU (MPS) support. To check that the GPU is visible:

```bash
cd blob_tool && uv run python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

`False` still works, but tracking runs on the CPU and is several times slower.

### 4. Start the interface

```bash
make run            # → open http://127.0.0.1:8000
```

`make run` re-runs `make install` (instant when nothing changed) and then starts uvicorn on `127.0.0.1:8000`. Leave the terminal open and stop the server with `Ctrl+C`.
- Use a different port with `make run PORT=8080`.
- To reach it from another device on your network, run `make run HOST=0.0.0.0`. Note that this exposes your videos to the network.

**First processing run:** the RF-DETR-Seg weights (about 130 MB for *small*) download automatically into `blob_tool/models/`. That first job needs internet access and takes a few seconds longer.

### 5. Try it with sample videos (optional)

```bash
make sample-data    # one 30 s clip per dataset → blob_tool/samples/sample/
make all-data       # every downloaded recording, full length → blob_tool/samples/all/
```

ClassBank lessons need a manual download first (see [Test data](#test-data)); without them only AMI is built. Drag any `.mp4` from `blob_tool/samples/sample/` onto the page and click **Process video**. A 30 s sample takes about 40 s on an M3 Max with the default RF-DETR engine.

### 6. Run the tests

```bash
make test           # unit tests + an end-to-end pipeline test on a synthetic video (no model download)
```

### Optional: EdgeTAM and SAM 3 engines

```bash
make install-sam    # adds transformers, accelerate, timm (EdgeTAM weights, about 56 MB, download on first use)
```

Restart `make run`. The engines are then selectable in the UI and greyed out until installed. `make install` keeps this extra installed (it uses `uv sync --inexact`). A plain `uv sync` would remove it.

SAM 3 weights are gated. Accept the license at <https://huggingface.co/facebook/sam3>, then run `cd blob_tool && uv run hf auth login`.

### Optional: Docker

CPU only, because Docker on macOS cannot reach the Apple GPU. It is useful for reproducibility or for running on a Linux server.

```bash
make docker-build   # ~4.7 GB image (CPU-only torch on Linux)
make docker-run     # → http://localhost:8000, Ctrl+C to stop (or `make docker-stop`)
```

Jobs are stored in `blob_tool/workspace/` on the host and are shared with `make run`. Model weights live in a Docker volume, so they download only once.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg not found on PATH` in a job error | `brew install ffmpeg`, then restart `make run` |
| `address already in use` | another server is on 8000: `make run PORT=8001`, or stop the other one (`lsof -i :8000`) |
| first job fails while loading the model | check the terminal running `make run`; the first job downloads the detector weights and needs internet once |
| a job is stuck at "interrupted by server restart" | the server was stopped mid-job; delete it in the Jobs list and upload again |
| EdgeTAM / SAM 3 greyed out | `make install-sam` and restart; SAM 3 additionally needs the HF login above |
| processing is slow | check MPS (step 3); choose detector size *nano*; Docker is always CPU-only |
| `make docker-run` port clash with `make run` | `make docker-run PORT=8001` |

## Using the UI

1. Drop a video (or click to choose one). Pick the objects to blob (default `person`), the blob style and the voice settings, then click **Process**.
2. When it finishes:
   - **Original / Blobs / Overlay / Split** (keys `1`–`4`) switch the view. Split has a draggable before/after divider.
   - **Seek bar**: the strip behind it shows how many objects are visible over time. The selected object's presence is drawn as a coloured line on top.
   - **Objects panel**: click a row, or click a blob in the video, to select it. The other blobs dim, its outline and label are drawn, and the player jumps to it if it is off-screen. 👁 hides a blob.
   - **Audio**: altered, original or mute.
   - `space` plays and pauses, `←`/`→` step one frame (with `shift`, ten frames), `esc` deselects.
3. **Downloads**: processed video (blobs + altered voice), altered wav, and `tracks.json`.
4. **Delete original files** removes the only copies of the original pixels and voice (`original.mp4`, `original.wav`). The checkbox *Discard original after processing* does this automatically, which matches how the phone version will behave. The raw upload is always deleted after processing.

URLs are shareable inside your machine: `#job=<id>&mode=overlay&t=12.5&sel=3`.

## Engines

| Engine | What it does | Speed (M3 Max, 1080p) | Phone path |
|---|---|---|---|
| `rfdetr` (default) | [RF-DETR-Seg](https://github.com/roboflow/rf-detr) [14] instance masks (DINOv2 backbone) + BoT-SORT with camera-motion compensation from Roboflow [`trackers`](https://github.com/roboflow/trackers). Everything is Apache-2.0 | ~25 fps at 320×240 | `model.export(format="coreml")` → `.mlpackage` (experimental), or ONNX |
| `yolo` | [YOLO26-seg](https://docs.ultralytics.com/models/yolo26/) instance masks + BoT-SORT with ReID (`tracking/botsort_classroom.yaml`). AGPL-3.0, so not usable in a closed-source app | ~14 fps at 1080p | CoreML export, ANE |
| `edgetam` | YOLO tracks decide *who*. [EdgeTAM](https://huggingface.co/papers/2501.07256) (on-device SAM 2, CVPR 2025) decides *which pixels*, re-seeded every `chunk_frames` | ~2–3 fps | EdgeTAM runs at ~16 fps on iPhone 15 Pro Max |
| `sam3` | [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) text-prompt video tracking ("person", "chair", …), streaming mode | slow | no (server-class model) |

On the 30 s samples RF-DETR + BoT-SORT gives 5 / 35 / 19 IDs (AMI / Curtis / TIMSS) against YOLO's 5 / 43 / 20, with similar per-frame coverage. `edgetam` still uses YOLO for its IDs.

SAM 3 weights are gated. Accept the license at <https://huggingface.co/facebook/sam3>, then run `uv run hf auth login`.

After tracking, `render.TrackStore` **stitches** tracks. A track that disappears and a new ID that appears close by within `stitch_gap_seconds` (3 s) are merged, and tracks shorter than `min_track_seconds` are dropped. This fixes most ID switches caused by a student bending over a desk or walking behind the teacher.

### Blob styles
- `silhouette`: smoothed, simplified outline (keeps posture and gesture).
- `hull`: convex hull (keeps size and position, hides limbs).
- `ellipse`: fitted ellipse (keeps only position, size and orientation; the most abstract).

Blobs are drawn only from simplified polygons onto a solid background, so the original pixels cannot leak through. `tests/test_render.py` checks that every output pixel is background, blob colour, or an anti-aliased mix of the two.

## Voice alteration

`voice.py` implements the **McAdams-coefficient** anonymiser, the signal-processing baseline of the [VoicePrivacy Challenge](https://www.voiceprivacychallenge.org/). Each 20 ms frame is LPC-analysed, every formant pole angle φ becomes φ^α, and the residual is re-filtered. It needs no training data, runs faster than real time on a CPU, and is simple enough for AVAudioEngine/Core Audio on a phone.

The **same α is used for the whole recording**, so all speakers move through the same transform and stay distinguishable from each other. On the AMI sample, pairwise speaker distances (coarse spectral envelope) are preserved while the waveform correlation with the original drops to about 0.23. A per-utterance random α would hide identity more strongly, but it would break diarization. McAdams is a *light* anonymiser: an attacker who knows the method can partly invert it. For stronger privacy, the next step is neural voice conversion (kNN-VC, or StreamVC-style streaming VC) to a fixed target pool behind the same `anonymize()` interface.

## Outputs (`workspace/<job>/`)

| File | Contents | Privacy-safe |
|---|---|---|
| `processed.mp4` | blob-only video + altered audio | ✅ |
| `altered.wav` | altered audio, 16 kHz mono | ✅ |
| `tracks.json` | `objects[]` (id, label, colour, first/last frame) and `frames[i] = [[id, [x0,y0,x1,y1,…normalised], …], …]` | ✅ |
| `original.mp4`, `original.wav` | normalised copy of the input, kept only for before/after comparison | ❌ deleted by "discard original" |

`tracks.json` is the intended input for the next stage: a blob-only multimodal or diarization model. It lets you test whether a model can still tell who is speaking when all it gets is blobs.

## Test data

`scripts/build_samples.py` reads the original recordings from `samples/raw/<dataset>/` and writes them, with ground-truth speaker turns, into `samples/` (gitignored, never committed). **The recordings themselves are not redistributed by this repo.**

- `make sample-data`: one recording per dataset/corpus, cut to its busiest 30 s (most speaker changes; `--duration` to change), into `samples/sample/`.
- `make all-data`: every recording found under `samples/raw/`, full length, into `samples/all/`.

Each recording gets its own folder, `samples/<sample|all>/<dataset>/[<corpus>/]<recording>/`, holding `<recording>.mp4`, `<recording>.rttm` and, for ClassBank, `<recording>.transcript.tsv` (start, end, speaker, role, text). Sources, licenses and modifications are listed in `samples/<sample|all>/SOURCES.md`.

```text
samples/
  raw/ami/                          ES2002a.Corner.avi, ES2002a.Mix-Headset.wav, ES2002a.rttm (auto-downloaded)
  raw/classbank/<corpus>/           .cha transcripts + lesson videos (manual download)
  sample/ami/ES2002a/               ES2002a.mp4, ES2002a.rttm
  sample/classbank/Curtis/dec01a/   dec01a.mp4, dec01a.rttm, dec01a.transcript.tsv
  all/…                             same layout, full length
```

- **AMI Meeting Corpus `ES2002a`**: corner camera + mixed headset audio, 4 speakers, downloaded automatically into `samples/raw/ami/`. © the AMI Consortium, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([AMI license page](https://groups.inf.ed.ac.uk/ami/corpus/license.shtml)). Modification: video and audio muxed and re-encoded to H.264/AAC; cut to the busiest 30 s for `sample-data`. Useful for checking voice distinguishability now and for diarization evaluation later.
- **AMI ground-truth speaker turns** (`.rttm`): from [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup) (Apache-2.0), "only_words" setup by Landini et al. [9]. Modification: timestamps shifted and clipped to the cut window (`sample-data` only).
- **ClassBank classroom lessons** (manual download): real-speed classroom video with audio and CHAT transcripts (who said what, with millisecond timestamps), from TalkBank's ClassBank [11], licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Recommended corpora: [TIMSS-Math/USA](https://talkbank.org/class/access/TIMSS-Math/USA.html) (8th-grade math, doi:10.21415/T5Z30S, cite [12]) and [Curtis](https://talkbank.org/class/access/Curtis.html) (2nd-grade geometry, doi:10.21415/T5V890, cite [13]). Media need a free TalkBank login. On the corpus page download the transcript zip and, from its media folder, only the lesson videos you need (each is 30–50 MB). Unzip the transcripts into `samples/raw/classbank/<corpus>/` (folder names `TIMSS-Math-USA`, `Curtis`) and put the videos anywhere under that folder; each video is matched to the `.cha` with the same name, and transcripts without a video are skipped.
- For larger evaluation: the OUC-CGE classroom group engagement dataset [10] ([OSF](https://doi.org/10.17605/OSF.IO/BRD2C); adult participants; check the dataset's own terms before use). It is not auto-downloaded.

## Road to the phone

The goal is to record live and save only the processed output.
1. Export RF-DETR-Seg (`pip install "rfdetr[coreml]"`, then `RFDETRSegNano().export(format="coreml", coreml_precision="float16")`) and run it on the Neural Engine. Port BoT-SORT and the stitching step to Swift (they are a few hundred lines of Kalman filtering and IoU matching).
2. Optionally add EdgeTAM via CoreML for better masks.
3. Render blobs with Metal/Core Graphics from the polygons, and write only blob frames to `AVAssetWriter`. Camera frames never touch disk.
4. Apply McAdams per audio buffer in an `AVAudioEngine` tap. It is frame-based with 50 % overlap-add, so it streams with 20 ms of latency.

## License

This project is licensed under the **GNU AGPL-3.0** (see [`LICENSE`](../LICENSE) at the repo root), because it builds on Ultralytics YOLO, which is AGPL-3.0. If you deploy a modified version as a network service, you must offer its source to the users of that service. The default `rfdetr` engine and its tracker are Apache-2.0, so a closed-source phone app can be built on that path alone. The repo stays AGPL while it ships the `yolo` engine (and `edgetam`, which seeds from YOLO).

### Third-party components

| Component | Used for | License |
|---|---|---|
| [Ultralytics](https://github.com/ultralytics/ultralytics) + YOLO26-seg weights [1] | detection, segmentation, BoT-SORT tracking | AGPL-3.0 |
| [RF-DETR](https://github.com/roboflow/rf-detr) [14] (`rfdetr`, RF-DETR-Seg N–L weights) | default detection + segmentation | Apache-2.0 (only the Seg and N–L detection weights; the PML 1.0 "plus" models are not used) |
| [Roboflow trackers](https://github.com/roboflow/trackers) (clean-room BoT-SORT [2]) | default tracking | Apache-2.0 |
| [supervision](https://github.com/roboflow/supervision) | detection containers for RF-DETR / trackers | MIT |
| `src/blob_tool/tracking/botsort_classroom.yaml` | tracker settings, adapted from Ultralytics' `botsort.yaml` | AGPL-3.0 |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) [3] (weights via [`yonigozlan/EdgeTAM-hf`](https://huggingface.co/yonigozlan/EdgeTAM-hf)) | optional mask engine | Apache-2.0 |
| [SAM 3](https://github.com/facebookresearch/sam3) [4] | optional engine; weights are downloaded by each user after accepting Meta's license | SAM License |
| [Hugging Face Transformers](https://github.com/huggingface/transformers), accelerate, timm | running EdgeTAM / SAM 3 | Apache-2.0 |
| PyTorch, torchvision | inference | BSD-style |
| OpenCV, NumPy, SciPy, soundfile, lap | image, audio and assignment utilities | Apache-2.0 / BSD |
| FastAPI, Uvicorn, python-multipart | web server | MIT / BSD / Apache-2.0 |
| ClassBank / TalkBank corpora [11–13] (manual download, never committed) | classroom test clips with speaker-timed transcripts | CC BY-NC-SA 4.0 |
| FFmpeg (system install, not bundled) | video/audio decoding and encoding | LGPL/GPL, depending on build |

The McAdams voice anonymiser (`voice.py`) is an independent implementation of the method described in [5]. No code was copied from the VoicePrivacy Challenge baselines.

## References

1. G. Jocher, J. Qiu, M. Liu, S. Lyu, F. C. Akyon, M. E. Kalfaoglu, "Ultralytics YOLO26: Unified Real-Time End-to-End Vision Models," arXiv:2606.03748, 2026. https://docs.ultralytics.com/models/yolo26/
2. N. Aharon, R. Orfaig, B.-Z. Bobrovsky, "BoT-SORT: Robust Associations Multi-Pedestrian Tracking," arXiv:2206.14651, 2022.
3. C. Zhou et al., "EdgeTAM: On-Device Track Anything Model," CVPR 2025. arXiv:2501.07256.
4. N. Carion et al., "SAM 3: Segment Anything with Concepts," arXiv:2511.16719, 2025.
5. J. Patino, N. Tomashenko, M. Todisco, A. Nautsch, N. Evans, "Speaker anonymisation using the McAdams coefficient," Interspeech 2021. arXiv:2011.01130.
6. VoicePrivacy Challenge. https://www.voiceprivacychallenge.org/
7. Y. Xiong et al., "Efficient Track Anything," arXiv:2411.18933, 2024. (Evaluated as an on-device alternative; not used in code.)
8. J. Carletta et al., "The AMI Meeting Corpus: A Pre-announcement," Machine Learning for Multimodal Interaction (MLMI), 2005.
9. F. Landini, J. Profant, M. Diez, L. Burget, "Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization: theory, implementation and analysis on standard tasks," Computer Speech & Language, 2022. arXiv:2012.14952.
10. W. Lu, Y. Yang, R. Song, Y. Chen, T. Wang, C. Bian, "A Video Dataset for Classroom Group Engagement Recognition," Scientific Data 12, 644, 2025. doi:10.1038/s41597-025-04987-w.
11. B. MacWhinney, "The CHILDES Project: Tools for Analyzing Talk," 3rd ed., Lawrence Erlbaum, 2000. ClassBank: https://talkbank.org/class/
12. J. W. Stigler, R. Gallimore, J. Hiebert, "Using video surveys to compare classrooms and teaching across cultures: Examples and lessons from the TIMSS and TIMSS-R video studies," Educational Psychologist 35, 87–100, 2000.
13. R. Lehrer, C. L. Curtis, "Why are some solids perfect?," Teaching Children Mathematics 6(5), 324, 2000.
14. I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, N. Peri, "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers," ICLR 2026. arXiv:2511.09554.
