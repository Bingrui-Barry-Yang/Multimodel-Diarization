"""End-to-end processing: original video → blob video + anonymised audio.

Outputs in the job directory:
  original.mp4    browser-playable, constant-frame-rate copy of the input (for before/after only)
  original.wav    original audio, 16 kHz mono (for before/after only)
  altered.wav     anonymised audio
  processed.mp4   blob-only video + anonymised audio  ← the privacy-safe deliverable
  tracks.json     per-frame blob polygons and object metadata
  poses.json      per-frame COCO-17 keypoints per blob ID (pose enabled)
  pose.mp4        stick figures only + anonymised audio
  blobpose.mp4    blobs + stick figures + anonymised audio
  blurred.mp4     strongly blurred original + anonymised audio (opt-in, after processing, see blur.py)
  faceblur.mp4    original with only the heads blurred + anonymised audio (opt-in, paper baseline)
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

from . import blur as blurmod
from . import media, voice
from . import pose as posemod
from .pose import PoseParams
from .pose import estimate as estimate_poses
from .render import BlobParams, TrackStore, mask_to_polygons, render_video
from .tracking import get_tracker
from .tracking.base import TrackParams

ProgressFn = Callable[[str, float, str], None]  # (stage, fraction, message)

STAGES = ["prepare", "track", "pose", "render", "voice", "mux"]
MAX_HEIGHT = 1080  # downscale 4K phone footage; blobs do not need more


@dataclass
class PipelineParams:
    engine: str = "rfdetr"
    track: TrackParams = field(default_factory=TrackParams)
    blob: BlobParams = field(default_factory=BlobParams)
    voice: voice.VoiceParams = field(default_factory=voice.VoiceParams)
    pose: PoseParams = field(default_factory=PoseParams)
    seed: int | None = None  # secret per job: voice transform + blur noise; None = draw a new one

    @classmethod
    def from_dict(cls, d: dict) -> PipelineParams:
        return cls(
            engine=d.get("engine", "rfdetr"),
            track=TrackParams(**d.get("track", {})),
            blob=BlobParams(**d.get("blob", {})),
            voice=voice.VoiceParams(**d.get("voice", {})),
            pose=PoseParams(**d.get("pose", {})),
            seed=parse_seed(d.get("seed")),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def parse_seed(value) -> int | None:
    if value is None or value == "":
        return None
    seed = int(value)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def new_seed() -> int:
    return secrets.randbits(63)


def run(src: Path, workdir: Path, params: PipelineParams, progress: ProgressFn) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    if params.seed is None:
        params.seed = new_seed()

    # 1. prepare: normalise to CFR H.264 so frame indices line up with the audio timeline
    progress("prepare", 0.0, "probing input")
    info = media.probe(src)
    original = workdir / "original.mp4"
    media.normalize(src, original, fps=info.fps, max_height=MAX_HEIGHT)
    info = media.probe(original)
    if info.has_audio:
        media.extract_audio(original, workdir / "original.wav")
    progress("prepare", 1.0, f"{info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f}s")

    # 2. track: detector/segmenter/tracker → polygons per frame
    tracker = get_tracker(params.engine)
    store = TrackStore(info.width, info.height, info.fps)
    for fr in tracker.run(original, params.track, info.frames, lambda f, m: progress("track", f, m)):
        store.add_frame([(o.track_id, o.label, o.score, mask_to_polygons(o.mask, params.blob)) for o in fr.objects])
    store.finalize(params.blob.min_track_seconds, params.blob.stitch_gap_seconds)
    store.write_json(workdir / "tracks.json")
    progress("track", 1.0, f"{len(store.meta)} objects tracked")

    # 3. pose: top-down keypoints on each tracked blob
    poses = None
    if params.pose.enabled:
        poses = estimate_poses(original, store, params.pose, lambda f, m: progress("pose", f, m))
        posemod.write_json(workdir / "poses.json", poses, info.width, info.height)
        progress("pose", 1.0, f"{sum(len(r) for r in poses)} poses")
    else:
        progress("pose", 1.0, "pose estimation off")

    # 4. render blob / pose / blob+pose videos from polygons and keypoints only
    raws = {"processed.mp4": workdir / "blob_raw.mp4"}
    render_video(store, raws["processed.mp4"], lambda f, m: progress("render", f / 3 if poses else f, m))
    if poses is not None:
        raws["pose.mp4"] = workdir / "pose_raw.mp4"
        raws["blobpose.mp4"] = workdir / "blobpose_raw.mp4"
        ms = params.pose.min_score
        render_video(store, raws["pose.mp4"], lambda f, m: progress("render", (1 + f) / 3, m), poses, False, ms)
        render_video(store, raws["blobpose.mp4"], lambda f, m: progress("render", (2 + f) / 3, m), poses, True, ms)
    progress("render", 1.0, "videos rendered")

    # 5. voice
    altered = None
    if info.has_audio:
        progress("voice", 0.1, "anonymising voices")
        audio, sr = sf.read(workdir / "original.wav", dtype="float32")
        altered = workdir / "altered.wav"
        sf.write(altered, voice.anonymize(audio, sr, params.voice, params.seed), sr)
        progress("voice", 1.0, "voices altered")
    else:
        progress("voice", 1.0, "no audio track")

    # 6. mux
    for j, (name, raw) in enumerate(raws.items()):
        progress("mux", j / len(raws), f"encoding {name}")
        media.to_browser_mp4(raw, workdir / name, audio=altered)
        raw.unlink(missing_ok=True)
    progress("mux", 1.0, "done")

    return {
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "duration": info.duration,
        "frames": len(store.frames),
        "has_audio": info.has_audio,
        "objects": len(store.meta),
        "poses": poses is not None,
    }


def blur(workdir: Path, params: blurmod.BlurParams, seed: int, min_score: float,
         progress: Callable[[float, str], None]) -> str:
    """Opt-in blurred version of a finished job, from its original and its saved tracks and poses."""
    original = workdir / "original.mp4"
    if not original.exists():
        raise FileNotFoundError("the original was discarded, so there is nothing to blur")
    store = TrackStore.read_json(workdir / "tracks.json")
    poses_path = workdir / "poses.json"
    poses = posemod.read_json(poses_path, store.width, store.height) if poses_path.exists() else None
    name = blurmod.METHODS[params.method]
    raw = workdir / name.replace(".mp4", "_raw.mp4")
    altered = workdir / "altered.wav"
    try:
        blurmod.render_video(original, store, poses, params, seed, raw, min_score,
                             lambda f, m: progress(0.9 * f, m))
        progress(0.9, f"encoding {name}")
        media.to_browser_mp4(raw, workdir / name, audio=altered if altered.exists() else None)
    finally:
        raw.unlink(missing_ok=True)
    progress(1.0, "done")
    return name
