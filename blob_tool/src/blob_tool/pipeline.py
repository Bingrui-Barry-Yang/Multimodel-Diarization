"""End-to-end processing: original video → blob video + anonymised audio.

Outputs in the job directory:
  original.mp4    browser-playable, constant-frame-rate copy of the input (for before/after only)
  original.wav    original audio, 16 kHz mono (for before/after only)
  altered.wav     anonymised audio
  processed.mp4   blob-only video + anonymised audio  ← the privacy-safe deliverable
  tracks.json     per-frame blob polygons and object metadata
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import soundfile as sf

from . import media, voice
from .render import BlobParams, TrackStore, mask_to_polygons, render_video
from .tracking import get_tracker
from .tracking.base import TrackParams

ProgressFn = Callable[[str, float, str], None]  # (stage, fraction, message)

STAGES = ["prepare", "track", "render", "voice", "mux"]
MAX_HEIGHT = 1080  # downscale 4K phone footage; blobs do not need more


@dataclass
class PipelineParams:
    engine: str = "yolo"
    track: TrackParams = field(default_factory=TrackParams)
    blob: BlobParams = field(default_factory=BlobParams)
    voice: voice.VoiceParams = field(default_factory=voice.VoiceParams)

    @classmethod
    def from_dict(cls, d: dict) -> PipelineParams:
        return cls(
            engine=d.get("engine", "yolo"),
            track=TrackParams(**d.get("track", {})),
            blob=BlobParams(**d.get("blob", {})),
            voice=voice.VoiceParams(**d.get("voice", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def run(src: Path, workdir: Path, params: PipelineParams, progress: ProgressFn) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)

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

    # 3. render blob-only video from polygons
    blob_raw = workdir / "blob_raw.mp4"
    render_video(store, blob_raw, lambda f, m: progress("render", f, m))
    progress("render", 1.0, "blob video rendered")

    # 4. voice
    altered = None
    if info.has_audio:
        progress("voice", 0.1, "anonymising voices")
        audio, sr = sf.read(workdir / "original.wav", dtype="float32")
        altered = workdir / "altered.wav"
        sf.write(altered, voice.anonymize(audio, sr, params.voice), sr)
        progress("voice", 1.0, "voices altered")
    else:
        progress("voice", 1.0, "no audio track")

    # 5. mux
    progress("mux", 0.2, "encoding processed video")
    media.to_browser_mp4(blob_raw, workdir / "processed.mp4", audio=altered)
    blob_raw.unlink(missing_ok=True)
    progress("mux", 1.0, "done")

    return {
        "width": info.width,
        "height": info.height,
        "fps": info.fps,
        "duration": info.duration,
        "frames": len(store.frames),
        "has_audio": info.has_audio,
        "objects": len(store.meta),
    }
