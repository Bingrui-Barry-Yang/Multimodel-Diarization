"""Thin ffmpeg / ffprobe wrappers."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise MediaError(f"{name} not found on PATH; install ffmpeg")
    return path


def _run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"{Path(args[0]).name} failed: {proc.stderr.strip()[-2000:]}")
    return proc.stdout


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration: float
    frames: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    out = _run([
        _bin("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise MediaError("no video stream found")
    num, den = (video.get("avg_frame_rate") or video.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(video.get("duration") or data["format"].get("duration") or 0.0)
    frames = int(video.get("nb_frames") or round(duration * fps))
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps or 30.0,
        duration=duration,
        frames=frames,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000) -> None:
    """Decode the first audio stream to mono 16-bit wav."""
    _run([
        _bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst),
    ])


def normalize(src: Path, dst: Path, fps: float, max_height: int = 1080) -> None:
    """Constant-frame-rate H.264 + AAC copy, downscaled to at most ``max_height`` (applies rotation metadata)."""
    _run([
        _bin("ffmpeg"), "-y", "-v", "error", "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-vf", f"scale=-2:'min({max_height},ih)'", "-fps_mode", "cfr", "-r", f"{fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dst),
    ])


def to_browser_mp4(src: Path, dst: Path, audio: Path | None = None) -> None:
    """Re-encode video to H.264/yuv420p (plays in every browser), optionally muxing a wav."""
    args = [_bin("ffmpeg"), "-y", "-v", "error", "-i", str(src)]
    if audio is not None:
        args += ["-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        args += ["-an"]
    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
    ]
    _run(args)
