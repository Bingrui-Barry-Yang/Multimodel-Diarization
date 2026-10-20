"""Common types for tracking engines."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

ProgressFn = Callable[[float, str], None]


@dataclass
class TrackParams:
    classes: list[str] = field(default_factory=lambda: ["person"])
    model_size: str = "s"  # n / s / m / l / x for YOLO weights
    imgsz: int = 960  # larger helps with small students at the back of the room
    conf: float = 0.1  # low on purpose: BoT-SORT's second stage recovers weak detections
    text_prompt: str = "person"  # SAM 3 only
    chunk_frames: int = 48  # EdgeTAM: re-seed from detector every N frames


@dataclass
class TrackedObject:
    track_id: int
    label: str
    score: float
    mask: np.ndarray  # bool, H x W, original video resolution


@dataclass
class FrameResult:
    index: int
    objects: list[TrackedObject]


class Tracker(Protocol):
    name: str

    def run(self, video: Path, params: TrackParams, total_frames: int, progress: ProgressFn) -> Iterator[FrameResult]:
        """Yield one FrameResult per decoded frame, in order."""
        ...


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
