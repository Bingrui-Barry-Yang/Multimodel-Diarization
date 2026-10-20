"""YOLO26 instance segmentation + BoT-SORT multi-object tracking (Ultralytics)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import FrameResult, ProgressFn, TrackedObject, TrackParams, pick_device

TRACKER_CFG = Path(__file__).with_name("botsort_classroom.yaml")
MODELS_DIR = Path(os.environ.get("BLOB_MODELS_DIR", Path(__file__).resolve().parents[3] / "models"))


@lru_cache(maxsize=4)
def load_model(size: str):
    from ultralytics import YOLO

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return YOLO(str(MODELS_DIR / f"yolo26{size}-seg.pt"))


def class_names() -> list[str]:
    return list(load_model("n").names.values())


def class_ids(model, names: list[str]) -> list[int]:
    lookup = {v: k for k, v in model.names.items()}
    ids = [lookup[n] for n in names if n in lookup]
    if not ids:
        raise ValueError(f"none of {names} are known classes")
    return ids


class YoloTracker:
    name = "yolo"

    def run(self, video: Path, params: TrackParams, total_frames: int, progress: ProgressFn) -> Iterator[FrameResult]:
        model = load_model(params.model_size)
        model.predictor = None  # fresh tracker state for every video
        results = model.track(
            source=str(video),
            stream=True,
            persist=False,
            tracker=str(TRACKER_CFG),
            classes=class_ids(model, params.classes),
            imgsz=params.imgsz,
            conf=params.conf,
            retina_masks=True,  # masks at original resolution
            device=pick_device(),
            verbose=False,
        )
        for i, r in enumerate(results):
            objects: list[TrackedObject] = []
            if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
                masks = r.masks.data.cpu().numpy() > 0.5
                ids = r.boxes.id.int().cpu().tolist()
                cls = r.boxes.cls.int().cpu().tolist()
                conf = r.boxes.conf.cpu().tolist()
                for m, tid, c, s in zip(masks, ids, cls, conf):
                    objects.append(TrackedObject(int(tid), model.names[c], float(s), np.ascontiguousarray(m)))
            progress((i + 1) / max(total_frames, 1), f"tracking frame {i + 1}/{total_frames}")
            yield FrameResult(i, objects)
