"""RF-DETR-Seg instance segmentation + BoT-SORT (Roboflow `rfdetr` + `trackers`, both Apache-2.0)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .base import FrameResult, ProgressFn, TrackedObject, TrackParams, pick_device
from .yolo import MODELS_DIR

VARIANTS = {"n": "RFDETRSegNano", "s": "RFDETRSegSmall", "m": "RFDETRSegMedium", "l": "RFDETRSegLarge"}


def coco_names() -> dict[int, str]:
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return dict(COCO_CLASSES)


@lru_cache(maxsize=2)
def load_model(size: str):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("RF_HOME", str(MODELS_DIR))  # weights cache, read at download time
    import rfdetr

    return getattr(rfdetr, VARIANTS.get(size, VARIANTS["s"]))(device=pick_device())


class RfDetrTracker:
    name = "rfdetr"

    def run(self, video: Path, params: TrackParams, total_frames: int, progress: ProgressFn) -> Iterator[FrameResult]:
        from trackers import BoTSORTTracker

        model = load_model(params.model_size)
        names = coco_names()
        wanted = {cid for cid, n in names.items() if n in params.classes}
        if not wanted:
            raise ValueError(f"none of {params.classes} are known classes")

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # classroom tuning: people sit still and get occluded, so keep lost tracks ~5 s; camera motion
        # compensation handles handheld phones. Thresholds picked on the AMI/ClassBank samples.
        tracker = BoTSORTTracker(lost_track_buffer=int(5 * fps), frame_rate=fps, track_activation_threshold=0.4,
                                 high_conf_det_threshold=0.35)
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                det = model.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), threshold=params.conf,
                                    include_source_image=False)
                det = det[np.isin(det.class_id, list(wanted))]
                det = tracker.update(det, frame)
                objects: list[TrackedObject] = []
                if det.mask is not None:
                    for m, tid, c, s in zip(det.mask, det.tracker_id, det.class_id, det.confidence):
                        if tid is not None and tid >= 0:
                            objects.append(TrackedObject(int(tid), names[int(c)], float(s), np.ascontiguousarray(m)))
                i += 1
                progress(i / max(total_frames, 1), f"tracking frame {i}/{total_frames}")
                yield FrameResult(i - 1, objects)
        finally:
            cap.release()
