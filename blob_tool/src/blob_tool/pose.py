"""Top-down pose: RTMPose (rtmlib, ONNX Runtime) on each tracked blob's box → COCO-17 keypoints per blob ID.

Running on the tracker's boxes keeps every skeleton tied to its blob (same ID, same colour) and works
on small, low-resolution students where one-stage multi-person models miss most people.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .render import TrackStore
from .tracking.yolo import MODELS_DIR

KEYPOINTS = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
             "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
             "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
            (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
_SDK = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
MODELS = {  # RTMPose trained on Body7, 256x192 input
    "s": _SDK + "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    "m": _SDK + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
}

PoseRow = list[tuple[int, np.ndarray]]  # [(blob id, 17x3 [x, y, score] in pixels)]


@dataclass
class PoseParams:
    enabled: bool = True
    model: str = "m"  # s / m
    min_score: float = 0.3  # keypoints below this are not drawn


@lru_cache(maxsize=2)
def load_model(size: str):
    from .tracking.base import pick_device

    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "rtmlib"))  # rtmlib's download cache
    from rtmlib import RTMPose

    return RTMPose(MODELS.get(size, MODELS["m"]), model_input_size=(192, 256), device=pick_device())


def estimate(video: Path, store: TrackStore, params: PoseParams,
             progress: Callable[[float, str], None]) -> list[PoseRow]:
    model = load_model(params.model)
    cap = cv2.VideoCapture(str(video))
    out: list[PoseRow] = []
    n = len(store.frames)
    try:
        for i, row in enumerate(store.frames):
            ok, frame = cap.read()
            if not ok:
                out.extend([] for _ in range(n - i))
                break
            ids, boxes = [], []
            for oid, polys in row:
                pts = np.vstack(polys)
                boxes.append([*pts.min(0), *pts.max(0)])
                ids.append(oid)
            poses: PoseRow = []
            if boxes:
                kpts, scores = model(frame, bboxes=boxes)
                poses = [(oid, np.column_stack([k, s])) for oid, k, s in zip(ids, kpts, scores)]
            out.append(poses)
            if i % 10 == 0:
                progress((i + 1) / n, f"pose frame {i + 1}/{n}")
    finally:
        cap.release()
    return out


def write_json(path: Path, poses: list[PoseRow], width: int, height: int) -> None:
    def flat(k: np.ndarray) -> list[float]:
        return np.column_stack([(k[:, :2] / [width, height]).round(4), k[:, 2].round(2)]).ravel().tolist()

    path.write_text(json.dumps({
        "keypoints": KEYPOINTS,
        "skeleton": SKELETON,
        # frames[i] = [[object_id, x0, y0, s0, x1, y1, s1, …17 keypoints, x/y normalised], …]
        "frames": [[[oid, *flat(k)] for oid, k in row] for row in poses],
    }, separators=(",", ":")))
