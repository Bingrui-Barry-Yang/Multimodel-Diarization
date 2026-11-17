"""Blurred versions of the original: people keep their motion and outline, faces lose their identity.

strong (blurred.mp4): the whole frame is area-downsampled to ``detail`` px on its long side, so fine
  detail (including faces the detector missed, name tags, writing) is gone for good; every head is
  replaced by ``head_blocks``² block means plus fresh noise; crisp outlines from the tracked polygons
  are drawn on top so edges and IDs survive.
face (faceblur.mp4): the baseline of Li et al. (AIED 2026), Gaussian blur on head boxes only and the
  rest left as it is. Weaker: blurred faces can be recognised, and missed faces are not touched.

Heads are located from the pose keypoints (nose, eyes, ears) of the clear video, or else the top of
the blob. Detection and pose always run on the clear frames; only the saved output is blurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import TrackStore, color_for

METHODS = {"strong": "blurred.mp4", "face": "faceblur.mp4"}
BLUR_STREAM = 3  # sub-stream of the job seed (voice uses 1 and 2)
HEAD = [0, 1, 2, 3, 4]  # COCO nose, eyes, ears
SHOULDERS = (5, 6)

Box = tuple[int, int, int, int]  # x0, y0, x1, y1


@dataclass
class BlurParams:
    method: str = "strong"
    detail: int = 160  # long side of the downsampled frame (strong)
    head_blocks: int = 3  # blocks across each head (strong)
    noise: float = 12.0  # std of the per-block noise, in grey levels (strong)
    outlines: bool = True  # draw the blob outlines (strong)

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {list(METHODS)}")
        if not 32 <= self.detail <= 640:
            raise ValueError("detail must be 32–640")
        if not 1 <= self.head_blocks <= 8:
            raise ValueError("head_blocks must be 1–8")


def head_boxes(row, pose_row, min_score: float, width: int, height: int) -> list[Box]:
    """One box per tracked object: the top of its blob, joined with a box around its head keypoints.

    Keypoints alone can sit too high or too small on low-resolution footage, so they only ever add
    coverage (for a person bending over, whose head is not at the top of the blob)."""
    kpts = dict(pose_row or [])
    boxes = []
    for oid, polys in row:
        pts = np.vstack(polys)
        bx0, by0 = pts.min(0)
        bx1, by1 = pts.max(0) + 1
        bw, bh = bx1 - bx0, by1 - by0
        k = kpts.get(oid)
        head = k[HEAD][k[HEAD, 2] >= min_score] if k is not None else np.empty((0, 3))
        x0, y0, x1, y1 = bx0, by0, bx1, by0 + min(bh, max(0.25 * bh, 0.6 * bw))  # top of the blob
        if len(head):
            cx, cy = head[:, :2].mean(0)
            size = max(2.2 * np.ptp(head[:, 0]), 0.3 * bw, 8.0)
            if k[SHOULDERS[0], 2] >= min_score and k[SHOULDERS[1], 2] >= min_score:
                size = max(size, 0.8 * abs(k[SHOULDERS[0], 0] - k[SHOULDERS[1], 0]))
            size *= 1.3
            x0, y0 = min(x0, cx - size / 2), min(y0, cy - 0.6 * size)
            x1, y1 = max(x1, cx + size / 2), max(y1, cy + 0.6 * size)
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def strong(frame: np.ndarray, boxes: list[Box], params: BlurParams, rng: np.random.Generator) -> np.ndarray:
    h, w = frame.shape[:2]
    s = min(1.0, params.detail / max(h, w))
    small = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    n = params.head_blocks
    for x0, y0, x1, y1 in boxes:  # from the already downsampled frame, so heads hold even less
        region = out[y0:y1, x0:x1]
        blocks = cv2.resize(region, (n, n), interpolation=cv2.INTER_AREA).astype(np.float64)
        blocks += rng.normal(0.0, params.noise, blocks.shape)
        blocks = np.clip(blocks, 0, 255).astype(np.uint8)
        out[y0:y1, x0:x1] = cv2.resize(blocks, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return out


def face_only(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    out = frame.copy()
    for x0, y0, x1, y1 in boxes:
        region = np.ascontiguousarray(out[y0:y1, x0:x1])
        out[y0:y1, x0:x1] = cv2.GaussianBlur(region, (0, 0), max(1.0, (x1 - x0) / 3))
    return out


def draw_outlines(img: np.ndarray, row) -> np.ndarray:
    t = max(2, round(min(img.shape[:2]) / 360))
    for oid, polys in row:
        r, g, b = color_for(oid - 1)
        cv2.polylines(img, [p.astype(np.int32) for p in polys], True, (b, g, r), t, cv2.LINE_AA)
    return img


def blur_frame(frame: np.ndarray, row, pose_row, params: BlurParams, rng: np.random.Generator,
               min_score: float = 0.3) -> np.ndarray:
    h, w = frame.shape[:2]
    boxes = head_boxes(row, pose_row, min_score, w, h)
    if params.method == "face":
        return face_only(frame, boxes)
    out = strong(frame, boxes, params, rng)
    return draw_outlines(out, row) if params.outlines else out


def render_video(video: Path, store: TrackStore, poses, params: BlurParams, seed: int, dst: Path,
                 min_score: float = 0.3, progress=None) -> None:
    rng = np.random.default_rng([seed, BLUR_STREAM])
    cap = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), store.fps, (store.width, store.height))
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")
    n = len(store.frames)
    try:
        for i, row in enumerate(store.frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"original ended at frame {i} of {n}")
            pose_row = poses[i] if poses is not None and i < len(poses) else None
            writer.write(blur_frame(frame, row, pose_row, params, rng, min_score))
            if progress and i % 10 == 0:
                progress((i + 1) / n, f"blurring frame {i + 1}/{n}")
    finally:
        cap.release()
        writer.release()
