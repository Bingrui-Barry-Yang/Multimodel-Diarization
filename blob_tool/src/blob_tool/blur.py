"""Light whole-frame blur of the original, alone or under blob / pose overlays.

The frame is area-downsampled by ``strength`` and scaled back up: a soft blur that keeps motion and
body edges readable and cannot be undone, since many originals map to the same output (a Gaussian
blur, by contrast, can be partly deconvolved). With ``faces`` on, each head found by the pose
keypoints (nose, eyes, ears) gets an extra Gaussian blur inside an ellipse about its own size.
Detection and pose always run on the clear video; only the saved output is blurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .render import TrackStore, color_for, draw_skeletons

LAYERS = {  # file → (blobs, pose) drawn over the blurred frame
    "blurred.mp4": (False, False),
    "blurred_blobs.mp4": (True, False),
    "blurred_pose.mp4": (False, True),
    "blurred_blobpose.mp4": (True, True),
}
HEAD = [0, 1, 2, 3, 4]  # COCO nose, eyes, ears
SHOULDERS = (5, 6)
BLOB_ALPHA = 0.45  # blob fill opacity over the blurred frame

Ellipse = tuple[float, float, float, float]  # cx, cy, semi-axis x, semi-axis y


@dataclass
class BlurParams:
    enabled: bool = False
    strength: float = 2.5  # downsampling factor of the whole frame
    faces: bool = True  # extra blur on heads located by pose keypoints

    def __post_init__(self) -> None:
        if not 1.5 <= self.strength <= 6:
            raise ValueError("strength must be 1.5–6")


def blur_base(frame: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, round(w / strength)), max(1, round(h / strength))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def face_ellipses(pose_row, min_score: float) -> list[Ellipse]:
    """An ellipse about the size of each head with at least two confident head keypoints."""
    out = []
    for _, k in pose_row or []:
        head = k[HEAD][k[HEAD, 2] >= min_score]
        if len(head) < 2:
            continue
        size = max(1.2 * np.ptp(head[:, 0]), 6.0)
        if k[SHOULDERS[0], 2] >= min_score and k[SHOULDERS[1], 2] >= min_score:
            size = max(size, 0.45 * abs(k[SHOULDERS[0], 0] - k[SHOULDERS[1], 0]))
        d = 1.3 * size
        cx, cy = head[:, :2].mean(0)
        out.append((float(cx), float(cy - 0.05 * d), d / 2, 0.6 * d))
    return out


def blur_faces(img: np.ndarray, ellipses: list[Ellipse]) -> np.ndarray:
    h, w = img.shape[:2]
    for cx, cy, ax, ay in ellipses:
        m = ax + ay  # room for the blur and the feathered edge
        x0, y0 = max(0, int(cx - ax - m)), max(0, int(cy - ay - m))
        x1, y1 = min(w, int(cx + ax + m) + 1), min(h, int(cy + ay + m) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = img[y0:y1, x0:x1].astype(np.float32)
        blurred = cv2.GaussianBlur(crop, (0, 0), max(1.0, ax / 2))
        mask = np.zeros(crop.shape[:2], np.float32)
        cv2.ellipse(mask, (round(cx - x0), round(cy - y0)), (max(1, round(ax)), max(1, round(ay))), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, ax / 4))[..., None]
        img[y0:y1, x0:x1] = (crop * (1 - mask) + blurred * mask).round().astype(np.uint8)
    return img


def blur_frame(frame: np.ndarray, pose_row, params: BlurParams, min_score: float = 0.3) -> np.ndarray:
    out = blur_base(frame, params.strength)
    return blur_faces(out, face_ellipses(pose_row, min_score)) if params.faces else out


def draw_overlay_blobs(img: np.ndarray, row) -> np.ndarray:
    """Translucent blobs with a solid outline, so the blurred people stay visible underneath."""
    fill = img.copy()
    t = max(2, round(min(img.shape[:2]) / 360))
    for oid, polys in row:
        r, g, b = color_for(oid - 1)
        cv2.fillPoly(fill, [p.astype(np.int32) for p in polys], (b, g, r), lineType=cv2.LINE_AA)
    img = cv2.addWeighted(fill, BLOB_ALPHA, img, 1 - BLOB_ALPHA, 0)
    for oid, polys in row:
        r, g, b = color_for(oid - 1)
        cv2.polylines(img, [p.astype(np.int32) for p in polys], True, (b, g, r), t, cv2.LINE_AA)
    return img


def layer_frame(base: np.ndarray, row, pose_row, blobs: bool, pose: bool, min_score: float = 0.3) -> np.ndarray:
    img = base.copy()
    if blobs:
        img = draw_overlay_blobs(img, row)
    if pose and pose_row:
        draw_skeletons(img, pose_row, min_score, on_blobs=blobs)
    return img


def render_videos(video: Path, store: TrackStore, poses, params: BlurParams, dsts: dict[str, Path],
                  min_score: float = 0.3, progress=None) -> None:
    """Read the original once and write every requested layer (``dsts`` maps a LAYERS name to a path)."""
    cap = cv2.VideoCapture(str(video))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {name: cv2.VideoWriter(str(p), fourcc, store.fps, (store.width, store.height)) for name, p in dsts.items()}
    n = len(store.frames)
    try:
        if not all(w.isOpened() for w in writers.values()):
            raise RuntimeError("could not open video writer")
        for i, row in enumerate(store.frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"original ended at frame {i} of {n}")
            pose_row = poses[i] if poses is not None and i < len(poses) else None
            base = blur_frame(frame, pose_row, params, min_score)
            for name, writer in writers.items():
                writer.write(layer_frame(base, row, pose_row, *LAYERS[name], min_score))
            if progress and i % 10 == 0:
                progress((i + 1) / n, f"blurring frame {i + 1}/{n}")
    finally:
        cap.release()
        for w in writers.values():
            w.release()
