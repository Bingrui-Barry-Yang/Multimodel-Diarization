"""Mask → blob polygons, and polygons → blob-only video.

The blob video is drawn *only* from simplified polygons on a solid background,
so no original pixel can leak into the output.
"""

from __future__ import annotations

import colorsys
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STYLES = ("silhouette", "hull", "ellipse")
BACKGROUND_BGR = (24, 20, 18)
WORK_SIZE = 480  # longest side used for mask smoothing


@dataclass
class BlobParams:
    style: str = "silhouette"
    smooth: float = 0.5  # 0 = raw mask outline, 1 = very rounded
    min_track_seconds: float = 0.5  # drop flickering tracks shorter than this
    stitch_gap_seconds: float = 3.0  # re-join a track that ends with one that starts nearby within this gap
    min_area_frac: float = 0.0005  # drop specks smaller than this fraction of the frame


def color_for(index: int) -> tuple[int, int, int]:
    """Distinct, stable RGB colour per object (golden-ratio hue walk)."""
    h = (0.08 + index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def mask_to_polygons(mask: np.ndarray, params: BlobParams) -> list[np.ndarray]:
    """Return blob outlines (int32 Nx2 pixel coords) for one object mask."""
    h, w = mask.shape
    bx, by, bw, bh = cv2.boundingRect(mask.view(np.uint8))
    if bw == 0 or bh == 0:
        return []

    # Work on the object's bounding box at a reduced resolution; blob outlines
    # are smoothed anyway, so full-resolution morphology only costs time.
    scale = min(1.0, WORK_SIZE / max(h, w))
    k = max(1, int(round(min(h, w) * scale * 0.03 * params.smooth))) | 1
    pad = 2 * k + 2
    x0, y0 = max(0, bx - int(pad / scale)), max(0, by - int(pad / scale))
    x1, y1 = min(w, bx + bw + int(pad / scale)), min(h, by + bh + int(pad / scale))
    m = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    if scale < 1.0:
        m = cv2.resize(m, (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))),
                       interpolation=cv2.INTER_AREA)
    if params.smooth > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = cv2.GaussianBlur(m, (k * 2 + 1, k * 2 + 1), 0)
    m = (m > 127).astype(np.uint8) * 255

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = params.min_area_frac * h * w * scale * scale
    contours = [
        (c.reshape(-1, 2).astype(np.float64) / scale + (x0, y0)).round().astype(np.int32).reshape(-1, 1, 2)
        for c in contours if cv2.contourArea(c) >= min_area
    ]
    if not contours:
        return []

    if params.style == "hull":
        return [cv2.convexHull(np.vstack(contours)).reshape(-1, 2)]
    if params.style == "ellipse":
        pts = np.vstack(contours)
        if len(pts) < 5:
            return []
        (cx, cy), (ew, eh), ang = cv2.fitEllipse(pts)
        poly = cv2.ellipse2Poly((int(cx), int(cy)), (int(ew / 2), int(eh / 2)), int(ang), 0, 360, 10)
        return [poly.reshape(-1, 2)]

    eps = 0.002 * (h + w) * (0.5 + params.smooth)
    out = []
    for c in contours:
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) >= 3:
            out.append(approx)
    return out


class TrackStore:
    """Accumulates per-frame polygons, then filters/renumbers tracks."""

    def __init__(self, width: int, height: int, fps: float):
        self.width, self.height, self.fps = width, height, fps
        self.frames: list[list[tuple[int, list[np.ndarray]]]] = []
        self.meta: dict[int, dict] = {}

    def add_frame(self, objects: list[tuple[int, str, float, list[np.ndarray]]]) -> None:
        idx = len(self.frames)
        row = []
        for tid, label, score, polys in objects:
            if not polys:
                continue
            c = _centroid(polys)
            m = self.meta.setdefault(tid, {"label": label, "first": idx, "last": idx, "count": 0, "score": 0.0,
                                           "first_c": c})
            m["last"] = idx
            m["last_c"] = c
            m["count"] += 1
            m["score"] += score
            row.append((tid, polys))
        self.frames.append(row)

    def _stitch(self, max_gap_frames: int) -> dict[int, int]:
        """Greedy track stitching: when the tracker loses someone (occlusion, a student
        bending over a desk) and re-detects them as a new ID close to where they
        vanished, merge the new ID into the old one. Returns {merged_id: kept_id}."""
        max_dist = 0.1 * float(np.hypot(self.width, self.height))
        merged: dict[int, int] = {}
        open_tracks: list[int] = []  # kept tracks, whose `last` may still be extended
        for tid in sorted(self.meta, key=lambda t: self.meta[t]["first"]):
            b = self.meta[tid]
            best, best_d = None, max_dist
            for aid in open_tracks:
                a = self.meta[aid]
                gap = b["first"] - a["last"]
                if a["label"] != b["label"] or gap <= 0 or gap > max_gap_frames:
                    continue
                d = float(np.hypot(*(np.subtract(a["last_c"], b["first_c"]))))
                if d < best_d:
                    best, best_d = aid, d
            if best is None:
                open_tracks.append(tid)
                continue
            a = self.meta[best]
            a["last"], a["last_c"] = b["last"], b["last_c"]
            a["count"] += b["count"]
            a["score"] += b["score"]
            merged[tid] = best
        return merged

    def finalize(self, min_track_seconds: float, stitch_gap_seconds: float = 0.0) -> None:
        """Stitch broken tracks, drop short ones and renumber the rest 1..N by first appearance."""
        merged = self._stitch(int(round(stitch_gap_seconds * self.fps))) if stitch_gap_seconds > 0 else {}
        for tid in merged:
            del self.meta[tid]
        min_frames = max(1, int(round(min_track_seconds * self.fps)))
        keep = sorted((m["first"], tid) for tid, m in self.meta.items() if m["count"] >= min_frames)
        remap = {tid: i + 1 for i, (_, tid) in enumerate(keep)}
        remap.update({t: remap[k] for t, k in merged.items() if k in remap})
        self.frames = [[(remap[t], p) for t, p in row if t in remap] for row in self.frames]
        self.meta = {remap[t]: m for t, m in self.meta.items() if t in remap}

    def to_json(self) -> dict:
        w, h = self.width, self.height

        def norm(poly: np.ndarray) -> list[float]:
            flat = (poly.astype(np.float64) / [w, h]).round(4).ravel()
            return flat.tolist()

        objects = []
        for oid in sorted(self.meta):
            m = self.meta[oid]
            objects.append({
                "id": oid,
                "label": m["label"],
                "color": "#%02x%02x%02x" % color_for(oid - 1),
                "first_frame": m["first"],
                "last_frame": m["last"],
                "frames_visible": m["count"],
                "mean_score": round(m["score"] / max(m["count"], 1), 3),
            })
        return {
            "width": w,
            "height": h,
            "fps": self.fps,
            "frame_count": len(self.frames),
            "objects": objects,
            # frames[i] = [[object_id, [x0,y0,x1,y1,...normalised], ...more polys], ...]
            "frames": [[[oid, *[norm(p) for p in polys]] for oid, polys in row] for row in self.frames],
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json(), separators=(",", ":")))


def _centroid(polys: list[np.ndarray]) -> tuple[float, float]:
    pts = np.vstack(polys).astype(np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def draw_blobs(store: TrackStore, index: int) -> np.ndarray:
    img = np.empty((store.height, store.width, 3), np.uint8)
    img[:] = BACKGROUND_BGR
    row = store.frames[index]
    # big blobs first so nearer/smaller people are not hidden behind them
    ordered = sorted(row, key=lambda r: -sum(cv2.contourArea(p.astype(np.int32)) for p in r[1]))
    for oid, polys in ordered:
        r, g, b = color_for(oid - 1)
        cv2.fillPoly(img, [p.astype(np.int32) for p in polys], (b, g, r), lineType=cv2.LINE_AA)
    return img


def render_video(store: TrackStore, dst: Path, progress=None) -> None:
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), store.fps, (store.width, store.height))
    if not writer.isOpened():
        raise RuntimeError("could not open video writer")
    n = len(store.frames)
    try:
        for i in range(n):
            writer.write(draw_blobs(store, i))
            if progress and i % 10 == 0:
                progress((i + 1) / n, f"rendering frame {i + 1}/{n}")
    finally:
        writer.release()
