"""EdgeTAM (on-device SAM 2) masks, seeded and identified by YOLO + BoT-SORT.

YOLO decides *who* is in the scene and keeps identities; EdgeTAM decides *which
pixels* belong to them, using its memory attention to keep masks temporally
coherent. The video is processed in chunks: at the first frame of every chunk
EdgeTAM is prompted with the YOLO boxes of all current tracks, then propagates
through the chunk. Objects that appear mid-chunk fall back to the YOLO mask
until the next chunk re-seeds them.

Requires `make install-sam` (transformers). Weights: yonigozlan/EdgeTAM-hf.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .base import FrameResult, ProgressFn, TrackedObject, TrackParams, pick_device
from .yolo import YoloTracker

CHECKPOINT = "yonigozlan/EdgeTAM-hf"


@lru_cache(maxsize=1)
def _load(device: str):
    import torch
    from transformers import EdgeTamVideoModel, Sam2VideoProcessor

    model = EdgeTamVideoModel.from_pretrained(CHECKPOINT, dtype=torch.float32).to(device).eval()
    processor = Sam2VideoProcessor.from_pretrained(CHECKPOINT)
    return model, processor


def _boxes(mask: np.ndarray) -> list[float]:
    x, y, w, h = cv2.boundingRect(mask.view(np.uint8))
    return [float(x), float(y), float(x + w), float(y + h)]


class EdgeTamTracker:
    name = "edgetam"

    def run(self, video: Path, params: TrackParams, total_frames: int, progress: ProgressFn) -> Iterator[FrameResult]:
        import torch

        device = pick_device()
        model, processor = _load(device)
        cap = cv2.VideoCapture(str(video))
        yolo_stream = YoloTracker().run(video, params, total_frames, lambda f, m: None)

        chunk: list[tuple[np.ndarray, FrameResult]] = []
        done = 0

        def flush() -> Iterator[FrameResult]:
            nonlocal done
            frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f, _ in chunk]
            seeds = [o for o in chunk[0][1].objects if o.mask.any()]
            refined: dict[int, dict[int, np.ndarray]] = {}
            if seeds:
                h, w = frames[0].shape[:2]
                with torch.inference_mode():
                    session = processor.init_video_session(video=frames, inference_device=device)
                    processor.add_inputs_to_inference_session(
                        inference_session=session,
                        frame_idx=0,
                        obj_ids=[o.track_id for o in seeds],
                        input_boxes=[[_boxes(o.mask) for o in seeds]],
                    )
                    model(inference_session=session, frame_idx=0)
                    for out in model.propagate_in_video_iterator(session):
                        masks = processor.post_process_masks(
                            [out.pred_masks], original_sizes=[[h, w]], binarize=True
                        )[0]
                        refined[out.frame_idx] = {
                            oid: masks[i, 0].cpu().numpy().astype(bool) for i, oid in enumerate(session.obj_ids)
                        }
            labels = {o.track_id: o.label for _, fr in chunk for o in fr.objects}
            for k, (_, fr) in enumerate(chunk):
                yolo_by_id = {o.track_id: o for o in fr.objects}
                objects = []
                for oid, m in refined.get(k, {}).items():
                    if m.any():
                        score = yolo_by_id[oid].score if oid in yolo_by_id else 0.5
                        objects.append(TrackedObject(oid, labels.get(oid, "object"), score, m))
                seen = {o.track_id for o in objects}
                objects += [o for o in fr.objects if o.track_id not in seen]  # not seeded yet → YOLO mask
                done += 1
                progress(done / max(total_frames, 1), f"EdgeTAM frame {done}/{total_frames}")
                yield FrameResult(fr.index, objects)
            chunk.clear()

        try:
            for fr in yolo_stream:
                ok, frame = cap.read()
                if not ok:
                    break
                chunk.append((frame, fr))
                if len(chunk) >= params.chunk_frames:
                    yield from flush()
            if chunk:
                yield from flush()
        finally:
            cap.release()
