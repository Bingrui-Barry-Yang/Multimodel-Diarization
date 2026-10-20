"""SAM 3 promptable concept segmentation + tracking ("person", "chair", …).

Runs SAM 3 in streaming mode (one frame at a time) so long videos do not have to
fit in memory. Streaming skips SAM 3's look-ahead "hotstart" clean-up, so the
short-track filter and stitching in render.TrackStore matter more here.

Requires `make install-sam` and access to the gated checkpoint: accept the
license at https://huggingface.co/facebook/sam3 and run `uv run hf auth login`.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .base import FrameResult, ProgressFn, TrackedObject, TrackParams, pick_device

CHECKPOINT = "facebook/sam3"


@lru_cache(maxsize=1)
def _load(device: str):
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    try:
        model = Sam3VideoModel.from_pretrained(CHECKPOINT, dtype=torch.float32).to(device).eval()
        processor = Sam3VideoProcessor.from_pretrained(CHECKPOINT)
    except OSError as e:
        raise RuntimeError(
            "could not load facebook/sam3 — accept the license on huggingface.co/facebook/sam3 "
            "and run `uv run hf auth login`"
        ) from e
    return model, processor


class Sam3Tracker:
    name = "sam3"

    def run(self, video: Path, params: TrackParams, total_frames: int, progress: ProgressFn) -> Iterator[FrameResult]:
        import torch

        device = pick_device()
        model, processor = _load(device)
        prompts = [p.strip() for p in (params.text_prompt or "person").split(",") if p.strip()]

        cap = cv2.VideoCapture(str(video))
        try:
            with torch.inference_mode():
                session = processor.init_video_session(
                    inference_device=device, processing_device="cpu", video_storage_device="cpu"
                )
                processor.add_text_prompt(inference_session=session, text=prompts)
                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = processor(images=rgb, device=device, return_tensors="pt")
                    out = model(inference_session=session, frame=inputs.pixel_values[0].to(device), reverse=False)
                    pp = processor.postprocess_outputs(session, out, original_sizes=inputs.original_sizes)
                    label_of = {oid: p for p, ids in pp.get("prompt_to_obj_ids", {}).items() for oid in ids}
                    objects = []
                    for oid, mask, score in zip(
                        pp["object_ids"].tolist(), pp["masks"].cpu().numpy(), pp["scores"].tolist()
                    ):
                        m = np.asarray(mask).astype(bool)
                        if m.ndim == 3:
                            m = m[0]
                        if m.any():
                            objects.append(TrackedObject(int(oid), label_of.get(oid, prompts[0]), float(score), m))
                    i += 1
                    progress(i / max(total_frames, 1), f"SAM 3 frame {i}/{total_frames}")
                    yield FrameResult(i - 1, objects)
        finally:
            cap.release()
