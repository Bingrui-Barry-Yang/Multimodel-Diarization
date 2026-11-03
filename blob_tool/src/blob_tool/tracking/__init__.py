"""Tracking engines. Heavy optional engines are imported lazily."""

from __future__ import annotations

import importlib.util

from .base import FrameResult, TrackedObject, Tracker, TrackParams

ENGINES = {
    "rfdetr": {
        "label": "RF-DETR-Seg + BoT-SORT (Apache-2.0, CoreML-exportable)",
        "requires": "rfdetr",
    },
    "yolo": {
        "label": "YOLO26-seg + BoT-SORT (fast, AGPL-3.0)",
        "requires": None,
    },
    "edgetam": {
        "label": "EdgeTAM masks seeded by YOLO tracks (SAM 2-quality, on-device class)",
        "requires": "transformers",
    },
    "sam3": {
        "label": "SAM 3 text-prompt video tracking (best quality, slow)",
        "requires": "transformers",
    },
}


def available_engines() -> list[dict]:
    out = []
    for key, spec in ENGINES.items():
        req = spec["requires"]
        ok = req is None or importlib.util.find_spec(req) is not None
        out.append({"id": key, "label": spec["label"], "available": ok,
                    "hint": None if ok else "run `make install-sam` to enable"})
    return out


def get_tracker(name: str) -> Tracker:
    if name == "rfdetr":
        from .rfdetr import RfDetrTracker

        return RfDetrTracker()
    if name == "yolo":
        from .yolo import YoloTracker

        return YoloTracker()
    if name == "edgetam":
        from .edgetam import EdgeTamTracker

        return EdgeTamTracker()
    if name == "sam3":
        from .sam3 import Sam3Tracker

        return Sam3Tracker()
    raise ValueError(f"unknown engine {name!r}")


__all__ = ["FrameResult", "TrackedObject", "Tracker", "TrackParams", "available_engines", "get_tracker"]
