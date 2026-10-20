"""End-to-end pipeline on a synthetic video with a fake tracker (no model download)."""

import json
import subprocess

import cv2
import numpy as np
import pytest

from blob_tool import media, pipeline
from blob_tool.tracking.base import FrameResult, TrackedObject

W, H, FPS, N = 320, 240, 10, 30


@pytest.fixture
def synthetic_video(tmp_path):
    path = tmp_path / "in.mp4"
    frames = tmp_path / "raw.mp4"
    vw = cv2.VideoWriter(str(frames), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(N):
        img = np.full((H, W, 3), 200, np.uint8)
        cv2.circle(img, (40 + i * 6, 120), 30, (0, 0, 255), -1)
        vw.write(img)
    vw.release()
    # add a 440 Hz tone as the "voice"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(frames), "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={N / FPS}", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)
    return path


class RedBlobTracker:
    """Segments the red disk by colour, always as track 1."""

    name = "fake"

    def run(self, video, params, total_frames, progress):
        cap = cv2.VideoCapture(str(video))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            mask = (img[:, :, 2] > 150) & (img[:, :, 1] < 100)
            yield FrameResult(i, [TrackedObject(1, "person", 0.9, mask)])
            i += 1
            progress(i / total_frames, "")
        cap.release()


def test_pipeline_end_to_end(tmp_path, synthetic_video, monkeypatch):
    monkeypatch.setattr(pipeline, "get_tracker", lambda name: RedBlobTracker())
    stages = []
    out = tmp_path / "job"
    result = pipeline.run(synthetic_video, out, pipeline.PipelineParams(),
                          lambda s, f, m: stages.append(s))

    assert result["objects"] == 1 and result["has_audio"]
    assert stages[0] == "prepare" and stages[-1] == "mux"
    for name in ("original.mp4", "original.wav", "altered.wav", "processed.mp4", "tracks.json"):
        assert (out / name).exists(), name

    info = media.probe(out / "processed.mp4")
    assert (info.width, info.height) == (W, H) and info.has_audio

    tracks = json.loads((out / "tracks.json").read_text())
    assert tracks["frame_count"] == N
    xs = [np.mean(row[0][1][0::2]) for row in tracks["frames"] if row]
    assert xs[-1] > xs[0]  # blob moves right like the disk

    # processed video contains no trace of the grey (200,200,200) background
    cap = cv2.VideoCapture(str(out / "processed.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.mean() < 120
