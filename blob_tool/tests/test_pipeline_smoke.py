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


def fake_poses(video, store, params, progress):
    """A stick figure centred on every blob, all keypoints confident."""
    rows = []
    for row in store.frames:
        poses = []
        for oid, polys in row:
            c = np.vstack(polys).mean(0)
            k = np.column_stack([c + np.linspace(-10, 10, 17)[:, None], np.ones(17)])
            poses.append((oid, k))
        rows.append(poses)
    return rows


def test_pipeline_end_to_end(tmp_path, synthetic_video, monkeypatch):
    monkeypatch.setattr(pipeline, "get_tracker", lambda name: RedBlobTracker())
    monkeypatch.setattr(pipeline, "estimate_poses", fake_poses)
    stages = []
    out = tmp_path / "job"
    params = pipeline.PipelineParams()
    result = pipeline.run(synthetic_video, out, params, lambda s, f, m: stages.append(s))

    assert result["objects"] == 1 and result["has_audio"]
    assert stages[0] == "prepare" and stages[-1] == "mux"
    for name in ("original.mp4", "original.wav", "altered.wav", "processed.mp4", "tracks.json",
                 "poses.json", "pose.mp4", "blobpose.mp4"):
        assert (out / name).exists(), name
    assert not list(out.glob("*_raw.mp4"))
    assert media.probe(out / "pose.mp4").has_audio and result["poses"]

    poses = json.loads((out / "poses.json").read_text())
    assert len(poses["frames"]) == N and len(poses["keypoints"]) == 17
    assert all(len(p) == 1 + 17 * 3 and p[0] == 1 for row in poses["frames"] for p in row)

    # the seed was drawn for this run and stays out of every served text file
    seed = str(params.seed)
    assert params.seed is not None
    for name in ("tracks.json", "poses.json"):
        assert seed not in (out / name).read_text()
    assert seed not in json.dumps(result)

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


def test_blur_after_processing(tmp_path, synthetic_video, monkeypatch):
    from blob_tool.blur import BlurParams

    monkeypatch.setattr(pipeline, "get_tracker", lambda name: RedBlobTracker())
    monkeypatch.setattr(pipeline, "estimate_poses", fake_poses)
    out = tmp_path / "job"
    params = pipeline.PipelineParams()
    pipeline.run(synthetic_video, out, params, lambda *a: None)
    for method, name in (("strong", "blurred.mp4"), ("face", "faceblur.mp4")):
        assert pipeline.blur(out, BlurParams(method=method), params.seed, 0.3, lambda *a: None) == name
        info = media.probe(out / name)
        assert (info.width, info.height) == (W, H) and info.has_audio and info.frames >= N - 1
    assert not list(out.glob("*_raw.mp4"))
    (out / "original.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.blur(out, BlurParams(), params.seed, 0.3, lambda *a: None)


def test_pose_can_be_turned_off(tmp_path, synthetic_video, monkeypatch):
    monkeypatch.setattr(pipeline, "get_tracker", lambda name: RedBlobTracker())
    monkeypatch.setattr(pipeline, "estimate_poses", lambda *a: pytest.fail("pose ran while disabled"))
    out = tmp_path / "job"
    result = pipeline.run(synthetic_video, out, pipeline.PipelineParams.from_dict({"pose": {"enabled": False}}),
                          lambda *a: None)
    assert not result["poses"] and (out / "processed.mp4").exists()
    assert not (out / "pose.mp4").exists() and not (out / "poses.json").exists()


def test_seed_parsing():
    assert pipeline.PipelineParams.from_dict({}).seed is None
    assert pipeline.PipelineParams.from_dict({"seed": ""}).seed is None
    assert pipeline.PipelineParams.from_dict({"seed": "42"}).seed == 42
    with pytest.raises(ValueError):
        pipeline.PipelineParams.from_dict({"seed": -1})
    with pytest.raises(ValueError):
        pipeline.PipelineParams.from_dict({"seed": "abc"})
