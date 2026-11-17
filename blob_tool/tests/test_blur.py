"""Light blur: what survives, what is gone for good, and how big the face blur is."""

import numpy as np
import pytest

from blob_tool.blur import BLOB_ALPHA, BlurParams, blur_base, blur_frame, face_ellipses, layer_frame
from blob_tool.render import BlobParams, TrackStore, color_for, mask_to_polygons

W, H = 320, 160


def person(x=160, top=30, w=60, h=110):
    m = np.zeros((H, W), bool)
    m[top:top + h, x - w // 2:x + w // 2] = True
    return [(1, mask_to_polygons(m, BlobParams(smooth=0)))]


def pose_for(x=160, head_y=45, ear=9):
    k = np.zeros((17, 3))
    k[:, 2] = 1.0
    k[:5, :2] = [[x, head_y], [x - 5, head_y - 3], [x + 5, head_y - 3], [x - ear, head_y], [x + ear, head_y]]
    k[5:, :2] = [x, head_y + 40]
    k[5, 0], k[6, 0] = x - 20, x + 20
    return [(1, k)]


def smooth_frame():
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 0.5, yy * 1.0, 100 + 0 * xx], -1)
    return (img // 2 * 2 + 40).astype(np.uint8)  # 40–198: room for ±30 without clipping


def test_fine_detail_cannot_be_recovered():
    """Two originals that differ only in fine detail (a ±30 checkerboard everywhere, faces included)
    give byte-identical blurred outputs: that information is not in the output at all."""
    a = smooth_frame()
    checker = np.where((np.indices((H, W)).sum(0) % 2) == 0, 30, -30)[..., None]
    b = (a.astype(int) + checker).astype(np.uint8)
    p = BlurParams(enabled=True, strength=2)
    np.testing.assert_array_equal(blur_frame(a, pose_for(), p), blur_frame(b, pose_for(), p))


def test_light_blur_keeps_edges():
    frame = np.zeros((H, W, 3), np.uint8)
    frame[:, W // 2:] = 200  # a hard vertical edge
    out = blur_base(frame, 2.5)
    row = out[H // 2, :, 0].astype(int)
    assert row[: W // 2 - 4].max() == 0 and row[W // 2 + 4:].min() == 200  # the edge moves by a few px at most


def test_face_blur_is_head_sized_and_local():
    (cx, cy, ax, ay), = face_ellipses(pose_for(ear=9), 0.3)
    assert abs(cx - 160) < 1 and 40 < cy < 46
    assert 1.2 * 18 <= 2 * ax <= 1.7 * 18  # ear-to-ear 18 px → about 1.3× head width, not a slab
    frame = np.random.default_rng(1).integers(0, 255, (H, W, 3), np.uint8)
    on = blur_frame(frame, pose_for(), BlurParams(faces=True))
    off = blur_frame(frame, pose_for(), BlurParams(faces=False))
    changed = np.abs(on.astype(int) - off).sum(-1) > 0
    ys, xs = np.nonzero(changed)
    assert changed[round(cy), round(cx)]
    assert xs.min() >= cx - 3 * ax and xs.max() <= cx + 3 * ax and ys.min() >= cy - 3 * ay and ys.max() <= cy + 3 * ay
    assert changed.mean() < 0.05


def test_no_head_keypoints_no_face_blur():
    low = pose_for()
    low[0][1][:5, 2] = 0.1
    assert face_ellipses(low, 0.3) == [] and face_ellipses(None, 0.3) == []
    frame = np.random.default_rng(2).integers(0, 255, (H, W, 3), np.uint8)
    np.testing.assert_array_equal(blur_frame(frame, low, BlurParams()), blur_frame(frame, low, BlurParams(faces=False)))


def test_blobs_are_translucent_with_a_solid_outline():
    base = np.full((H, W, 3), 100, np.uint8)
    out = layer_frame(base, person(), None, blobs=True, pose=False)
    r, g, b = color_for(0)
    bgr = np.array([b, g, r], float)
    inside = out[90, 160].astype(float)
    np.testing.assert_allclose(inside, BLOB_ALPHA * bgr + (1 - BLOB_ALPHA) * 100, atol=1.5)
    assert (np.abs(out.astype(int) - bgr).sum(-1) <= 3).sum() > 50  # outline in the blob colour
    np.testing.assert_array_equal(out[5, 5], base[5, 5])
    np.testing.assert_array_equal(layer_frame(base, person(), None, False, False), base)


def test_pose_layer_draws_skeletons():
    base = np.full((H, W, 3), 100, np.uint8)
    out = layer_frame(base, person(), pose_for(), blobs=False, pose=True)
    r, g, b = color_for(0)
    assert (np.abs(out.astype(int) - (b, g, r)).sum(-1) <= 3).sum() > 20


def test_bad_params_are_rejected():
    for bad in (1.0, 8):
        with pytest.raises(ValueError):
            BlurParams(strength=bad)


def test_tracks_round_trip(tmp_path):
    store = TrackStore(W, H, 10)
    store.add_frame([(4, "person", 0.9, person()[0][1])])
    store.finalize(0)
    store.write_json(tmp_path / "tracks.json")
    back = TrackStore.read_json(tmp_path / "tracks.json")
    assert (back.width, back.height, back.fps) == (W, H, 10) and list(back.meta) == [1]
    (oid, polys), = back.frames[0]
    assert oid == 1 and np.abs(polys[0] - store.frames[0][0][1][0]).max() <= 1
