"""Blurred outputs: what survives, what is gone for good."""

import numpy as np
import pytest

from blob_tool.blur import BlurParams, blur_frame, head_boxes
from blob_tool.render import BlobParams, TrackStore, color_for, mask_to_polygons

W, H = 320, 160  # detail=160 → exactly 2×2 pixel cells, so cell means are easy to control


def person(x=160, top=30, w=60, h=110):
    m = np.zeros((H, W), bool)
    m[top:top + h, x - w // 2:x + w // 2] = True
    return [(1, mask_to_polygons(m, BlobParams(smooth=0)))]


def pose_for(x=160, head_y=45):
    k = np.zeros((17, 3))
    k[:, 2] = 1.0
    k[:5, :2] = [[x, head_y], [x - 5, head_y - 3], [x + 5, head_y - 3], [x - 9, head_y], [x + 9, head_y]]
    k[5:, :2] = [x, head_y + 40]
    k[5, 0], k[6, 0] = x - 20, x + 20
    return [(1, k)]


def smooth_frame():
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 0.5, yy * 1.0, 100 + 0 * xx], -1)
    return (img // 2 * 2 + 40).astype(np.uint8)  # 40–198: room for ±30 without clipping


def test_fine_detail_cannot_be_recovered():
    """Two originals that differ only in fine detail (a ±30 checkerboard, everywhere including the
    face) give byte-identical strong outputs: that information is not in the output at all."""
    a = smooth_frame()
    checker = np.where((np.indices((H, W)).sum(0) % 2) == 0, 30, -30)[..., None]
    b = (a.astype(int) + checker).astype(np.uint8)
    assert np.abs(a.astype(int) - b).mean() > 25
    p = BlurParams(detail=160)
    outs = [blur_frame(x, person(), pose_for(), p, np.random.default_rng(0)) for x in (a, b)]
    np.testing.assert_array_equal(outs[0], outs[1])


def test_head_becomes_a_few_noisy_blocks():
    p = BlurParams(head_blocks=3, outlines=False)
    frame = np.random.default_rng(1).integers(0, 255, (H, W, 3), np.uint8)
    (x0, y0, x1, y1), = head_boxes(person(), pose_for(), 0.3, W, H)
    assert x0 < 160 < x1 and y0 < 45 < y1
    outs = [blur_frame(frame, person(), pose_for(), p, np.random.default_rng(s)) for s in (0, 1)]
    head = outs[0][y0:y1, x0:x1].reshape(-1, 3)
    assert len(np.unique(head, axis=0)) <= 9
    assert not np.array_equal(outs[0][y0:y1, x0:x1], outs[1][y0:y1, x0:x1])  # fresh noise per seed
    np.testing.assert_array_equal(outs[0][:y0], outs[1][:y0])  # noise only inside heads


def test_without_keypoints_the_top_of_the_blob_is_covered():
    (x0, y0, x1, y1), = head_boxes(person(top=30, w=60, h=110), None, 0.3, W, H)
    assert (y0, x0, x1) == (30, 130, 190) and y1 >= 30 + 0.25 * 110
    low = pose_for()
    low[0][1][:5, 2] = 0.1  # head keypoints below min_score count as missing
    assert head_boxes(person(), low, 0.3, W, H) == [(x0, y0, x1, y1)]


def test_keypoints_only_add_coverage():
    top, = head_boxes(person(), None, 0.3, W, H)
    bent, = head_boxes(person(), pose_for(head_y=100), 0.3, W, H)  # head well below the blob top
    assert bent[0] <= top[0] and bent[1] <= top[1] and bent[2] >= top[2] and bent[3] > 100


def test_outlines_use_the_blob_colour():
    out = blur_frame(np.full((H, W, 3), 128, np.uint8), person(), None, BlurParams(), np.random.default_rng(0))
    r, g, b = color_for(0)
    assert (np.abs(out.astype(int) - (b, g, r)).sum(-1) <= 3).sum() > 50


def test_face_method_only_touches_heads():
    frame = np.random.default_rng(2).integers(0, 255, (H, W, 3), np.uint8)
    out = blur_frame(frame, person(), pose_for(), BlurParams(method="face"), np.random.default_rng(0))
    (x0, y0, x1, y1), = head_boxes(person(), pose_for(), 0.3, W, H)
    inside = np.zeros((H, W), bool)
    inside[y0:y1, x0:x1] = True
    np.testing.assert_array_equal(out[~inside], frame[~inside])
    assert out[inside].std() < frame[inside].std() * 0.5


def test_bad_params_are_rejected():
    for bad in ({"method": "pixelate"}, {"detail": 8}, {"head_blocks": 0}):
        with pytest.raises(ValueError):
            BlurParams(**bad)


def test_tracks_round_trip(tmp_path):
    store = TrackStore(W, H, 10)
    store.add_frame([(4, "person", 0.9, person()[0][1])])
    store.finalize(0)
    store.write_json(tmp_path / "tracks.json")
    back = TrackStore.read_json(tmp_path / "tracks.json")
    assert (back.width, back.height, back.fps) == (W, H, 10) and list(back.meta) == [1]
    (oid, polys), = back.frames[0]
    assert oid == 1 and np.abs(polys[0] - store.frames[0][0][1][0]).max() <= 1
