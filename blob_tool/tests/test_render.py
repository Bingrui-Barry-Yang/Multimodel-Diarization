import json

import cv2
import numpy as np

from blob_tool.render import (BACKGROUND_BGR, BlobParams, TrackStore, color_for, draw_blobs, draw_frame,
                              mask_to_polygons)


def disk(h, w, cx, cy, r):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def test_polygons_follow_mask_in_every_style():
    m = disk(480, 640, 300, 200, 60)
    for style in ("silhouette", "hull", "ellipse"):
        polys = mask_to_polygons(m, BlobParams(style=style))
        assert len(polys) == 1
        x, y, w, h = cv2.boundingRect(polys[0].astype(np.int32))
        assert abs(x + w / 2 - 300) < 8 and abs(y + h / 2 - 200) < 8
        assert 100 < w < 140


def test_empty_and_speck_masks_give_nothing():
    assert mask_to_polygons(np.zeros((100, 100), bool), BlobParams()) == []
    assert mask_to_polygons(disk(1000, 1000, 50, 50, 2), BlobParams()) == []


def test_blob_frame_has_no_source_pixels():
    store = TrackStore(320, 240, 25)
    store.add_frame([(7, "person", 0.9, mask_to_polygons(disk(240, 320, 160, 120, 40), BlobParams()))])
    store.finalize(0)
    img = draw_blobs(store, 0).reshape(-1, 3).astype(float)
    bg = np.array(BACKGROUND_BGR, float)
    blob = np.array(color_for(0)[::-1], float)
    # every pixel is background, blob colour, or an anti-aliased mix of the two
    t = ((img - bg) @ (blob - bg)) / ((blob - bg) @ (blob - bg))
    residual = img - (bg + np.outer(np.clip(t, 0, 1), blob - bg))
    assert np.abs(residual).max() <= 2.0


def test_stitching_merges_a_lost_and_refound_track():
    store = TrackStore(640, 480, 10)
    blob = lambda x: mask_to_polygons(disk(480, 640, x, 240, 40), BlobParams())  # noqa: E731
    for i in range(20):
        store.add_frame([(1, "person", 0.9, blob(200 + i))])
    for _ in range(5):  # occluded for 0.5 s
        store.add_frame([])
    for i in range(20):  # re-detected nearby under a new tracker id
        store.add_frame([(9, "person", 0.9, blob(225 + i))])
    store.add_frame([(4, "person", 0.9, blob(600))])  # one-frame flicker elsewhere

    store.finalize(min_track_seconds=0.5, stitch_gap_seconds=2.0)
    data = store.to_json()
    assert [o["id"] for o in data["objects"]] == [1]
    assert data["objects"][0]["frames_visible"] == 40
    assert data["frames"][-2][0][0] == 1
    assert data["frames"][-1] == []
    json.dumps(data)


def test_far_apart_tracks_are_not_stitched():
    store = TrackStore(640, 480, 10)
    for _ in range(10):
        store.add_frame([(1, "person", 0.9, mask_to_polygons(disk(480, 640, 100, 240, 40), BlobParams()))])
    for _ in range(10):
        store.add_frame([(2, "person", 0.9, mask_to_polygons(disk(480, 640, 540, 240, 40), BlobParams()))])
    store.finalize(min_track_seconds=0.5, stitch_gap_seconds=2.0)
    assert len(store.meta) == 2


def test_pose_frames_have_no_source_pixels():
    """Pose-only and blob+pose frames contain only the background and colours derived from the blob colour."""
    store = TrackStore(320, 240, 25)
    store.add_frame([(7, "person", 0.9, mask_to_polygons(disk(240, 320, 160, 120, 40), BlobParams()))])
    store.finalize(0)
    k = np.column_stack([np.linspace(140, 180, 17), np.linspace(100, 140, 17), np.ones(17)])  # inside the blob
    k[0, 2] = 0.1  # below min_score: not drawn
    bg = np.array(BACKGROUND_BGR, float)
    blob = np.array(color_for(0)[::-1], float)
    tint = np.floor(blob + (255 - blob) * 0.7)  # draw_skeletons truncates to int
    for on_blobs, colours in ((False, [blob]), (True, [blob, tint])):
        img = draw_frame(store, 0, [[(1, k)]], blobs=on_blobs).reshape(-1, 3).astype(float)
        drawn = (np.abs(img - bg).sum(1) > 6)
        assert drawn.any()
        # every pixel lies on a segment between known colours (anti-aliasing mixes neighbours)
        anchors = [bg, *colours]
        best = np.full(len(img), np.inf)
        for i, a in enumerate(anchors):
            for b in anchors[i + 1:]:
                t = np.clip(((img - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
                best = np.minimum(best, np.abs(img - (a + np.outer(t, b - a))).max(1))
        assert best.max() <= 2.0
