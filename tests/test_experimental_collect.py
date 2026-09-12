from pathlib import Path

import numpy as np
import pytest

from s20_pipeline.camera import CameraFrame, FisheyeCalibration
from s20_pipeline.cpu_visibility import CpuVisibility
from s20_pipeline.experimental_collect import (
    _bucket_photo_targets,
    _insert_observations,
    _insert_top4,
    _photo_targets,
    build_cell_index,
    select_keyframes,
    select_keyframes_for_scan,
)

LIBRARY = Path(__file__).parents[1] / "build/libs20_visibility.dylib"
native_required = pytest.mark.skipif(not LIBRARY.is_file(), reason="CPU visibility is not built")


def frame(name, center):
    calibration = FisheyeCalibration(
        name, 64, 48, (0.0,) * 6, 18.0, 0.0, 18.0, 32.0, 24.0, np.eye(4), 100.0
    )
    return CameraFrame(
        name, Path("unused"), 0.0, np.asarray(center, dtype="float64"), np.eye(3), calibration
    )


def test_cell_index_splits_opposing_normals_and_maps_every_source_point():
    xyz = np.array([[0.01, 0.01, 1], [0.02, 0.02, 1], [0.03, 0.03, 1]], dtype="float32")
    normals = np.array([[1, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype="float32")
    cells = build_cell_index(xyz, normals, 0.1, True)
    assert len(cells.representatives) == 2
    assert cells.inverse[0] == cells.inverse[1]
    assert cells.inverse[0] != cells.inverse[2]
    np.testing.assert_array_equal(np.sort(cells.counts), [1, 2])


def test_photo_targets_expand_cell_shortlists_to_original_order():
    xyz = np.array([[0, 0, 1], [0.01, 0, 1], [1, 0, 1]], dtype="float32")
    normals = np.ones_like(xyz)
    cells = build_cell_index(xyz, normals, 0.1, False)
    photos = np.array([[2, 4], [3, -1]], dtype="int16")
    np.testing.assert_array_equal(_photo_targets(cells, photos, 2), [0, 1])
    np.testing.assert_array_equal(_photo_targets(cells, photos, 3), [2])


def test_bucketed_photo_targets_preserve_original_order():
    xyz = np.array([[0, 0, 1], [0.01, 0, 1], [1, 0, 1]], dtype="float32")
    cells = build_cell_index(xyz, np.ones_like(xyz), 0.1, False)
    photos = np.array([[2, 4], [3, -1]], dtype="int16")
    buckets = _bucket_photo_targets(cells, photos, 5, 1)
    np.testing.assert_array_equal(buckets[2], [0, 1])
    np.testing.assert_array_equal(buckets[3], [2])
    assert sum(map(len, buckets)) == len(xyz)


def test_top4_insertion_is_strict_and_chronological_on_ties():
    scores = np.zeros((2, 4), dtype="float32")
    photos = np.full((2, 4), -1, dtype="int16")
    positions = np.array([0, 1])
    assert _insert_top4(scores, photos, positions, np.array([1, 2], dtype="float32"), 3) == 2
    assert _insert_top4(scores, photos, positions, np.array([1, 1], dtype="float32"), 4) == 2
    for photo in (5, 6):
        _insert_top4(scores, photos, positions, np.array([1, 1], dtype="float32"), photo)
    assert _insert_top4(scores, photos, positions, np.array([1, 1], dtype="float32"), 7) == 0
    assert photos[0].tolist() == [3, 4, 5, 6]


def test_single_slot_insertion_keeps_strict_best_and_leaves_other_slots_empty():
    observations = np.zeros((3, 4, 8), dtype="float32")
    ids = np.array([0, 2], dtype="uint32")
    uv = np.array([10, 20], dtype="float32")
    score = np.array([2, 1], dtype="float32")
    assert _insert_observations(None, observations, ids, uv, uv + 1, score, 4, 1) == 2
    assert _insert_observations(None, observations, ids, uv, uv, score, 5, 1) == 0
    assert _insert_observations(
        None, observations, ids, uv + 2, uv + 3, score + 1, 6, 1
    ) == 2
    np.testing.assert_array_equal(observations[ids, 0, 6], [6, 6])
    np.testing.assert_array_equal(observations[:, 1:, :], 0)


@native_required
def test_keyframes_are_balanced_and_deterministic():
    rng = np.random.default_rng(22)
    xyz = rng.uniform([-1, -1, 0.5], [1, 1, 3], (2000, 3)).astype("float32")
    normals = -xyz / np.linalg.norm(xyz, axis=1)[:, None]
    frames = [
        frame(name, [offset, 0, 0])
        for offset in (-0.4, 0.0, 0.4)
        for name in ("left", "right")
    ]
    native = CpuVisibility(xyz, normals.astype("float32"), LIBRARY)
    first, meta = select_keyframes(xyz, normals, frames, native, 4)
    second, _ = select_keyframes(xyz, normals, frames, native, 4)
    np.testing.assert_array_equal(first, second)
    assert meta["camera_counts"] == {"left": 2, "right": 2}
    assert len(first) == 4


@native_required
def test_long_scan_keyframe_budget_scales_in_bounded_windows():
    rng = np.random.default_rng(23)
    xyz = rng.uniform([-1, -1, 0.5], [1, 1, 3], (2000, 3)).astype("float32")
    normals = (-xyz / np.linalg.norm(xyz, axis=1)[:, None]).astype("float32")
    frames = [frame("left" if i % 2 == 0 else "right", [i * 0.01, 0, 0]) for i in range(124)]
    native = CpuVisibility(xyz, normals, LIBRARY)
    selected, meta = select_keyframes_for_scan(xyz, normals, frames, native)
    assert len(selected) == 40
    assert len(meta["windows"]) == 2
    assert meta["selection_window_photos"] == 62
    assert meta["selection_sample_points"] == 2000
    assert np.count_nonzero(selected < 62) == 20
    assert np.count_nonzero(selected >= 62) == 20
    short, short_meta = select_keyframes_for_scan(xyz, normals, frames[:62], native)
    assert len(short) == 20
    assert short_meta["photo_percent"] == 30
    reviewed, _ = select_keyframes(xyz, normals, frames[:62], native, 20)
    np.testing.assert_array_equal(short, reviewed)
    fewer, fewer_meta = select_keyframes_for_scan(xyz, normals, frames, native, 20)
    assert len(fewer) == 24
    assert fewer_meta["photo_percent"] == 20
    with pytest.raises(ValueError, match="1–100"):
        select_keyframes_for_scan(xyz, normals, frames, native, 0)
