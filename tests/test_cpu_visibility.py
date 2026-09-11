from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from s20_pipeline.camera import CameraFrame, FisheyeCalibration
from s20_pipeline.collect import visibility_decisions
from s20_pipeline.cpu_visibility import CpuVisibility

REPOSITORY = Path(__file__).parents[1]
LIBRARY = REPOSITORY / "build/libs20_visibility.dylib"
native_required = pytest.mark.skipif(not LIBRARY.is_file(), reason="CPU visibility is not built")


def camera():
    calibration = FisheyeCalibration(
        "left", 64, 48, (0.0,) * 6, 18.0, 0.0, 18.0, 32.0, 24.0, np.eye(4), 90.0
    )
    return CameraFrame(
        "left",
        Path("unused"),
        0.0,
        np.array([0.123456789, -0.987654321, 0.333333333]),
        np.eye(3),
        calibration,
    )


@native_required
def test_native_visibility_matches_mixed_numpy_bits_and_mask():
    rng = np.random.default_rng(4920)
    point_count = 20_000
    count = 15_000
    xyz = np.ascontiguousarray(rng.normal(size=(point_count, 3)), dtype="float32")
    normals = np.ascontiguousarray(rng.normal(size=(point_count, 3)), dtype="float32")
    ids = rng.integers(0, point_count, count, dtype="uint32")
    blockers = rng.integers(0, point_count, count, dtype="uint32")
    distance = rng.uniform(0.11, 10, count).astype("float32")
    u = rng.uniform(0, 62, count).astype("float32")
    v = rng.uniform(0, 46, count).astype("float32")
    exact_quantized = np.maximum(
        0, np.rint((distance + rng.normal(0, 0.03, count)) * 1e6)
    ).astype("uint64")
    exact_keys = exact_quantized * np.uint64(point_count) + ids
    blocker_keys = (
        rng.integers(0, 1000, count, dtype="uint64") * np.uint64(point_count) + blockers
    )
    mask = np.ascontiguousarray(rng.integers(0, 8, (48, 64), dtype="uint8") == 0, dtype="uint8")

    expected = visibility_decisions(
        xyz, normals, ids, distance, blockers, exact_keys, point_count, camera(), "mixed"
    )
    result = CpuVisibility(xyz, normals, LIBRARY).decide(
        ids, u, v, distance, blocker_keys, exact_keys, camera(), mask
    )
    expected_flags = np.zeros(count, dtype="uint8")
    expected_flags[expected[1]] |= 1
    expected_flags[expected[2]] |= 2
    expected_flags[expected[0]] |= 4
    x = u.astype("int32")
    y = v.astype("int32")
    usable = (
        expected[0]
        & (mask[y, x] == 0)
        & (mask[y + 1, x] == 0)
        & (mask[y, x + 1] == 0)
        & (mask[y + 1, x + 1] == 0)
    )
    expected_flags[usable] |= 8
    np.testing.assert_array_equal(result.flags, expected_flags)
    np.testing.assert_array_equal(result.incidence.view("uint32"), expected[3].view("uint32"))


@native_required
def test_native_visibility_accepts_empty_chunks():
    xyz = np.ones((1, 3), dtype="float32")
    mask = np.zeros((2, 2), dtype="uint8")
    empty32 = np.empty(0, dtype="float32")
    empty64 = np.empty(0, dtype="uint64")
    result = CpuVisibility(xyz, xyz, LIBRARY).decide(
        np.empty(0, dtype="uint32"),
        empty32,
        empty32,
        empty32,
        empty64,
        empty64,
        camera(),
        mask,
    )
    assert not result.flags.any()
    assert not result.incidence.any()


@native_required
def test_native_visibility_preserves_nonzero_wide_masks():
    xyz = np.array([[0.0, 0.0, 1.0]], dtype="float32")
    normals = np.array([[0.0, 0.0, -1.0]], dtype="float32")
    frame = CameraFrame(
        "left", Path("unused"), 0.0, np.zeros(3), np.eye(3), camera().calibration
    )
    mask = np.zeros((48, 64), dtype="uint16")
    mask[0, 0] = 256
    result = CpuVisibility(xyz, normals, LIBRARY).decide(
        np.array([0], dtype="uint32"),
        np.array([0.25], dtype="float32"),
        np.array([0.25], dtype="float32"),
        np.array([1.0], dtype="float32"),
        np.array([0], dtype="uint64"),
        np.array([1_000_000], dtype="uint64"),
        frame,
        mask,
    )
    assert result.flags[0] & 4
    assert not result.flags[0] & 8


@native_required
def test_native_visibility_matches_threshold_neighbors_and_concurrent_calls():
    threshold = np.float32(0.15)
    denominator_x = np.array(
        [
            np.nextafter(threshold, np.float32(0)),
            threshold,
            np.nextafter(threshold, np.float32(1)),
            np.nextafter(-threshold, np.float32(0)),
            -threshold,
            np.nextafter(-threshold, np.float32(-1)),
            0.0,
        ],
        dtype="float32",
    )
    point_count = len(denominator_x) + 1
    xyz = np.zeros((point_count, 3), dtype="float32")
    xyz[0] = (1, 0, 0)
    xyz[1:, 0] = 1
    normals = np.zeros_like(xyz)
    normals[1:, 0] = denominator_x
    ids = np.zeros(len(denominator_x), dtype="uint32")
    blockers = np.arange(1, point_count, dtype="uint32")
    distance = np.ones(len(ids), dtype="float32")
    exact_keys = np.uint64(1_000_000 * point_count) + ids
    blocker_keys = blockers.astype("uint64")
    u = np.full(len(ids), 1.25, dtype="float32")
    v = np.full(len(ids), 1.25, dtype="float32")
    mask = np.zeros((48, 64), dtype="uint8")
    frame = CameraFrame(
        "left", Path("unused"), 0.0, np.zeros(3), np.eye(3), camera().calibration
    )
    expected = visibility_decisions(
        xyz, normals, ids, distance, blockers, exact_keys, point_count, frame, "mixed"
    )
    native = CpuVisibility(xyz, normals, LIBRARY)

    def run():
        return native.decide(ids, u, v, distance, blocker_keys, exact_keys, frame, mask)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: run(), range(8)))
    expected_flags = np.zeros(len(ids), dtype="uint8")
    expected_flags[expected[1]] |= 1
    expected_flags[expected[2]] |= 2
    expected_flags[expected[0]] |= 4 | 8
    for result in results:
        np.testing.assert_array_equal(result.flags, expected_flags)
        np.testing.assert_array_equal(result.incidence.view("uint32"), expected[3].view("uint32"))
