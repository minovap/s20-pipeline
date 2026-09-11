from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from s20_pipeline.camera import CameraFrame, FisheyeCalibration
from s20_pipeline.collect import cpu_photo_decisions
from s20_pipeline.metal_collect import MetalCollector

REPOSITORY = Path(__file__).resolve().parents[1]
LIBRARY = REPOSITORY / "build/libs20_collector.dylib"
KERNELS = REPOSITORY / "native/collector.metal"
metal_required = pytest.mark.skipif(not LIBRARY.is_file(), reason="Metal collector is not built")


def frame(width=160, height=120, coefficients=(0, 0, 0, 0, 0, 0)):
    calibration = FisheyeCalibration(
        "left",
        width,
        height,
        coefficients,
        58,
        -3,
        55,
        (width - 1) / 2,
        (height - 1) / 2,
        np.eye(4),
        120,
    )
    return CameraFrame("left", Path("unused.jpg"), 0, np.zeros(3), np.eye(3), calibration)


@metal_required
def test_metal_depth_ties_neighborhood_axis_and_visibility_match_cpu():
    rng = np.random.default_rng(944)
    xyz = rng.normal(size=(4096, 3)).astype("f4")
    xyz[:, 2] += 4
    xyz[:4] = [[0, 0, 1], [0, 0, 1], [0.001, 0, 1], [0.3, 0, 1.02]]
    normals = (-xyz / np.linalg.norm(xyz, axis=1)[:, None]).astype("f4")
    camera = frame(coefficients=(-0.00234, 0.02452, -0.02327, 0.03899, -0.02704, 0.00771))
    expected = cpu_photo_decisions(xyz, normals, camera, None, 193)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=193)
    actual = collector.process(np.arange(len(xyz), dtype="u4"), camera)
    for reference, result in zip(
        expected[1:],
        (actual.depth_keys, actual.exact_keys, actual.blocker_keys, actual.flags),
        strict=True,
    ):
        np.testing.assert_array_equal(result, reference)


@metal_required
def test_metal_collector_reuses_buffers_across_empty_and_changing_selections():
    xyz = np.array([[0, 0, 1], [0.1, 0, 1.1], [-0.1, 0, 1.2]], dtype="f4")
    normals = np.array([[0, 0, -1]] * 3, dtype="f4")
    camera = frame(64, 64)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=1)
    for selected in (
        np.array([], dtype="u4"),
        np.array([1], dtype="u4"),
        np.array([2, 0, 1], dtype="u4"),
        np.array([], dtype="u4"),
        np.array([0, 2], dtype="u4"),
    ):
        expected = cpu_photo_decisions(xyz, normals, camera, selected, 1)
        actual = collector.process(selected, camera)
        for reference, result in zip(
            expected[1:],
            (actual.depth_keys, actual.exact_keys, actual.blocker_keys, actual.flags),
            strict=True,
        ):
            np.testing.assert_array_equal(result, reference)


@metal_required
def test_large_camera_center_uses_mixed_precision_visibility_recheck():
    xyz = np.array([[2e6, 0, 1], [2e6, 0, 1.029]], dtype="f4")
    normals = np.array([[0, 0, -1], [0, 0, -1]], dtype="f4")
    base = frame(64, 64)
    camera = replace(
        base,
        center=np.array([2000000.05, 0, 0]),
        calibration=replace(base.calibration, a11=10, a12=0, a22=10, u0=32, v0=32),
    )
    expected = cpu_photo_decisions(xyz, normals, camera, None, 2)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=2)
    actual = collector.process(np.arange(2, dtype="u4"), camera)
    np.testing.assert_array_equal(actual.flags, expected[4])
    assert actual.visibility_rechecks == 2


@metal_required
def test_large_exact_world_coordinates_use_mixed_precision_visibility_recheck():
    xyz = np.array([[2000002, 0, 1], [2000002, 0, 1.03]], dtype="f4")
    normals = np.array([[0, 0, -1], [0, 0, -1]], dtype="f4")
    base = frame(64, 64)
    camera = replace(
        base,
        center=np.array([2000000, 0, 0], dtype="f8"),
        calibration=replace(base.calibration, a11=10, a12=0, a22=10, u0=32, v0=32),
    )
    expected = cpu_photo_decisions(xyz, normals, camera, None, 2)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=1)
    actual = collector.process(np.arange(2, dtype="u4"), camera)
    np.testing.assert_array_equal(actual.flags, expected[4])
    assert actual.visibility_rechecks == 2


@metal_required
def test_conditioned_plane_intersection_uses_mixed_precision_visibility_recheck():
    xyz = np.array(
        [
            [19.60161018371582, -472.551025390625, 680.5578002929688],
            [19.602079391479492, -472.5623474121094, 680.5740966796875],
        ],
        dtype="f4",
    )
    normal = [-0.14729951322078705, 0.8949989676475525, 0.4210459887981415]
    normals = np.array([normal, normal], dtype="f4")
    base = frame(64, 64)
    camera = replace(
        base,
        calibration=replace(base.calibration, a11=10, a12=0, a22=10, u0=32, v0=32),
    )
    expected = cpu_photo_decisions(xyz, normals, camera, None, 2)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=1)
    assert collector.coordinate_spacing < 1e-4
    actual = collector.process(np.arange(2, dtype="u4"), camera)
    np.testing.assert_array_equal(actual.flags, expected[4])
    assert actual.visibility_rechecks >= 1


@metal_required
def test_subthreshold_center_rounding_is_included_in_visibility_error_bound():
    center = np.array([512.00003, 0, 0], dtype="f8")
    relative = np.array(
        [
            [19.60161018371582, -472.551025390625, 680.5578002929688],
            [19.602079391479492, -472.5623474121094, 680.5740966796875],
        ],
        dtype="f4",
    )
    xyz = relative.copy()
    xyz[:, 0] += np.float32(center[0])
    normal = [-0.14729951322078705, 0.8949989676475525, 0.4210459887981415]
    normals = np.array([normal, normal], dtype="f4")
    base = frame(64, 64)
    camera = replace(
        base,
        center=center,
        calibration=replace(base.calibration, a11=10, a12=0, a22=10, u0=32, v0=32),
    )
    expected = cpu_photo_decisions(xyz, normals, camera, None, 2)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=1)
    assert collector.coordinate_spacing < 1e-4
    assert 0 < np.linalg.norm(center - center.astype("f4")) < 1e-4
    actual = collector.process(np.arange(2, dtype="u4"), camera)
    np.testing.assert_array_equal(actual.flags, expected[4])
    assert actual.visibility_rechecks >= 1


@metal_required
def test_integer_pixel_boundary_is_rechecked_for_mask_sampling():
    xyz = np.array([[0.06034594029188156, 0.2, 1]], dtype="f4")
    normals = np.array([[0, 0, -1]], dtype="f4")
    base = frame(64, 64)
    camera = replace(
        base,
        calibration=replace(
            base.calibration,
            coefficients=(0.013, -0.004, 0.0003, 0, 0, 0),
            a11=10,
            a12=0.17,
            a22=10,
            u0=32.37,
            v0=32.6,
        ),
    )
    expected = cpu_photo_decisions(xyz, normals, camera, None, 1)
    collector = MetalCollector(xyz, normals, LIBRARY, KERNELS, chunk=1)
    actual = collector.process(np.arange(1, dtype="u4"), camera)
    assert actual.projection_rechecks == 1
    np.testing.assert_array_equal(actual.projection, expected[0])
