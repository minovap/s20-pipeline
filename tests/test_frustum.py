import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from s20_pipeline.camera import CameraFrame, FisheyeCalibration, project
from s20_pipeline.frustum import PhotoPointIndex


def frame(coefficients=(0, 0, 0, 0, 0, 0)):
    cal = FisheyeCalibration("left", 160, 120, coefficients, 58, -3, 55, 79, 61, np.eye(4), 120)
    return CameraFrame("left", Path("unused.jpg"), 0, np.zeros(3), np.eye(3), cal)


def valid_projection(xyz, camera):
    u, v, theta, d = project(xyz, camera)
    cal = camera.calibration
    return (
        (d > 0.1)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (v >= 0)
        & (u < cal.width - 1)
        & (v < cal.height - 1)
        & (theta < np.deg2rad(cal.max_incident_angle_deg))
    )


@pytest.mark.parametrize(
    "coefficients",
    [
        (0, 0, 0, 0, 0, 0),
        (-0.00234, 0.02452, -0.02327, 0.03899, -0.02704, 0.00771),
        (0, -1, 0, 0, 0, 0),  # nonmonotonic and negative distortion
    ],
)
def test_culling_retains_every_valid_projection(coefficients):
    rng = np.random.default_rng(571)
    xyz = rng.uniform(-100, 100, (60000, 3)).astype("f4")
    # Voxel faces, axis singularities, near-camera points and >90-degree views.
    xyz = np.concatenate(
        [
            xyz,
            np.array(
                [[0, 0, 0], [0, 0, 1e-6], [0, 0, -30], [2, 2, 2], [-2, -2, -2], [50, 0, -1]],
                dtype="f4",
            ),
        ]
    )
    index = PhotoPointIndex(xyz, chunk=1709)
    for _ in range(4):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        camera = replace(
            frame(coefficients), camera_to_world=rotation, center=rng.uniform(-30, 30, 3)
        )
        retained = np.zeros(len(xyz), dtype=bool)
        ids = index.point_ids(camera)
        retained[ids] = True
        assert np.all(np.diff(ids) > 0)
        assert np.all(retained[valid_projection(xyz, camera)])
        assert retained.sum() < 0.9 * len(xyz)


def test_box_intersection_without_visible_corners_and_unlimited_range():
    camera = replace(frame(), calibration=replace(frame().calibration, a11=5000, a22=5000))
    xyz = np.array([[1, 1, 20], [1, 1, 1e6], [1, 1, -100]], dtype="f4")
    camera = replace(camera, center=np.array([1.0, 1.0, 0.0]))
    ids = PhotoPointIndex(xyz).point_ids(camera)
    assert 0 in ids and 1 in ids and 2 not in ids


def test_unencodable_grid_retains_all_points():
    xyz = np.array([[-1e20, -1e20, -1e20], [1e20, 1e20, 1e20]], dtype="f4")
    assert PhotoPointIndex(xyz).point_ids(frame()) is None


def test_dense_view_uses_contiguous_path():
    xyz = np.array([[0, 0, 1], [0, 0, 2]], dtype="f4")
    assert PhotoPointIndex(xyz).point_ids(frame()) is None


@pytest.mark.parametrize("kept_cells", [[], [2, 4], [1, 4], [1, 5], [0, 1, 3, 4, 5, 6]])
def test_compact_index_preserves_ids_across_selection_paths(monkeypatch, kept_cells):
    # Unequal leaf sizes and interleaved original IDs exercise both range ends
    # and restoration of original point order, including the 25% boundary.
    cells = np.repeat(np.arange(8), [1, 7, 2, 10, 3, 5, 8, 4])
    cells = cells[np.random.default_rng(181).permutation(len(cells))]
    xyz = np.column_stack([cells * 10, np.zeros(len(cells)), np.ones(len(cells))]).astype(
        "f4"
    )
    index = PhotoPointIndex(xyz, chunk=7)
    assert index.order.dtype == np.dtype("uint32")
    assert index.order.nbytes == 4 * len(xyz)
    assert index.starts.dtype == np.dtype("uint32")
    assert index.starts.nbytes == 4 * len(index.counts)
    keep = np.isin(np.arange(8), kept_cells)
    monkeypatch.setattr(index, "visible_voxels", lambda camera: keep)
    expected = np.flatnonzero(np.isin(cells, kept_cells))
    actual = index.point_ids(frame())
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.diff(actual.astype("i8")) > 0)
    if len(expected) <= len(xyz) // 4:
        assert actual.dtype == np.dtype("uint32")
        assert actual.nbytes == 4 * len(expected)
    else:
        assert actual.dtype == np.dtype(np.intp)


@pytest.mark.parametrize("focal_scale", [1, 4])
def test_collector_matches_unculled_with_masks_ties_and_chunk_boundaries(
    tmp_path, monkeypatch, focal_scale
):
    from PIL import Image

    import s20_pipeline.collect as module

    rng = np.random.default_rng(832)
    xyz = rng.uniform(-60, 60, (11000, 3)).astype("f4")
    xyz[1:4] = xyz[0]  # equal-depth original point-ID ties
    normals = -xyz / np.linalg.norm(xyz, axis=1)[:, None]
    points = np.zeros(
        len(xyz),
        dtype=[(name, "f4") for name in ("x", "y", "z", "normal_x", "normal_y", "normal_z")],
    )
    for i, key in enumerate(points.dtype.names):
        points[key] = np.column_stack([xyz, normals])[:, i]
    frames = []
    camera = frame()
    camera = replace(
        camera,
        calibration=replace(
            camera.calibration,
            a11=camera.calibration.a11 * focal_scale,
            a22=camera.calibration.a22 * focal_scale,
        ),
    )
    mask_root = tmp_path / "masks"
    (mask_root / "left_mask").mkdir(parents=True)
    for i in range(7):
        image = tmp_path / f"photo{i}.png"
        Image.fromarray(rng.integers(0, 256, (120, 160, 3), dtype="u1")).save(image)
        mask = np.zeros((120, 160), dtype="u1")
        mask[35:55, 45:85] = 255
        Image.fromarray(mask).save(mask_root / "left_mask" / image.name)
        frames.append(
            replace(
                camera,
                image_path=image,
                center=np.array([i * 3.0, 0, 0]) if i < 6 else np.array([0.0, 0.0, 200.0]),
            )
        )
    monkeypatch.setattr(module, "load_camera_frames", lambda *args: frames)
    monkeypatch.setattr(module, "read_ply_info", lambda *args: None)
    monkeypatch.setattr(module, "map_points", lambda *args: points)
    progress = []
    module.collect(
        image,
        image,
        image,
        mask_root,
        tmp_path / "culled",
        workers=3,
        chunk=193,
        progress=lambda *args: progress.append(args),
    )
    assert progress == [(i, 7) for i in range(1, 8)]
    meta = json.loads((tmp_path / "culled/candidates/meta.json").read_text())
    assert meta["images"][-1]["projected_points"] == 0
    if focal_scale == 4:
        assert 0 < meta["images"][0]["projected_points"] <= len(xyz) // 4

    class AllPoints:
        def __init__(self, xyz, **kwargs):
            self.count = len(xyz)

        def point_ids(self, camera):
            return np.arange(self.count)

    monkeypatch.setattr(module, "PhotoPointIndex", AllPoints)
    module.collect(image, image, image, mask_root, tmp_path / "reference", workers=1, chunk=1024)
    assert (tmp_path / "culled/candidates/observations.bin").read_bytes() == (
        tmp_path / "reference/candidates/observations.bin"
    ).read_bytes()
