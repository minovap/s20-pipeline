import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import s20_pipeline.collect as module
from s20_pipeline.camera import CameraFrame, FisheyeCalibration


def camera():
    calibration = FisheyeCalibration(
        "left", 64, 48, (0.0,) * 6, 18.0, 0.0, 18.0, 32.0, 24.0, np.eye(4), 90.0
    )
    return CameraFrame("left", Path("unused"), 0.0, np.zeros(3), np.eye(3), calibration)


def geometry(count=803):
    rng = np.random.default_rng(939)
    xyz = rng.uniform([-3, -2, 0.2], [3, 2, 6], (count, 3)).astype("f4")
    xyz[[1, 102, 302]] = xyz[0]  # Equal keys, original-ID ties across workers.
    normals = -xyz / np.linalg.norm(xyz, axis=1)[:, None]
    points = np.zeros(
        count, dtype=[(name, "f4") for name in ("x", "y", "z", "normal_x", "normal_y", "normal_z")]
    )
    for index, name in enumerate(points.dtype.names):
        points[name] = np.column_stack([xyz, normals])[:, index]
    return points, xyz, normals


@pytest.mark.parametrize("mapped", [False, True])
def test_geometry_copies_own_data_and_close_source(tmp_path, monkeypatch, mapped):
    points, xyz, normals = geometry()
    if mapped:
        path = tmp_path / "source.bin"
        points.tofile(path)
        points = np.memmap(path, dtype=points.dtype, mode="r", shape=points.shape)
    monkeypatch.setattr(module, "read_ply_info", lambda _: None)
    monkeypatch.setattr(module, "map_points", lambda _: points)
    actual_xyz, actual_normals = module._load_geometry_arrays(Path("unused"))
    if mapped:
        assert points._mmap.closed
    else:
        assert not np.shares_memory(actual_xyz, points)
        assert not np.shares_memory(actual_normals, points)
    assert actual_xyz.flags.owndata and actual_normals.flags.owndata
    np.testing.assert_array_equal(actual_xyz, xyz)
    np.testing.assert_array_equal(actual_normals, normals)


def test_geometry_mapping_closed_when_a_required_field_is_missing(tmp_path, monkeypatch):
    source = np.zeros(3, dtype=[("x", "f4")])
    path = tmp_path / "source.bin"
    source.tofile(path)
    points = np.memmap(path, dtype=source.dtype, mode="r", shape=source.shape)
    monkeypatch.setattr(module, "read_ply_info", lambda _: None)
    monkeypatch.setattr(module, "map_points", lambda _: points)
    with pytest.raises(ValueError):
        module._load_geometry_arrays(path)
    assert points._mmap.closed


def test_fresh_observations_are_zero_and_existing_files_are_preserved(tmp_path):
    path = tmp_path / "observations.bin"
    records = module._empty_observations(path, 13, 4)
    assert path.stat().st_size == 13 * 4 * 8 * 4
    assert not records.any()
    records[3, 1, 7] = 7
    records.flush()
    with pytest.raises(FileExistsError):
        module._empty_observations(path, 13, 4)
    assert records[3, 1, 7] == 7
    records._mmap.close()


def test_pread_exact_retries_short_reads_and_rejects_truncation(monkeypatch):
    source = b"candidate-slots"
    monkeypatch.setattr(
        module.os,
        "pread",
        lambda _descriptor, size, offset: source[offset : offset + min(size, 3)],
    )
    assert module._pread_exact(7, len(source), 0) == source
    with pytest.raises(OSError, match="Unexpected end"):
        module._pread_exact(7, len(source) + 1, 0)


@pytest.mark.parametrize("workers", [1, 3])
def test_deferred_finalizer_matches_legacy_packing_and_stable_sort(tmp_path, workers):
    calibration = replace(camera().calibration, width=4, height=4)
    frames = []
    images = []
    rng = np.random.default_rng(238)
    for photo in range(4):
        path = tmp_path / f"photo-{photo}.png"
        image = rng.integers(0, 256, (4, 4, 3), dtype="u1")
        Image.fromarray(image).save(path)
        images.append(image)
        frames.append(replace(camera(), calibration=calibration, image_path=path))

    records = module._empty_observations(tmp_path / "observations.bin", 3, 4)
    for point, slot, photo, score, u, v in (
        (0, 0, 1, 0.5, 1.25, 1.5),
        (0, 1, 0, 0.7, 0.75, 2.1),
        (0, 2, 2, 0.5, 2.0, 0.25),
        (1, 0, 2, 0.1, 1.1, 1.2),
        (1, 1, 0, 0.9, 2.4, 2.3),
        (1, 2, 1, 0.4, 0.2, 0.6),
        (1, 3, 2, 0.8, 1.8, 1.7),
    ):
        records[point, slot, :2] = (u, v)
        records[point, slot, 6] = photo
        records[point, slot, 7] = score

    expected = np.asarray(records).copy()
    for photo, image in enumerate(images):
        points, slots = np.nonzero((expected[:, :, 7] > 0) & (expected[:, :, 6] == photo))
        u = expected[points, slots, 0]
        v = expected[points, slots, 1]
        x = u.astype("int32")
        y = v.astype("int32")
        p00 = image[y, x].astype("float32")
        p10 = image[y, x + 1].astype("float32")
        p01 = image[y + 1, x].astype("float32")
        p11 = image[y + 1, x + 1].astype("float32")
        a = (u - x)[:, None]
        b = (v - y)[:, None]
        expected[points, slots, :3] = (
            (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
        )
        expected[points, slots, 3] = np.maximum(abs(p10 - p00).max(1), abs(p01 - p00).max(1))
        expected[points, slots, 4] = np.clip(u / 4 * 8 - 0.5, 0, 7)
        expected[points, slots, 5] = np.clip(v / 4 * 6 - 0.5, 0, 5)
    order = np.argsort(-expected[:, :, 7], axis=1, kind="stable")
    expected = np.take_along_axis(expected, order[:, :, None], axis=1)

    actual, stats = module._finalize_observations(
        tmp_path, records, frames, chunk=1, grid=(8, 6), workers=workers
    )
    np.testing.assert_array_equal(actual, expected)
    assert stats["final_occupied"] == 7
    assert stats["represented_photos"] == 3
    assert not (tmp_path / "ranking-slots.bin").exists()


def test_deferred_finalizer_handles_no_candidates(tmp_path):
    records = module._empty_observations(tmp_path / "observations.bin", 3, 4)
    actual, stats = module._finalize_observations(
        tmp_path, records, [camera()], chunk=2, grid=(8, 6), workers=2
    )
    assert not actual.any()
    assert stats["final_occupied"] == 0
    assert stats["slot_index_bytes"] == 0
    assert not (tmp_path / "ranking-slots.bin").exists()


@pytest.mark.parametrize("selection", ["all", "sparse", "sparse_uint32", "empty"])
def test_parallel_projection_preserves_reference_keys_and_chunk_boundaries(selection):
    _, xyz, normals = geometry()
    selected = {
        "all": None,
        "sparse": np.arange(0, len(xyz), 2),
        "sparse_uint32": np.arange(0, len(xyz), 2, dtype="uint32"),
        "empty": np.array([], dtype=np.int64),
    }[selection]
    count = len(xyz) if selected is None else len(selected)
    frame = camera()
    reference = module.cpu_photo_decisions(xyz, normals, frame, selected, 101)
    outputs = []
    for workers in (1, 4):
        projection = np.empty((count, 4), dtype="f4")
        depth = np.full(16 * 12, np.iinfo(np.int64).max, dtype="i8")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            active = module._cpu_projection_depth(
                xyz, frame, selected, projection, depth, 101, workers, executor
            )
        assert active == (0 if count == 0 else workers)
        np.testing.assert_array_equal(projection, reference[0])
        valid = (reference[4] & module.VALID) != 0
        u, v = projection[valid, :2].T
        pixels = v.astype("int32") // 4 * 16 + u.astype("int32") // 4
        np.testing.assert_array_equal(depth[pixels], reference[2][valid])
        filled = module.minimum_depth_keys(depth.reshape(12, 16)).ravel()
        np.testing.assert_array_equal(filled[pixels], reference[3][valid])
        outputs.append((projection, depth))
    for serial, parallel in zip(outputs[0], outputs[1], strict=True):
        np.testing.assert_array_equal(serial, parallel)


def test_depth_workers_respect_chunks_requested_workers_and_scratch_budget():
    assert module._depth_worker_count(0, 1024, 8, 4096) == 0
    assert module._depth_worker_count(10, 1024, 8, 4096) == 1
    assert module._depth_worker_count(10000, 1024, 2, 4096) == 2
    assert module._depth_worker_count(10000, 1024, 128, 4096) == 4
    assert module._depth_worker_count(10000, 1024, 8, 70 << 20) == 1
    assert module._depth_worker_count(10000, 1024, 8, 256 << 20) == 1


@pytest.mark.parametrize("view", ["normal", "masked", "empty"])
def test_collector_serial_parallel_observations_and_unwritten_zeros(tmp_path, monkeypatch, view):
    points, _, _ = geometry()
    rng = np.random.default_rng(930)
    image = tmp_path / "photo.png"
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype="u1")).save(image)
    frame = replace(camera(), image_path=image)
    if view == "empty":
        frame = replace(frame, center=np.array([0.0, 0.0, 100.0]))
    mask_root = tmp_path / "masks"
    (mask_root / "left_mask").mkdir(parents=True)
    mask = np.full((48, 64), 255 if view == "masked" else 0, dtype="u1")
    Image.fromarray(mask).save(mask_root / "left_mask" / image.name)
    monkeypatch.setattr(module, "read_ply_info", lambda _: None)
    monkeypatch.setattr(module, "map_points", lambda _: points)
    monkeypatch.setattr(module, "load_camera_frames", lambda *_: [frame])
    outputs = []
    for workers in (1, 4):
        out = tmp_path / str(workers)
        progress = []
        module.collect(
            image,
            image,
            image,
            mask_root,
            out,
            workers=workers,
            chunk=101,
            progress=lambda *args: progress.append(args),
        )
        assert progress == [(1, 1)]
        records = (out / "candidates/observations.bin").read_bytes()
        outputs.append(records)
        meta = json.loads((out / "candidates/meta.json").read_text())
        assert (
            json.loads((out / "candidates/photo-progress.jsonl").read_text()) == meta["images"][0]
        )
        assert meta["images"][0]["cpu_depth_workers"] <= workers
        values = np.frombuffer(records, dtype="f4").reshape(-1, 4, 8)
        assert not values[:, 1:, :].any()
        if view == "normal":
            assert values[:, 0, 7].max() > 0
        else:
            assert not values.any()
    assert outputs[0] == outputs[1]
