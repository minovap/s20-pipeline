import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from s20_pipeline.collect import minimum_depth_keys
from s20_pipeline.estimate import estimate
from s20_pipeline.exposure import Exposure
from s20_pipeline.runner import files_identity, run
from s20_pipeline.storage import validate_destination


def test_packed_depth_preserves_ids_above_double_precision():
    values = np.array([[2**60 + 13, 2**60 + 9], [2**60 + 11, 2**60 + 12]], dtype="i8")
    assert np.array_equal(
        minimum_depth_keys(values, radius=1), np.full((2, 2), 2**60 + 9, dtype="i8")
    )


def candidates(folder, count=3, images=2):
    dest = folder / "candidates"
    dest.mkdir()
    records = np.zeros((count, 4, 8), dtype="f4")
    records[:, :, :3] = 100
    records[:, 0, 7] = 1
    records[:, 1, 7] = 0.9
    records[:, 1, 6] = images - 1
    records.tofile(dest / "observations.bin")
    (dest / "meta.json").write_text(json.dumps({"points": count}))
    return records


@pytest.mark.parametrize("images", [1, 2, 63])
def test_exposure_dynamic_photo_count_and_empty_overlap(tmp_path, images):
    records = candidates(tmp_path, images=images)
    records[:, :, 7] = 0
    records.tofile(tmp_path / "candidates/observations.bin")
    ex = Exposure(tmp_path, images)
    ex.solve("global")
    ex.solve("local")
    field = np.load(tmp_path / "local/field.npy")
    assert field.shape == (images, 6, 8, 3)
    assert np.isfinite(field).all() and not field.any()
    assert (
        json.loads((tmp_path / "local/solve.json").read_text())["heldout_after_channel_p95"] is None
    )


def test_consensus_rejects_strong_bad_view(tmp_path):
    ex = Exposure(tmp_path, 3)
    records = candidates(tmp_path, images=3)
    records[:, 0, :3] = [40, 40, 40]
    records[:, 1, :3] = [150, 150, 150]
    records[:, 2, :3] = [152, 152, 152]
    records[:, 2, 6] = 1
    records[:, 2, 7] = 0.8
    result = ex.reference(records, np.zeros((3, 6, 8, 3), dtype="f4"), robust=True)
    assert (result[:, :3] > 149).all() and (result[:, :3] < 153).all()


def test_destination_resolves_symlink(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(raw)
    with pytest.raises(ValueError):
        validate_destination(alias / "result", [raw])
    with pytest.raises(ValueError):
        validate_destination(tmp_path, [raw])


def test_estimate_does_not_scale_from_cpu_core_count():
    data = {"lidar_frames": 671, "photos": 58}
    result = estimate(data, {"cpu_model": "Other hardware", "logical_cpu_cores": 64})
    assert result["estimated_seconds"] is None
    assert result["reference_machine_seconds"] == pytest.approx(71.02)
    large = estimate(
        {"lidar_frames": 671 * 20, "photos": 58 * 20},
        {"cpu_model": "M4 Max", "logical_cpu_cores": 16},
    )
    assert large["stages_on_reference_machine"][
        "visibility_exposure_blend_export_s"
    ] == pytest.approx(25.2 * 400)


def test_resume_refuses_modified_stage(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "receipts").mkdir()
    (output / "geometry").mkdir()
    (output / "geometry/cloud.txt").write_text("original")
    report = {"outputs": files_identity(output / "geometry")}
    (output / "receipts/geometry.json").write_text(json.dumps(report))
    (output / "geometry/cloud.txt").write_text("modified")
    job = {
        "output": str(output),
        "options": {"color": False, "cpu_threads": 1, "pose_refinement": False},
        "mode": "test",
    }
    # Select a completed stage without executing a processing worker.
    from unittest.mock import patch

    (output / "job.json").write_text(json.dumps(job))
    with patch("s20_pipeline.runner.stage_names", return_value=["geometry"]):
        with pytest.raises(ValueError, match="Completed stage changed"):
            run(job, resume=True)
    assert json.loads((output / "state.json").read_text())["status"] == "failed"


def test_metal_dynamic_field_and_numpy_agreement(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    binary = repo / "build/s20_blend"
    if os.environ.get("S20_SKIP_GPU_TESTS") == "1":
        pytest.skip("GPU checks explicitly disabled; run on a physical Metal-capable Mac")
    if not binary.exists():
        pytest.skip("Build Metal worker to exercise GPU parity")
    rng = np.random.default_rng(17)
    for image_count in (2, 63):
        records = rng.uniform(0, 1, (193, 4, 8)).astype("f4")
        records[:, :, :3] *= 255
        records[:, :, 3] *= 35
        records[:, :, 4] *= 7
        records[:, :, 5] *= 5
        records[:, :, 6] = rng.integers(0, image_count, (193, 4))
        records[:, 0, 7] = 1
        field = rng.uniform(-20, 20, (image_count, 6, 8, 3)).astype("f4")
        records.tofile(tmp_path / "observations.bin")
        field.tofile(tmp_path / "field.bin")
        subprocess.run(
            [
                str(binary),
                str(repo / "native/blend.metal"),
                str(tmp_path / "observations.bin"),
                str(tmp_path / "field.bin"),
                str(tmp_path / "colors.bin"),
                str(tmp_path / "metal.json"),
            ],
            check=True,
            capture_output=True,
        )
        actual = np.fromfile(tmp_path / "colors.bin", dtype="f4").reshape(-1, 4)
        expected = Exposure(tmp_path, image_count).reference(records, field, robust=True)
        assert np.max(abs(actual - expected)) < 0.015
    # Invalid image address must fail on the host, before GPU dispatch.
    records[0, 0, 6] = image_count
    records.tofile(tmp_path / "observations.bin")
    result = subprocess.run(
        [
            str(binary),
            str(repo / "native/blend.metal"),
            str(tmp_path / "observations.bin"),
            str(tmp_path / "field.bin"),
            str(tmp_path / "colors.bin"),
            str(tmp_path / "metal.json"),
        ],
        capture_output=True,
    )
    assert result.returncode == 5


def test_chunking_does_not_lose_foreground_occluder(tmp_path, monkeypatch):
    import s20_pipeline.collect as module
    from PIL import Image
    from s20_pipeline.camera import CameraFrame, FisheyeCalibration

    image = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), (210, 80, 30)).save(image)
    cal = FisheyeCalibration("left", 8, 8, (0, 0, 0, 0, 0, 0), 1, 0, 1, 3, 3, np.eye(4), 80)
    frame = CameraFrame("left", image, 0, np.zeros(3), np.eye(3), cal)
    points = np.zeros(
        2, dtype=[(name, "f4") for name in ("x", "y", "z", "normal_x", "normal_y", "normal_z")]
    )
    points["z"] = [1, 2]
    points["normal_z"] = -1
    monkeypatch.setattr(module, "load_camera_frames", lambda *args: [frame])
    monkeypatch.setattr(module, "read_ply_info", lambda *args: None)
    monkeypatch.setattr(module, "map_points", lambda *args: points)
    for name, chunk in [("split", 1), ("whole", 2)]:
        module.collect(image, image, image, None, tmp_path / name, workers=2, chunk=chunk)
    split = np.fromfile(tmp_path / "split/candidates/observations.bin", dtype="f4").reshape(2, 4, 8)
    whole = np.fromfile(tmp_path / "whole/candidates/observations.bin", dtype="f4").reshape(2, 4, 8)
    assert np.array_equal(split, whole)
    assert split[0, 0, 7] > 0 and split[1, 0, 7] == 0


def test_camera_clock_uses_raw_origin_and_rejects_gap(tmp_path, monkeypatch):
    from s20_pipeline import capture
    from s20_pipeline.camera import FisheyeCalibration

    cal = FisheyeCalibration("left", 8, 8, (0, 0, 0, 0, 0, 0), 1, 0, 1, 3, 3, np.eye(4), 80)
    monkeypatch.setattr(capture, "load_calibration", lambda *args: ({"left": cal}, np.zeros(3)))
    poses = tmp_path / "poses.txt"
    poses.write_text(
        "# origin_ns 1000000000000\n0 0 0 0 0 0 0 0 1\n1 .1 1 0 0 0 0 0 1\n2 2 2 0 0 0 0 0 1\n"
    )
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            [
                {"sensor_ns": 1000050000000, "camera": "left", "image": "photo.jpg"},
                {"sensor_ns": 1001000000000, "camera": "left", "image": "gap.jpg"},
            ]
        )
    )
    target = tmp_path / "frames.json"
    assert capture.prepare_cameras(images, poses, tmp_path / "calibration", target) == 1
    result = json.loads(target.read_text())
    assert result["frames"][0]["center"] == [0.5, 0.0, 0.0]
    assert result["metadata"]["rejected"][0]["reason"] == "pose gap exceeds limit"
    images.write_text(
        json.dumps([{"sensor_ns": 2000050000000, "camera": "left", "image": "wrong-clock.jpg"}])
    )
    with pytest.raises(ValueError, match="No camera/pose clock overlap"):
        capture.prepare_cameras(images, poses, tmp_path / "calibration", target)


def test_cancel_terminates_worker_process_group():
    import sys

    import psutil
    from s20_pipeline.runner import stop

    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            'import subprocess,time; p=subprocess.Popen(["sleep","30"]); print(p.pid,flush=True); time.sleep(30)',
        ],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    grandchild = int(child.stdout.readline())
    stop(child)
    assert child.poll() is not None
    try:
        process = psutil.Process(grandchild)
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    child.stdout.close()


def test_scheduler_runs_color_prep_beside_geometry():
    from s20_pipeline.runner import blocked, ready

    names = [
        "decode",
        "pack",
        "tracking",
        "pose_refinement",
        "registered",
        "geometry",
        "photos",
        "cameras",
        "masks",
        "candidates",
        "global",
        "local",
        "blend",
        "export",
    ]
    assert ready(names, set(), set()) == ["decode", "photos"]
    assert ready(names, {"decode", "photos"}, set()) == ["pack", "masks"]
    # cameras needs registered poses; with masks done the color lane waits for geometry
    assert ready(names, {"decode", "photos", "masks"}, {"pack"}) == []
    assert list(blocked(names, {"decode", "photos", "masks"}, {"pack"})) == [
        ("cameras", ["registered"])
    ]
    done = {"decode", "pack", "tracking", "pose_refinement", "registered", "photos", "masks"}
    assert ready(names, done, {"geometry"}) == ["cameras"]
    assert list(blocked(names, done | {"cameras"}, {"geometry"})) == [("candidates", ["geometry"])]
    # colorize mode has no photo stage, so masks depend on nothing
    assert ready(["masks", "candidates", "blend", "export"], set(), set()) == ["masks"]
