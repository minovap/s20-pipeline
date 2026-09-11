"""Recording-independent raw S20 packer with explicit calibration and manifest.

Consumes the existing extract_s20_raw.py output directory. ROS bag decoding stays
separate. Raw sources are read-only; this command requires a fresh output folder.
"""

import json
import struct
from pathlib import Path

import numpy as np
import yaml

from .storage import digest

POINT = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("offset", "<u4"),
        ("intensity", "u1"),
        ("tag", "u1"),
        ("line", "u1"),
        ("pad", "u1"),
    ]
)


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor(
    "tag:yaml.org,2002:opencv-matrix",
    lambda loader, node: loader.construct_mapping(node, deep=True),
)


def pack(source, calibration, out, progress=lambda *args: None):
    if out.exists():
        raise FileExistsError("Use a fresh package directory")
    cal = yaml.load(calibration.read_text().replace("%YAML:1.0", ""), Loader=Loader)
    rotation = np.asarray(cal["LIDAR_IMU_R"]["data"], dtype=float).reshape(3, 3)
    translation = np.asarray(cal["LIDAR_IMU_T"], dtype=float)
    shift = float(cal["IMU_time_offset"])
    if (
        translation.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError("Invalid extrinsics")
    if (
        np.linalg.norm(rotation.T @ rotation - np.eye(3)) > 1e-8
        or abs(np.linalg.det(rotation) - 1) > 1e-8
    ):
        raise ValueError("Calibration rotation is not rigid")
    if not np.isfinite(shift) or abs(shift) > 0.1:
        raise ValueError("Invalid IMU clock shift")
    frames = json.loads((source / "frames.json").read_text())
    imu = np.load(source / "imu.npy", mmap_mode="r")
    if (
        not frames
        or len(frames) > 1000000
        or imu.ndim != 2
        or imu.shape[1] != 7
        or not 100 <= len(imu) <= 10000000
    ):
        raise ValueError("Invalid input dimensions")
    if not np.isfinite(imu).all() or not (np.diff(imu[:, 0]) > 0).all():
        raise ValueError("Invalid IMU stream")
    origin = int(frames[0]["base_ns"])
    if not 0 < origin < 2**64:
        raise ValueError("Invalid recording origin")
    extraction = json.loads((source / "extraction.json").read_text())
    if (
        extraction["time_origin_ns"] != origin
        or extraction["frames"] != len(frames)
        or extraction["imu_samples"] != len(imu)
    ):
        raise ValueError("Mixed extraction origin or counts")
    # imu.npy times are relative to this recording origin, per extraction contract.
    if not (imu[0, 0] + shift <= 0.1 and imu[-1, 0] + shift > 0):
        raise ValueError("IMU and frame clocks do not overlap")
    paths = []
    last_end = -float("inf")
    for i, frame in enumerate(frames):
        if frame["index"] != i or frame["base_ns"] > frame["end_ns"] or frame["base_ns"] < last_end:
            raise ValueError("Unordered frame metadata")
        path = (source / frame["path"]).resolve()
        if not path.is_relative_to(source.resolve()):
            raise ValueError("Frame path escapes extraction folder")
        a = np.load(path, mmap_mode="r")
        duration = int(frame["end_ns"]) - int(frame["base_ns"])
        if (
            len(a) != frame["count"]
            or len(a) > 1000000
            or not all(k in a.dtype.names for k in POINT.names if k != "pad")
        ):
            raise ValueError("Invalid point records")
        if any(a.dtype[k] != POINT[k] for k in POINT.names if k != "pad"):
            raise ValueError("Unexpected point field units/types")
        if len(a) and (
            not (np.diff(a["offset"].astype("i8")) >= 0).all() or int(a["offset"].max()) > duration
        ):
            raise ValueError("Invalid per-point times")
        last_end = frame["end_ns"]
        paths.append(path)
    out.mkdir(parents=True)
    raw = out / "raw-native.bin"
    temp = out / "raw-native.bin.tmp"
    with temp.open("xb") as f:
        f.write(b"S20RAW02")
        f.write(
            struct.pack(
                "<Qd3d9dII", origin, shift, *translation, *rotation.ravel(), len(imu), len(frames)
            )
        )
        f.write(np.asarray(imu, dtype="<f8").tobytes())
        for number, (frame, path) in enumerate(zip(frames, paths), 1):
            if number % 50 == 0 or number == len(frames):
                progress(number, len(frames), "frames")
            a = np.load(path)
            b = np.zeros(len(a), dtype=POINT)
            for key in POINT.names:
                if key != "pad":
                    b[key] = a[key]
            f.write(
                struct.pack(
                    "<ddI",
                    (frame["base_ns"] - origin) / 1e9,
                    (frame["end_ns"] - origin) / 1e9,
                    len(a),
                )
            )
            f.write(b.tobytes())
    temp.replace(raw)
    sources = [
        calibration,
        source / "extraction.json",
        source / "frames.json",
        source / "imu.npy",
    ] + paths
    manifest = {
        "schema": 2,
        "format": "S20RAW02",
        "origin_ns": origin,
        "uses_studio_data": False,
        "coordinate_contract": "raw LiDAR XYZ; R maps LiDAR vectors into IMU; T is LiDAR origin expressed in IMU; q/pose exports remain LiDAR-to-world",
        "imu_time_contract": "imu.npy seconds from frames[0].base_ns; configured shift added once by engine",
        "lidar_imu_R": rotation.tolist(),
        "lidar_imu_T_m": translation.tolist(),
        "imu_shift_seconds": shift,
        "sources": [
            {"path": str(p.resolve()), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sources
        ],
        "output_sha256": digest(raw),
        "packer_sha256": digest(Path(__file__)),
        "imu_samples": len(imu),
        "frames": len(frames),
        "limits": "Does not infer a clock epoch from filenames or reuse another recording clock-sync map. Rejects malformed times; engine must skip scans without IMU coverage.",
    }
    (out / "input-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(raw),
                "frames": len(frames),
                "imu_samples": len(imu),
                "origin_ns": origin,
            }
        )
    )
