#!/usr/bin/env python3
"""SHARE S20 calibration, pose interpolation, and polynomial fisheye projection."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FisheyeCalibration:
    name: str
    width: int
    height: int
    coefficients: tuple[float, float, float, float, float, float]
    a11: float
    a12: float
    a22: float
    u0: float
    v0: float
    lidar_to_camera: np.ndarray
    max_incident_angle_deg: float


@dataclass(frozen=True)
class CameraFrame:
    name: str
    image_path: Path
    timestamp: float
    center: np.ndarray
    camera_to_world: np.ndarray
    calibration: FisheyeCalibration


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + amount * (q1 - q0)
        return result / np.linalg.norm(result)
    angle = math.acos(max(-1.0, min(1.0, dot)))
    scale = math.sin(angle)
    return math.sin((1 - amount) * angle) / scale * q0 + math.sin(amount * angle) / scale * q1


def load_frame_poses(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            # Studio's optimized FrameOptPose rows begin with an integer frame
            # index, while the trajectory bundled in a recording begins with
            # the timestamp.  The remaining eight values have the same
            # timestamp/translation/quaternion layout.
            if len(fields) == 8:
                pose_fields = fields
            elif len(fields) >= 9:
                pose_fields = fields[1:9]
            else:
                raise ValueError(f"malformed frame pose line: {line.rstrip()}")
            rows.append([float(value) for value in pose_fields])
    result = np.asarray(rows, dtype=np.float64)
    if len(result) < 2 or np.any(np.diff(result[:, 0]) <= 0):
        raise ValueError("frame poses must contain increasing timestamps")
    return result


def interpolate_pose(frame_poses: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray]:
    index = int(np.searchsorted(frame_poses[:, 0], timestamp))
    if index == 0 or index >= len(frame_poses):
        raise ValueError(
            f"image timestamp {timestamp:.9f} lies outside frame pose interval "
            f"[{frame_poses[0, 0]:.9f}, {frame_poses[-1, 0]:.9f}]"
        )
    before = frame_poses[index - 1]
    after = frame_poses[index]
    amount = (timestamp - before[0]) / (after[0] - before[0])
    translation = before[1:4] * (1 - amount) + after[1:4] * amount
    rotation = quaternion_to_matrix(slerp(before[4:8], after[4:8], amount))
    return translation, rotation


def _section(text: str, start: str, stop: str | None) -> str:
    start_at = text.index(start)
    stop_at = text.find(stop, start_at + len(start)) if stop else -1
    return text[start_at : None if stop_at < 0 else stop_at]


def _number(section: str, key: str) -> float:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([-+0-9.eE]+)", section, re.MULTILINE)
    if not match:
        raise ValueError(f"missing calibration value {key}")
    return float(match.group(1))


def _matrix(text: str, key: str) -> np.ndarray:
    section = _section(text, f"   {key}:", None)
    match = re.search(r"data:\s*\[([^]]+)\]", section, re.DOTALL)
    if not match:
        raise ValueError(f"missing matrix {key}")
    values = [float(value) for value in match.group(1).replace("\n", " ").split(",")]
    if len(values) != 16:
        raise ValueError(f"expected 16 values in {key}; got {len(values)}")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def load_calibration(path: Path) -> tuple[dict[str, FisheyeCalibration], np.ndarray]:
    text = path.read_text(encoding="utf-8")
    calibrations: dict[str, FisheyeCalibration] = {}
    for name, stop in (("left", "   fisheye_right:"), ("right", "   fisheye_middle:")):
        section = _section(text, f"   fisheye_{name}:", stop)
        calibrations[name] = FisheyeCalibration(
            name=name,
            width=int(_number(section, "image_width")),
            height=int(_number(section, "image_height")),
            coefficients=tuple(_number(section, f"k{index}") for index in range(2, 8)),
            a11=_number(section, "A11"),
            a12=_number(section, "A12"),
            a22=_number(section, "A22"),
            u0=_number(section, "u0"),
            v0=_number(section, "v0"),
            lidar_to_camera=_matrix(text, f"lidar_{name}camera"),
            max_incident_angle_deg=_number(section, "maxIncidentAngle"),
        )
    match = re.search(r"^LIDAR_IMU_T:\s*\[([^]]+)\]", text, re.MULTILINE)
    if not match:
        raise ValueError("missing LIDAR_IMU_T")
    lidar_in_imu = np.asarray([float(value) for value in match.group(1).split(",")])
    return calibrations, lidar_in_imu


def camera_frame_from_imu_pose(
    name: str,
    image_path: Path,
    timestamp: float,
    imu_translation: np.ndarray,
    imu_to_world: np.ndarray,
    calibration: FisheyeCalibration,
    lidar_in_imu: np.ndarray,
) -> CameraFrame:
    extrinsic = calibration.lidar_to_camera
    camera_in_lidar = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    camera_in_imu = lidar_in_imu + camera_in_lidar
    center = imu_translation + imu_to_world @ camera_in_imu
    # Proper camera-to-world transform. The Studio TransformedCam JSON stores the
    # negative of this matrix (an improper OpenGL-style transform); projection is
    # equivalent after negating all three camera coordinates.
    camera_to_world = imu_to_world @ extrinsic[:3, :3].T
    return CameraFrame(
        name=name,
        image_path=image_path,
        timestamp=timestamp,
        center=center,
        camera_to_world=camera_to_world,
        calibration=calibration,
    )


def save_camera_frames(
    frames: list[CameraFrame], destination: Path, metadata: dict[str, object]
) -> None:
    payload = {
        "format": "share-native-camera-frames-v1",
        "metadata": metadata,
        "frames": [
            {
                "camera": frame.name,
                "image": str(frame.image_path),
                "timestamp": frame.timestamp,
                "center": frame.center.tolist(),
                "camera_to_world": frame.camera_to_world.tolist(),
            }
            for frame in frames
        ],
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_camera_frames(path: Path, calibration_path: Path) -> list[CameraFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibrations, _ = load_calibration(calibration_path)
    frames = []
    for item in payload["frames"]:
        name = item["camera"]
        frames.append(
            CameraFrame(
                name=name,
                image_path=Path(item["image"]).resolve(),
                timestamp=float(item["timestamp"]),
                center=np.asarray(item["center"], dtype=np.float64),
                camera_to_world=np.asarray(item["camera_to_world"], dtype=np.float64),
                calibration=calibrations[name],
            )
        )
    if not frames:
        raise ValueError("No camera frames")
    seen = set()
    for frame in frames:
        r = frame.camera_to_world
        if (
            frame.center.shape != (3,)
            or r.shape != (3, 3)
            or not np.isfinite(frame.center).all()
            or not np.isfinite(r).all()
        ):
            raise ValueError("Invalid camera transform")
        if np.linalg.norm(r.T @ r - np.eye(3)) > 1e-3 or abs(np.linalg.det(r) - 1) > 1e-3:
            raise ValueError("Camera transform must be a proper rotation")
        key = (frame.name, frame.image_path.stem)
        if key in seen:
            raise ValueError("Duplicate camera image identity")
        seen.add(key)
    return frames


def project(
    points_world: np.ndarray, frame: CameraFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points; return u, v, incident angle, and radial range."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera = (points_world - frame.center.astype(np.float32)) @ frame.camera_to_world.astype(
            np.float32
        )
    radial = np.hypot(camera[:, 0], camera[:, 1])
    theta = np.arctan2(radial, camera[:, 2])
    distorted = theta.copy()
    power = theta * theta
    for coefficient in frame.calibration.coefficients:
        distorted += coefficient * power
        power *= theta
    scale = np.divide(
        distorted,
        radial,
        out=np.ones_like(distorted),
        where=radial > 1e-10,
    )
    normalized_x = camera[:, 0] * scale
    normalized_y = camera[:, 1] * scale
    calibration = frame.calibration
    u = calibration.a11 * normalized_x + calibration.a12 * normalized_y + calibration.u0
    v = calibration.a22 * normalized_y + calibration.v0
    distance = np.linalg.norm(camera, axis=1)
    return u, v, theta, distance
