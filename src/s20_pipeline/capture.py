"""Read-only S20 inventory, raw JPEG extraction and native camera poses."""

import json
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

from .camera import CameraFrame, interpolate_pose, load_calibration, save_camera_frames
from .storage import atomic_json, digest

CAMERA_TOPICS = {
    "/camera_agent/img_left/compressed": "left",
    "/camera_agent/img_right/compressed": "right",
}


def inspect_capture(folder):
    folder = Path(folder).resolve(strict=True)
    bags = sorted(folder.glob("all_*.bag"))
    if len(bags) != 1:
        raise ValueError("Expected one all_*.bag. Split/multiple bags are not supported yet.")
    calibration = folder / "info/calibration.yaml"
    if not calibration.is_file():
        raise ValueError("Missing info/calibration.yaml")
    cal, _ = load_calibration(calibration)
    with Reader(bags[0]) as reader:
        topics = {
            name: {"messages": info.msgcount, "type": info.msgtype}
            for name, info in reader.topics.items()
        }
        duration = reader.duration / 1e9
    if any(name not in topics for name in ("/livox/lidar", "/livox/imu")):
        raise ValueError("Missing raw LiDAR or IMU topic")
    metadata = folder / "project_info.json"
    device = json.loads(metadata.read_text()) if metadata.is_file() else {}
    return {
        "schema": 1,
        "capture": str(folder),
        "bag": str(bags[0]),
        "bag_bytes": bags[0].stat().st_size,
        "bag_duration_s": duration,
        "calibration": str(calibration),
        "calibration_sha256": digest(calibration),
        "device": {
            k: device.get(k)
            for k in (
                "device_model",
                "lidar_model",
                "work_duration",
                "project_finished",
                "rtk_fixed_count",
            )
        },
        "topics": topics,
        "lidar_frames": topics["/livox/lidar"]["messages"],
        "imu_samples": topics["/livox/imu"]["messages"],
        "photos": sum(t["messages"] for name, t in topics.items() if name in CAMERA_TOPICS),
        "cameras": {name: {"width": c.width, "height": c.height} for name, c in cal.items()},
        "clock_status": "unverified until raw sensor timestamps are checked against native poses",
        "point_count": None,
        "point_count_note": "Requires reading Livox payloads; bag index counts frames only.",
    }


def extract_photos(bag, destination, progress=lambda *args: None):
    destination.mkdir(parents=True, exist_ok=False)
    store = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic in CAMERA_TOPICS]
        total = sum(c.msgcount for c in connections)
        for connection, _, raw in reader.messages(connections=connections):
            msg = store.deserialize_ros1(raw, connection.msgtype)
            side = CAMERA_TOPICS[connection.topic]
            time_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            path = destination / side / f"{msg.header.seq:08d}-{time_ns}.jpg"
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(msg.data.tobytes())
            rows.append(
                {
                    "camera": side,
                    "image": str(path.resolve()),
                    "sensor_ns": time_ns,
                    "sequence": msg.header.seq,
                    "sha256": digest(path),
                }
            )
            progress(len(rows), total)
    if not rows:
        raise ValueError("No supported main-camera images")
    atomic_json(destination / "images.json", rows)


def prepare_cameras(images, poses, calibration, destination, max_gap=0.5):
    """Explicit sensor-header clock to native raw-origin pose time, never Studio epoch."""
    rows = np.atleast_2d(np.loadtxt(poses))
    origin = int(
        next(
            line.split()[2]
            for line in poses.read_text().splitlines()
            if line.startswith("# origin_ns ")
        )
    )
    if (
        len(rows) < 2
        or rows.shape[1] != 9
        or not np.isfinite(rows).all()
        or not np.all(np.diff(rows[:, 1]) > 0)
    ):
        raise ValueError("Invalid native pose table")
    calibrations, _ = load_calibration(calibration)
    selected, rejected = [], []
    for item in json.loads(images.read_text()):
        relative = (int(item["sensor_ns"]) - origin) / 1e9
        index = int(np.searchsorted(rows[:, 1], relative))
        reason = None
        if index == 0 or index >= len(rows):
            reason = "outside pose coverage"
        elif rows[index, 1] - rows[index - 1, 1] > max_gap:
            reason = "pose gap exceeds limit"
        if reason:
            rejected.append({"image": item["image"], "reason": reason})
            continue
        translation, rotation = interpolate_pose(rows[:, 1:9], relative)
        cal = calibrations[item["camera"]]
        # Poses are LiDAR-to-world, so no IMU lever or device-wall-clock conversion.
        camera_to_lidar = cal.lidar_to_camera[:3, :3].T
        center = translation + rotation @ (-camera_to_lidar @ cal.lidar_to_camera[:3, 3])
        selected.append(
            CameraFrame(
                item["camera"],
                Path(item["image"]),
                relative,
                center,
                rotation @ camera_to_lidar,
                cal,
            )
        )
    if not selected:
        raise ValueError(
            "No camera/pose clock overlap. Supply a verified clock adapter; do not guess an offset."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_camera_frames(
        selected,
        destination,
        {
            "origin_ns": origin,
            "clock": "sensor-header minus native raw origin",
            "calibration": "recorded LiDAR-to-camera",
            "max_pose_gap_s": max_gap,
            "rejected": rejected,
            "native_pose_sha256": digest(poses),
            "camera_geometry_requires_dataset_validation": True,
        },
    )
    return len(selected)
