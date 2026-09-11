"""Vectorized S20 ROS1 extraction. Input is raw data only, never Studio poses."""

import json
import struct
import time

import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

DT = np.dtype(
    [
        ("offset", "<u4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("intensity", "u1"),
        ("tag", "u1"),
        ("line", "u1"),
    ]
)


def extract(source, dest):
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "scans").mkdir(exist_ok=False)
    store = get_typestore(Stores.ROS1_NOETIC)
    index = []
    imu = []
    tick = time.perf_counter()
    verified = False
    with Reader(source) as reader:
        conns = [c for c in reader.connections if c.topic in ["/livox/lidar", "/livox/imu"]]
        for c in conns:
            store.register(get_types_from_msg(c.msgdef.data, c.msgtype))
        for c, bt, raw in reader.messages(connections=conns):
            if c.topic == "/livox/imu":
                m = store.deserialize_ros1(raw, c.msgtype)
                t = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
                imu.append(
                    (
                        t,
                        m.angular_velocity.x,
                        m.angular_velocity.y,
                        m.angular_velocity.z,
                        m.linear_acceleration.x,
                        m.linear_acceleration.y,
                        m.linear_acceleration.z,
                    )
                )
                continue
            seq, sec, nsec, n = struct.unpack_from("<4I", raw, 0)
            at = 16 + n
            base, count = struct.unpack_from("<QI", raw, at)
            arraycount = struct.unpack_from("<I", raw, at + 16)[0]
            if count != arraycount or count == 0 or len(raw) != at + 20 + count * DT.itemsize:
                raise ValueError("Invalid Livox point message")
            points = np.frombuffer(raw, dtype=DT, count=count, offset=at + 20).copy()
            if not verified:
                m = store.deserialize_ros1(raw, c.msgtype)
                assert m.timebase == base and m.header.seq == seq
                for i in [0, count // 2, count - 1]:
                    p = m.points[i]
                    q = points[i]
                    assert (
                        p.offset_time == q["offset"]
                        and p.x == q["x"]
                        and p.y == q["y"]
                        and p.z == q["z"]
                    )
                verified = True
            fname = f"scans/{len(index):05d}.npy"
            np.save(dest / fname, points)
            index.append(
                {
                    "index": len(index),
                    "seq": seq,
                    "header_ns": sec * 10**9 + nsec,
                    "base_ns": base,
                    "end_ns": base + int(points["offset"].max()),
                    "bag_ns": bt,
                    "count": count,
                    "path": fname,
                }
            )
    if not index or not imu:
        raise ValueError("Capture must contain LiDAR and IMU messages")
    start = index[0]["base_ns"]
    im = np.array(imu, dtype=np.float64)
    im[:, 0] = [(row[0] - start) / 1e9 for row in imu]
    np.save(dest / "imu.npy", im)
    (dest / "frames.json").write_text(json.dumps(index, indent=2))
    summary = {
        "source": str(source),
        "time_origin_ns": start,
        "frames": len(index),
        "points": sum(r["count"] for r in index),
        "imu_samples": len(imu),
        "duration_s": (index[-1]["end_ns"] - start) / 1e9,
        "extraction_seconds": time.perf_counter() - tick,
        "vector_parser_checked_against_rosbags": verified,
        "first_2s_gyro_mean": im[im[:, 0] < 2, 1:4].mean(0).tolist(),
        "first_2s_gyro_std": im[im[:, 0] < 2, 1:4].std(0).tolist(),
        "first_2s_accel_mean": im[im[:, 0] < 2, 4:7].mean(0).tolist(),
    }
    (dest / "extraction.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
