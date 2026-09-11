"""Version 1 native scan-local observations; read-only helpers."""

import struct
from pathlib import Path

import numpy as np

HEADER = struct.Struct("<8sIIQIIIIdd7d3ddQQ")
DTYPE = np.dtype(
    [
        ("xyz", "<f8", (3,)),
        ("raw_xyz", "<f4", (3,)),
        ("range", "<f4"),
        ("sample", "<u4"),
        ("offset_ns", "<u4"),
        ("intensity", "u1"),
        ("tag", "u1"),
        ("line", "u1"),
        ("reserved", "u1"),
        ("export_rank", "<u4"),
    ]
)
assert HEADER.size == 160 and DTYPE.itemsize == 56
NOT_EXPORTED = 2**32 - 1


def read(path):
    path = Path(path)
    with path.open("rb") as f:
        data = f.read(HEADER.size)
        if len(data) != HEADER.size:
            raise ValueError("Truncated header")
        h = HEADER.unpack(data)
    if h[:3] != (b"S20OBS01", 160, 56) or path.stat().st_size != 160 + h[6] * 56:
        raise ValueError("Invalid observation format or size")
    if (
        not h[3]
        or h[21]
        or h[22]
        or not 0 <= h[7] <= h[6] <= h[5] <= 1000000
        or h[9] < h[8]
        or not np.isfinite(h[8:21]).all()
    ):
        raise ValueError("Invalid observation header")
    if abs(np.linalg.norm(h[13:17]) - 1) > 1e-8:
        raise ValueError("Invalid observation quaternion")
    header = {
        "origin_ns": h[3],
        "frame": h[4],
        "raw_count": h[5],
        "count": h[6],
        "export_count": h[7],
        "begin": h[8],
        "end": h[9],
        "translation": np.array(h[10:13]),
        "quaternion_xyzw": np.array(h[13:17]),
        "local_velocity": np.array(h[17:20]),
        "imu_shift": h[20],
    }
    records = (
        np.memmap(path, mode="r", dtype=DTYPE, offset=160, shape=(h[6],))
        if h[6]
        else np.empty(0, dtype=DTYPE)
    )
    if (
        not np.isfinite(records["xyz"]).all()
        or not np.isfinite(records["raw_xyz"]).all()
        or not np.isfinite(records["range"]).all()
    ):
        raise ValueError("Nonfinite observation")
    if (
        np.any(records["range"] < 0)
        or np.any(records["reserved"] != 0)
        or np.any(records["sample"] >= h[5])
        or np.any(np.diff(records["sample"].astype("i8")) <= 0)
        or np.any(records["offset_ns"] * 1e-9 > h[9] - h[8] + 1e-6)
    ):
        raise ValueError("Invalid observation records")
    ranks = records["export_rank"]
    ranks = ranks[ranks != NOT_EXPORTED]
    if len(ranks) != h[7] or not np.array_equal(np.sort(ranks), np.arange(h[7])):
        raise ValueError("Invalid observation export ranks")
    return header, records
