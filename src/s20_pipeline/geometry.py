"""Convert retained native scan observations to the Metal filter interface."""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .observations import NOT_EXPORTED, read
from .storage import atomic_json


def _convert(path, destination, lookup):
    """One observation frame to one PCD file. Returns (origin_ns, pose row)."""
    header, records = read(path)
    frame = header["frame"]
    pose = (
        lookup[frame]
        if lookup is not None
        else np.r_[frame, header["end"], header["translation"], header["quaternion_xyzw"]]
    )
    if abs(pose[1] - header["end"]) > 1e-7:
        raise ValueError("Pose/observation time mismatch")
    ids = np.flatnonzero(records["export_rank"] != NOT_EXPORTED)
    ids = ids[np.argsort(records["export_rank"][ids])]
    points = np.zeros((len(ids), 8), dtype="<f4")
    points[:, :3] = records["xyz"][ids]
    points[:, 6] = records["intensity"][ids]
    points[:, 7] = (
        header["begin"] + records["offset_ns"][ids].astype("f8") * 1e-9 - pose[1]
    ) * 1000
    text = (
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z normal_x normal_y normal_z intensity curvature\n"
        "SIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\n"
        f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    with (destination / "SCANS" / f"{frame}.pcd").open("xb") as stream:
        stream.write(text.encode())
        points.tofile(stream)
    return header["origin_ns"], pose


def stage_registered(observations, destination, revised=None):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "SCANS").mkdir()
    lookup = {int(r[0]): r for r in np.atleast_2d(np.loadtxt(revised))} if revised else None
    paths = sorted(observations.glob("*.s20obs"))
    # Frames are independent; convert them in parallel, keeping frame order.
    workers = max(
        1, min(int(os.environ.get("S20_CPU_THREADS", os.cpu_count() or 1)), 8, len(paths) or 1)
    )
    rows, origin = [], None
    with ProcessPoolExecutor(workers) as pool:
        for frame_origin, pose in pool.map(
            _convert, paths, [destination] * len(paths), [lookup] * len(paths), chunksize=16
        ):
            if origin is not None and origin != frame_origin:
                raise ValueError("Mixed observation clock origins")
            origin = frame_origin
            rows.append(pose)
    if not rows:
        raise ValueError("No native observations")
    np.savetxt(
        destination / "FrameOptPose.txt",
        rows,
        fmt="%.17g",
        header=f"origin_ns {origin}\nframe seconds_from_raw_origin x y z qx qy qz qw",
    )
    atomic_json(
        destination / "metadata.json",
        {
            "frames": len(rows),
            "origin_ns": origin,
            "coordinate_frame": "LiDAR to native world",
            "uses_studio_poses": False,
        },
    )
