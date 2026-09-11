"""Convert retained native scan observations to the Metal filter interface."""

import numpy as np

from .observations import NOT_EXPORTED, read
from .storage import atomic_json


def stage_registered(observations, destination, revised=None):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "SCANS").mkdir()
    lookup = {int(r[0]): r for r in np.atleast_2d(np.loadtxt(revised))} if revised else None
    rows, origin = [], None
    for path in sorted(observations.glob("*.s20obs")):
        header, records = read(path)
        if origin is not None and origin != header["origin_ns"]:
            raise ValueError("Mixed observation clock origins")
        origin = header["origin_ns"]
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
