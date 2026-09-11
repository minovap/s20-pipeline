"""LAS export with bounded temporary arrays and stable source-point mapping."""

import laspy
import numpy as np

from .ply import map_points, read_ply_info
from .storage import atomic_json, digest


def export(geometry, candidates, colors, destination, chunk=262144):
    info = read_ply_info(geometry)
    points = map_points(info)
    c = np.memmap(candidates, dtype="<f4", mode="r", shape=(info.point_count, 4, 8))
    rgb = np.memmap(colors, dtype="<f4", mode="r", shape=(info.point_count, 4))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.full(3, 0.0001)
    # Centering offsets extends LAS int32 range for large local/survey coordinates.
    header.offsets = np.floor([float(points[k].min()) for k in ("x", "y", "z")])
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    blended = 0
    with (
        laspy.open(destination / "colorized.las", mode="w", header=header) as writer,
        (destination / "source-indices.u64").open("xb") as index_file,
    ):
        for start in range(0, info.point_count, chunk):
            ids = np.flatnonzero(c[start : start + chunk, 0, 7] > 0) + start
            if not len(ids):
                continue
            values = rgb[ids]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite blended colors")
            records = laspy.ScaleAwarePointRecord.zeros(len(ids), header=header)
            records.x = points["x"][ids]
            records.y = points["y"][ids]
            records.z = points["z"][ids]
            quant = np.rint(np.clip(values[:, :3], 0, 255) * 257).astype("uint16")
            records.red = quant[:, 0]
            records.green = quant[:, 1]
            records.blue = quant[:, 2]
            writer.write_points(records)
            ids.astype("<u8").tofile(index_file)
            count += len(ids)
            blended += int(values[:, 3].sum())
    if not count:
        raise ValueError("No visible colored points")
    atomic_json(
        destination / "result.json",
        {
            "input_points": info.point_count,
            "colored_points": count,
            "blended_points": blended,
            "source_sha256": digest(geometry),
            "output_sha256": digest(destination / "colorized.las"),
            "coordinate_quantization_m": 0.0001,
        },
    )
