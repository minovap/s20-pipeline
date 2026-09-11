"""Desktop bridge: capture overview and bounded binary cloud previews."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .capture import inspect_capture
from .estimate import estimate
from .storage import atomic_json
from .telemetry import hardware


def preview(source, cache, budget):
    if not 10000 <= budget <= 2000000:
        raise ValueError("Preview point budget must be 10,000–2,000,000")
    source = source.resolve(strict=True)
    key = hashlib.sha256(
        f"{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{budget}".encode()
    ).hexdigest()
    dest = cache / key
    meta = dest / "preview.json"
    if meta.exists():
        return json.loads(meta.read_text())
    dest.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".las":
        import laspy

        reader = laspy.open(source)
        n = reader.header.point_count
        origin = (reader.header.mins + reader.header.maxs) / 2

        def chunks():
            with reader:
                for points in reader.chunk_iterator(262144):
                    xyz = np.column_stack([points.x, points.y, points.z])
                    names = set(points.point_format.dimension_names)
                    rgb = (
                        np.column_stack([points.red, points.green, points.blue]) / 65535
                        if "red" in names
                        else np.full_like(xyz, 0.72)
                    )
                    yield xyz, rgb
    elif source.suffix.lower() == ".ply":
        from .ply import map_points, read_ply_info

        data = map_points(read_ply_info(source))
        n = len(data)
        origin = np.array(
            [(float(data[k].min()) + float(data[k].max())) / 2 for k in ("x", "y", "z")]
        )

        def chunks():
            for start in range(0, n, 262144):
                part = data[start : start + 262144]
                xyz = np.column_stack([part[k] for k in ("x", "y", "z")])
                yield xyz, np.tile([0.52, 0.73, 0.70], (len(part), 1))
    else:
        raise ValueError("Preview currently supports uncompressed LAS or binary PLY")
    if not n or not np.isfinite(origin).all():
        raise ValueError("Invalid or empty point cloud")
    step = max(1, math.ceil(n / budget))
    cursor = 0
    count = 0
    low = np.full(3, np.inf)
    high = -low
    target = dest / "points.f32"
    temporary = dest / "points.tmp"
    with temporary.open("wb") as stream:
        for xyz, rgb in chunks():
            ids = np.arange((-cursor) % step, len(xyz), step)
            cursor += len(xyz)
            x = xyz[ids] - origin
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite point coordinates")
            if not len(x):
                continue
            low = np.minimum(low, x.min(0))
            high = np.maximum(high, x.max(0))
            np.column_stack([x, np.clip(rgb[ids], 0, 1)]).astype("<f4").tofile(stream)
            count += len(ids)
    temporary.replace(target)
    result = {
        "key": key,
        "source": str(source),
        "name": source.name,
        "source_points": n,
        "display_points": count,
        "origin": origin.tolist(),
        "bounds": [low.tolist(), high.tolist()],
        "bytes": count * 24,
        "file": str(target.resolve()),
        "note": "Deterministic display sample; exported source is unchanged. This is not octree LOD.",
    }
    atomic_json(meta, result)
    return result


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)
    i = s.add_parser("inspect")
    i.add_argument("capture", type=Path)
    s.add_parser("hardware")
    v = s.add_parser("preview")
    v.add_argument("source", type=Path)
    v.add_argument("cache", type=Path)
    v.add_argument("--budget", type=int, default=1000000)
    a = p.parse_args()
    if a.command == "inspect":
        capture = inspect_capture(a.capture)
        host = hardware()
        result = {"capture": capture, "hardware": host, "estimate": estimate(capture, host)}
    elif a.command == "hardware":
        result = hardware()
    else:
        result = preview(a.source, a.cache, a.budget)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
