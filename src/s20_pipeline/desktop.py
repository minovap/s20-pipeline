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

PREVIEW_MAX = 60_000_000


def cloud_info(source):
    """Point count and name from the file header only."""
    source = source.resolve(strict=True)
    if source.suffix.lower() == ".las":
        import laspy

        with laspy.open(source) as reader:
            n = reader.header.point_count
    elif source.suffix.lower() == ".ply":
        from .ply import read_ply_info

        n = read_ply_info(source).point_count
    else:
        raise ValueError("Preview currently supports uncompressed LAS or binary PLY")
    return {"source": str(source), "name": source.name, "source_points": int(n)}


def preview(source, cache, budget):
    if not 10000 <= budget <= PREVIEW_MAX:
        raise ValueError(f"Preview point budget must be 10,000–{PREVIEW_MAX:,}")
    source = source.resolve(strict=True)
    key = hashlib.sha256(
        f"v2:{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{budget}".encode()
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
    # Positions as float32 and colors as 8-bit, in two files: 15 bytes per point.
    target = dest / "points.f32"
    colors_target = dest / "colors.u8"
    temporary = dest / "points.tmp"
    colors_temporary = dest / "colors.tmp"
    with temporary.open("wb") as stream, colors_temporary.open("wb") as color_stream:
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
            x.astype("<f4").tofile(stream)
            np.rint(np.clip(rgb[ids], 0, 1) * 255).astype("u1").tofile(color_stream)
            count += len(ids)
    temporary.replace(target)
    colors_temporary.replace(colors_target)
    result = {
        "key": key,
        "source": str(source),
        "name": source.name,
        "source_points": n,
        "display_points": count,
        "origin": origin.tolist(),
        "bounds": [low.tolist(), high.tolist()],
        "bytes": count * 12,
        "color_bytes": count * 3,
        "file": str(target.resolve()),
        "colors": str(colors_target.resolve()),
        "note": "Deterministic display sample; exported source is unchanged. This is not octree LOD.",
    }
    atomic_json(meta, result)
    return result


def _chunks(source):
    """Yield (xyz float64, rgb float 0-1 or None) chunks plus header info for a LAS/PLY source."""
    if source.suffix.lower() == ".las":
        import laspy

        reader = laspy.open(source)
        n = reader.header.point_count
        info = {"scales": reader.header.scales.tolist(), "offsets": reader.header.offsets.tolist()}

        def gen():
            with reader:
                for points in reader.chunk_iterator(262144):
                    xyz = np.column_stack([points.x, points.y, points.z])
                    names = set(points.point_format.dimension_names)
                    rgb = (
                        np.column_stack([points.red, points.green, points.blue]).astype("<f8")
                        / 65535
                        if "red" in names
                        else None
                    )
                    yield xyz, rgb

        return n, info, gen
    if source.suffix.lower() == ".ply":
        from .ply import map_points, read_ply_info

        data = map_points(read_ply_info(source))
        n = len(data)
        names = data.dtype.names
        has_rgb = all(k in names for k in ("red", "green", "blue"))

        def gen():
            for start in range(0, n, 262144):
                part = data[start : start + 262144]
                xyz = np.column_stack([part[k].astype("<f8") for k in ("x", "y", "z")])
                rgb = (
                    np.column_stack([part[k].astype("<f8") for k in ("red", "green", "blue")]) / 255
                    if has_rgb
                    else None
                )
                yield xyz, rgb

        return n, {"scales": [0.0001] * 3, "offsets": None}, gen
    raise ValueError("Slice export supports uncompressed LAS or binary PLY sources")


def _inside_polygon(x, y, vertices):
    """Even-odd point-in-polygon for arrays x, y."""
    inside = np.zeros(len(x), dtype=bool)
    n = len(vertices)
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[i - 1]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            at = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < at)
    return inside


def _distance_to_edges(x, y, vertices):
    best = np.full(len(x), np.inf)
    n = len(vertices)
    for i in range(n):
        ax, ay = vertices[i - 1]
        bx, by = vertices[i]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / l2, 0, 1) if l2 else np.zeros(len(x))
        ex, ey = ax + t * dx - x, ay + t * dy - y
        best = np.minimum(best, ex * ex + ey * ey)
    return np.sqrt(best)


def _region_mask(xyz, region):
    """Points inside a region: its box and every polygon test (inside, or a perimeter ring)."""
    low, high = np.asarray(region["box"], dtype="<f8")
    mask = np.all((xyz >= low) & (xyz <= high), axis=1)
    for test in region.get("polygons", []):
        vertices = np.asarray(test["vertices"], dtype="<f8")
        if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
            raise ValueError("Polygon needs at least three x y vertices")
        ids = np.flatnonzero(mask)
        if not len(ids):
            break
        x, y = xyz[ids, 0], xyz[ids, 1]
        inside = _inside_polygon(x, y, vertices)
        if test.get("mode") == "ring":
            keep = ~inside & (_distance_to_edges(x, y, vertices) <= float(test.get("expand", 0)))
        else:
            keep = inside
        mask[ids[~keep]] = False
    return mask


def export_slices(spec):
    """Write points inside the union of boxes per source to one LAS file.

    Each source has "boxes" and/or "regions" ({"box": ..., "polygons": [{"vertices": [[x, y], ...],
    "mode": "inside" | "ring", "expand": metres}]}); a point is kept when it satisfies any of them.

    spec = {"output": path, "sources": [{"path": str, "boxes": [[[minx,miny,minz],[maxx,maxy,maxz]], ...],
            "transform": {"rotation": [9 row-major], "origin": [3], "translation": [3]} | None}]}
    An optional transform maps source points to the calibrated frame first
    (world = R·(p - origin) + origin + t) and the file is written in that frame.
    A point is written once even when it lies inside several boxes of the same source.
    Progress lines are printed as JSON so the desktop app can show them.
    """
    import laspy

    output = Path(spec["output"])
    if output.suffix.lower() != ".las":
        raise ValueError("Slice export writes .las files")
    if output.exists():
        raise FileExistsError(f"Export already exists: {output}")
    sources = []
    total = 0
    for item in spec["sources"]:
        source = Path(item["path"]).resolve(strict=True)
        regions = [{"box": b} for b in item.get("boxes", [])] + list(item.get("regions", []))
        if not regions:
            raise ValueError("Nothing selected for export")
        for region in regions:
            box = np.asarray(region["box"], dtype="<f8")
            if box.shape != (2, 3) or not np.isfinite(box).all():
                raise ValueError("Boxes must be [[min xyz],[max xyz]] triples")
        boxes = regions
        n, info, gen = _chunks(source)
        transform = item.get("transform")
        if transform:
            rotation = np.asarray(transform["rotation"], dtype="<f8").reshape(3, 3)
            origin = np.asarray(transform["origin"], dtype="<f8")
            shift = origin + np.asarray(transform["translation"], dtype="<f8")
            if not (np.isfinite(rotation).all() and np.isfinite(shift).all()):
                raise ValueError("Transform must be finite")
            transform = (rotation, origin, shift)
        sources.append((source, boxes, n, info, gen, transform))
        total += n
    if not sources:
        raise ValueError("Nothing to export")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray(sources[0][3]["scales"], dtype="<f8")
    first_offsets = sources[0][3]["offsets"]
    if first_offsets is None:
        # PLY has no offsets; centre on the first box so int32 coordinates stay in range.
        first_offsets = np.floor(sources[0][1][0, 0])
    header.offsets = np.asarray(first_offsets, dtype="<f8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".las.tmp")
    done = 0
    written = 0
    last = 0.0
    import time

    with laspy.open(temporary, mode="w", header=header) as writer:
        for source, boxes, n, info, gen, transform in sources:
            for xyz, rgb in gen():
                if transform:
                    rotation, origin, shift = transform
                    xyz = (xyz - origin) @ rotation.T + shift
                mask = np.zeros(len(xyz), dtype=bool)
                for region in boxes:
                    mask |= _region_mask(xyz, region)
                ids = np.flatnonzero(mask)
                if len(ids):
                    records = laspy.ScaleAwarePointRecord.zeros(len(ids), header=header)
                    records.x = xyz[ids, 0]
                    records.y = xyz[ids, 1]
                    records.z = xyz[ids, 2]
                    colors = (
                        np.rint(np.clip(rgb[ids], 0, 1) * 65535).astype("uint16")
                        if rgb is not None
                        else np.full((len(ids), 3), 47000, dtype="uint16")
                    )
                    records.red = colors[:, 0]
                    records.green = colors[:, 1]
                    records.blue = colors[:, 2]
                    writer.write_points(records)
                    written += len(ids)
                done += len(xyz)
                now = time.monotonic()
                if now - last > 0.25:
                    last = now
                    print(
                        json.dumps({"event": "progress", "done": done, "total": total}), flush=True
                    )
    if not written:
        temporary.unlink(missing_ok=True)
        raise ValueError("No points inside the selected slices")
    temporary.replace(output)
    return {"event": "completed", "file": str(output), "points": written, "source_points": total}


def main():
    p = argparse.ArgumentParser()
    s = p.add_subparsers(dest="command", required=True)
    i = s.add_parser("inspect")
    i.add_argument("capture", type=Path)
    s.add_parser("hardware")
    c = s.add_parser("info")
    c.add_argument("source", type=Path)
    v = s.add_parser("preview")
    v.add_argument("source", type=Path)
    v.add_argument("cache", type=Path)
    v.add_argument("--budget", type=int, default=1000000)
    e = s.add_parser("export-slices")
    e.add_argument("spec", type=Path, help="JSON file: {output, sources:[{path, boxes}]}")
    a = p.parse_args()
    if a.command == "inspect":
        capture = inspect_capture(a.capture)
        host = hardware()
        result = {"capture": capture, "hardware": host, "estimate": estimate(capture, host)}
    elif a.command == "hardware":
        result = hardware()
    elif a.command == "info":
        result = cloud_info(a.source)
    elif a.command == "export-slices":
        result = export_slices(json.loads(a.spec.read_text()))
    else:
        result = preview(a.source, a.cache, a.budget)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
