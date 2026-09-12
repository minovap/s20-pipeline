#!/usr/bin/env python3
"""Publish the exact control and three experiments to the local 3-pane viewer."""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _write_array(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    values.tofile(temporary)
    temporary.replace(path)


def publish_cloud(output, cloud_id, source, description, alignment=None):
    """Write aligned XYZ and RGB in one deterministic shuffled order."""
    cloud = laspy.read(source)
    xyz = np.asarray(cloud.xyz)
    if alignment is not None:
        xyz = xyz @ alignment["rotation"].T + alignment["translation"]
    if not np.isfinite(xyz).all():
        raise ValueError(f"Non-finite coordinates in {source}")
    count = len(xyz)
    order = np.arange(count, dtype=np.uint32)
    np.random.default_rng(20909).shuffle(order)

    xyz_path = output / f"{cloud_id}.xyz.bin"
    rgb_path = output / f"{cloud_id}.rgb.bin"
    shuffled_xyz = np.asarray(xyz[order], dtype="<f4")
    rgb = np.column_stack((cloud.red, cloud.green, cloud.blue))
    shuffled_rgb = np.asarray(rgb[order], dtype="<u2")
    _write_array(xyz_path, shuffled_xyz)
    _write_array(rgb_path, shuffled_rgb)
    if xyz_path.stat().st_size != count * 12 or rgb_path.stat().st_size != count * 6:
        raise RuntimeError(f"Incomplete viewer output for {cloud_id}")
    return {
        "id": cloud_id,
        "count": count,
        "url": f"/data/{xyz_path.name}",
        "bytes": xyz_path.stat().st_size,
        "bounds": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
        "source": str(source),
        "description": description,
        "sha256": _digest(xyz_path),
        "rgb": {
            "url": f"/data/{rgb_path.name}",
            "bytes": rgb_path.stat().st_size,
            "sha256": _digest(rgb_path),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer-data", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--studio", type=Path, required=True)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--keyframes20", type=Path, required=True)
    parser.add_argument("--cell1", type=Path, required=True)
    parser.add_argument("--proxy8cm", type=Path, required=True)
    args = parser.parse_args()
    args.viewer_data.mkdir(parents=True, exist_ok=True)
    alignment = np.load(args.alignment)
    specifications = [
        (
            "studio-original",
            args.studio,
            "Studio 2.6.1 original color export",
            None,
        ),
        (
            "exact-control",
            args.exact,
            "Byte-identical optimized control · 7.134 s candidate stage",
            alignment,
        ),
        (
            "experiment-keyframes20",
            args.keyframes20,
            "A · 20 coverage-selected keyframes · 3.173 s · 2.25×",
            alignment,
        ),
        (
            "experiment-cell1",
            args.cell1,
            "B · 8 cm cell analytic top-1 · 4.011 s · 1.78×",
            alignment,
        ),
        (
            "experiment-proxy8cm",
            args.proxy8cm,
            "C · 8 cm proxy visibility top-1 · 5.375 s · 1.33×",
            alignment,
        ),
    ]
    rows = [
        publish_cloud(args.viewer_data, cloud_id, source, description, transform)
        for cloud_id, source, description, transform in specifications
    ]
    manifest = {
        "recording": "S20 short indoor color-matching experiments · 12 September 2026",
        "floorZ": -0.74,
        "coordinates": (
            "Studio reference frame; exact and experimental clouds share the same fixed "
            "rigid alignment with no scale change"
        ),
        "nativeAlignment": {
            "rotation": alignment["rotation"].tolist(),
            "translation": alignment["translation"].tolist(),
        },
        "comparisonOrder": [
            "experiment-keyframes20",
            "experiment-cell1",
            "experiment-proxy8cm",
        ],
        "processingSteps": [],
        "clouds": rows,
    }
    temporary = args.viewer_data / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.viewer_data / "manifest.json")
    print(json.dumps({"clouds": [[row["id"], row["count"]] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
