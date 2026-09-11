"""Bounded point chunks with full-image visibility and deterministic view ranking."""

import json

import numpy as np
from PIL import Image

from .camera import load_camera_frames, project
from .ply import map_points, read_ply_info
from .storage import digest


def minimum_depth_keys(a, radius=3):
    """Exact integer separable minimum; never converts packed IDs to double."""
    sentinel = np.iinfo(np.int64).max
    for axis in [0, 1]:
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(a, pad, constant_values=sentinel)
        out = np.full_like(a, sentinel)
        for offset in range(2 * radius + 1):
            slices = [slice(None), slice(None)]
            slices[axis] = slice(offset, offset + a.shape[axis])
            np.minimum(out, padded[tuple(slices)], out=out)
        a = out
    return a


def collect(
    geometry,
    cameras,
    calibration,
    mask_root,
    output,
    workers=8,
    chunk=262144,
    progress=lambda *args: None,
):
    fs = load_camera_frames(cameras, calibration)
    K = 4
    GW = 8
    GH = 6
    CHUNK = chunk
    dest = output / "candidates"
    dest.mkdir(parents=True, exist_ok=False)
    p = map_points(read_ply_info(geometry))
    xyz = np.column_stack([p[k] for k in ["x", "y", "z"]])
    normals = np.column_stack([p[k] for k in ["normal_x", "normal_y", "normal_z"]])
    n = len(p)
    if n == 0 or not np.isfinite(xyz).all() or not np.isfinite(normals).all():
        raise ValueError("Empty or nonfinite geometry")
    c = np.memmap(dest / "observations.bin", dtype="float32", mode="w+", shape=(n, K, 8))
    c[:] = 0
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as executor:
        projection = np.empty((n, 4), dtype="float32")
        rows = []
        sentinel = np.iinfo(np.int64).max
        for index, f in enumerate(fs):
            cal = f.calibration
            w = (cal.width + 3) // 4
            h = (cal.height + 3) // 4
            depth_id = np.full(w * h, sentinel, dtype="int64")
            # All points contribute to the image depth buffer before ANY are colored.
            # Point chunks therefore cannot hide occluders in other chunks.
            for start in range(0, n, CHUNK):
                end = min(n, start + CHUNK)
                u, v, angle, d = project(xyz[start:end], f)
                projection[start:end] = np.column_stack([u, v, angle, d])
                valid = (
                    (d > 0.1)
                    & np.isfinite(u)
                    & np.isfinite(v)
                    & (u >= 0)
                    & (v >= 0)
                    & (u < cal.width - 1)
                    & (v < cal.height - 1)
                    & (angle < np.deg2rad(cal.max_incident_angle_deg))
                )
                ids = np.flatnonzero(valid) + start
                pix = v[valid].astype("int32") // 4 * w + u[valid].astype("int32") // 4
                quant = np.rint(d[valid] * 1e6).astype("int64")
                if len(ids):
                    if int(quant.max()) >= (sentinel - n) // n:
                        raise ValueError("Scene exceeds int64 depth encoding range")
                    np.minimum.at(depth_id, pix, quant * n + ids)
            filled = minimum_depth_keys(depth_id.reshape(h, w)).ravel()
            im = np.asarray(Image.open(f.image_path).convert("RGB"))
            mask = (
                np.asarray(
                    Image.open(mask_root / (f.name + "_mask") / (f.image_path.stem + ".png"))
                )
                if mask_root
                else np.zeros((cal.height, cal.width), dtype="uint8")
            )
            if im.shape != (cal.height, cal.width, 3) or mask.shape != (cal.height, cal.width):
                raise ValueError("Image/mask dimensions do not match calibration")

            def color_chunk(start):
                end = min(n, start + CHUNK)
                u, v, angle, d = projection[start:end].T
                valid = (
                    (d > 0.1)
                    & np.isfinite(u)
                    & np.isfinite(v)
                    & (u >= 0)
                    & (v >= 0)
                    & (u < cal.width - 1)
                    & (v < cal.height - 1)
                    & (angle < np.deg2rad(cal.max_incident_angle_deg))
                )
                ids = np.flatnonzero(valid) + start
                u = u[valid]
                v = v[valid]
                d = d[valid]
                angle = angle[valid]
                pix = v.astype("int32") // 4 * w + u.astype("int32") // 4
                blocker = (filled[pix] % n).astype("int64")
                ray = (xyz[ids] - f.center.astype("float32")) / d[:, None]
                bn = normals[blocker]
                denom = np.einsum("ij,ij->i", bn, ray)
                plane_depth = np.einsum("ij,ij->i", bn, xyz[blocker] - f.center) / np.where(
                    abs(denom) > 0.05, denom, 1
                )
                hit = f.center + ray * plane_depth[:, None]
                patch_distance = np.linalg.norm(hit - xyz[blocker], axis=1)
                reliable = (abs(denom) > 0.15) & (plane_depth > 0.1) & (patch_distance < 0.04)
                exact_depth = (depth_id[pix] // n) / 1e6
                visible = (d <= exact_depth + 0.025 + 0.005 * d) & (
                    ~reliable | (d <= plane_depth + 0.02)
                )
                incidence = np.einsum("ij,ij->i", normals[ids], -ray)
                visible &= incidence > 0.05
                rejected_chunk = int((reliable & (d > plane_depth + 0.02)).sum())
                ids = ids[visible]
                u = u[visible]
                v = v[visible]
                d = d[visible]
                angle = angle[visible]
                x = u.astype("int32")
                y = v.astype("int32")
                usable = (
                    (mask[y, x] == 0)
                    & (mask[y + 1, x] == 0)
                    & (mask[y, x + 1] == 0)
                    & (mask[y + 1, x + 1] == 0)
                )
                ids = ids[usable]
                u = u[usable]
                v = v[usable]
                x = x[usable]
                y = y[usable]
                score = (
                    np.exp(-2 * angle[usable] ** 2)
                    / (d[usable] ** 2 + 0.25)
                    * np.maximum(incidence[visible][usable], 0.05) ** 2
                )
                slot = np.argmin(c[ids, :, 7], axis=1)
                take = score > c[ids, slot, 7]
                ids = ids[take]
                slot = slot[take]
                x = x[take]
                y = y[take]
                u = u[take]
                v = v[take]
                score = score[take]
                p00 = im[y, x].astype("float32")
                p10 = im[y, x + 1].astype("float32")
                p01 = im[y + 1, x].astype("float32")
                p11 = im[y + 1, x + 1].astype("float32")
                a = (u - x)[:, None]
                b = (v - y)[:, None]
                color = (
                    (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
                )
                gradient = np.maximum(abs(p10 - p00).max(1), abs(p01 - p00).max(1))
                c[ids, slot, :3] = color
                c[ids, slot, 3] = gradient
                c[ids, slot, 4] = np.clip(u / cal.width * GW - 0.5, 0, GW - 1)
                c[ids, slot, 5] = np.clip(v / cal.height * GH - 0.5, 0, GH - 1)
                c[ids, slot, 6] = index
                c[ids, slot, 7] = score
                return len(ids), rejected_chunk

            totals = list(executor.map(color_chunk, range(0, n, CHUNK)))
            inserted = sum(t[0] for t in totals)
            rejected = sum(t[1] for t in totals)
            rows.append(
                {
                    "image": str(f.image_path),
                    "inserted": inserted,
                    "surface_patch_rejections": rejected,
                }
            )
            progress(index + 1, len(fs))
    for start in range(0, n, CHUNK):
        part = c[start : start + CHUNK]
        order = np.argsort(-part[:, :, 7], axis=1, kind="stable")
        c[start : start + CHUNK] = np.take_along_axis(part, order[:, :, None], axis=1)
    c.flush()
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "points": n,
                "candidates": K,
                "record_floats": 8,
                "images": rows,
                "chunk_points": CHUNK,
                "color_workers": workers,
                "input_sha256": digest(geometry),
                "camera_sha256": digest(cameras),
                "calibration_sha256": digest(calibration),
                "backend": "Eight disjoint point-chunk CPU color workers; serial global depth pass; full-image global occlusion buffer; exact int64 neighborhood minimum; file-backed candidates",
            },
            indent=2,
        )
    )
