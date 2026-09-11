"""Bounded point chunks with full-image visibility and deterministic view ranking."""

import json

import numpy as np
from PIL import Image

from .camera import load_camera_frames, project
from .frustum import PhotoPointIndex
from .ply import map_points, read_ply_info
from .storage import digest

VALID = np.uint32(1)
RELIABLE = np.uint32(2)
SURFACE_REJECTED = np.uint32(4)
VISIBLE = np.uint32(8)


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


def visibility_decisions(xyz, normals, ids, d, blocker, exact_keys, point_count, frame, precision):
    """Return visibility intermediates using the requested CPU precision contract."""
    center32 = frame.center.astype("float32")
    ray = (xyz[ids] - center32) / d[:, None]
    bn = normals[blocker]
    denom = np.einsum("ij,ij->i", bn, ray)
    if precision == "float32":
        center = center32
        plane_depth = np.einsum("ij,ij->i", bn, xyz[blocker] - center) / np.where(
            abs(denom) > np.float32(0.05), denom, np.float32(1)
        )
        hit = center + ray * plane_depth[:, None]
        patch_distance = np.linalg.norm(hit - xyz[blocker], axis=1)
        exact_depth = (exact_keys // point_count).astype("float32") * np.float32(1e-6)
    elif precision == "mixed":
        center = frame.center
        plane_depth = np.einsum("ij,ij->i", bn, xyz[blocker] - center) / np.where(
            abs(denom) > 0.05, denom, 1
        )
        hit = center + ray * plane_depth[:, None]
        patch_distance = np.linalg.norm(hit - xyz[blocker], axis=1)
        exact_depth = (exact_keys // point_count) / 1e6
    else:
        raise ValueError(f"Unknown visibility precision: {precision}")
    reliable = (abs(denom) > 0.15) & (plane_depth > 0.1) & (patch_distance < 0.04)
    surface_rejected = reliable & (d > plane_depth + 0.02)
    visible = (d <= exact_depth + 0.025 + 0.005 * d) & (~reliable | ~surface_rejected)
    incidence = np.einsum("ij,ij->i", normals[ids], -ray)
    visible &= incidence > 0.05
    return visible, reliable, surface_rejected, incidence


def cpu_photo_decisions(xyz, normals, frame, selected, chunk, precision="mixed"):
    """Materialize CPU decisions for diagnostics and float32-reference validation."""
    n = len(xyz)
    selected_ids = (
        np.arange(n, dtype="uint32") if selected is None else np.asarray(selected, dtype="uint32")
    )
    count = len(selected_ids)
    projection = np.empty((count, 4), dtype="float32")
    flags = np.zeros(count, dtype="uint32")
    depth_keys = np.full(count, np.iinfo(np.int64).max, dtype="uint64")
    cal = frame.calibration
    width = (cal.width + 3) // 4
    height = (cal.height + 3) // 4
    depth_id = np.full(width * height, np.iinfo(np.int64).max, dtype="int64")
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        ids = selected_ids[start:end]
        u, v, angle, d = project(xyz[ids], frame)
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
        positions = np.flatnonzero(valid) + start
        quantized = np.rint(d[valid] * 1e6).astype("int64")
        keys = quantized * n + ids[valid]
        depth_keys[positions] = keys.astype("uint64")
        pixels = v[valid].astype("int32") // 4 * width + u[valid].astype("int32") // 4
        np.minimum.at(depth_id, pixels, keys)
        flags[positions] |= VALID
    filled = minimum_depth_keys(depth_id.reshape(height, width)).ravel()
    exact_keys = np.full(count, np.iinfo(np.int64).max, dtype="uint64")
    blocker_keys = np.full(count, np.iinfo(np.int64).max, dtype="uint64")
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        valid = (flags[start:end] & VALID) != 0
        if not valid.any():
            continue
        positions = np.flatnonzero(valid) + start
        u, v, _, d = projection[positions].T
        pixels = v.astype("int32") // 4 * width + u.astype("int32") // 4
        exact = depth_id[pixels]
        blockers = filled[pixels]
        exact_keys[positions] = exact.astype("uint64")
        blocker_keys[positions] = blockers.astype("uint64")
        ids = selected_ids[positions]
        visible, reliable, rejected, _ = visibility_decisions(
            xyz, normals, ids, d, blockers % n, exact, n, frame, precision
        )
        flags[positions[reliable]] |= RELIABLE
        flags[positions[rejected]] |= SURFACE_REJECTED
        flags[positions[visible]] |= VISIBLE
    return projection, depth_keys, exact_keys, blocker_keys, flags


def diagnostic_difference(reference, candidate, selected, photo):
    """Summarize the first parity failure without changing production records."""
    names = ("projection", "depth_keys", "exact_keys", "blocker_keys", "flags")
    result = {"image": str(photo), "differences": {}}
    for name, expected, actual in zip(names, reference, candidate, strict=True):
        if name == "projection":
            expected_finite = np.isfinite(expected)
            actual_finite = np.isfinite(actual)
            finite = expected_finite & actual_finite
            delta = np.abs(expected[finite] - actual[finite])
            result["projection_max_abs"] = float(delta.max(initial=0))
            result["projection_finiteness_differences"] = int(
                np.count_nonzero(expected_finite != actual_finite)
            )
            continue
        unequal = np.flatnonzero(expected != actual)
        if len(unequal):
            position = int(unequal[0])
            result["differences"][name] = {
                "count": int(len(unequal)),
                "selected_position": position,
                "point_id": int(selected[position]),
                "reference": int(expected[position]),
                "candidate": int(actual[position]),
            }
    return result


def collect(
    geometry,
    cameras,
    calibration,
    mask_root,
    output,
    workers=8,
    chunk=262144,
    progress=lambda *args: None,
    backend="cpu",
    metal_library=None,
    metal_kernels=None,
    diagnostics=False,
):
    if backend not in ("cpu", "cpu_float32", "metal"):
        raise ValueError(f"Unknown collector backend: {backend}")
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
    point_index = PhotoPointIndex(xyz, chunk=CHUNK)
    metal = None
    if backend == "metal":
        from .metal_collect import MetalCollector

        if metal_library is None or metal_kernels is None:
            raise ValueError("Metal collector requires its library and kernel source")
        metal = MetalCollector(xyz, normals, metal_library, metal_kernels, CHUNK)
        all_point_ids = np.arange(n, dtype="uint32")
    else:
        all_point_ids = None
    c = np.memmap(dest / "observations.bin", dtype="float32", mode="w+", shape=(n, K, 8))
    c[:] = 0
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = []
        diagnostic_rows = []
        sentinel = np.iinfo(np.int64).max
        for index, f in enumerate(fs):
            selected = point_index.point_ids(f)
            count = n if selected is None else len(selected)
            projection = None
            cal = f.calibration
            w = (cal.width + 3) // 4
            h = (cal.height + 3) // 4
            depth_id = np.full(w * h, sentinel, dtype="int64")
            # All potentially visible voxels contribute before ANY points are colored.
            # Point chunks therefore cannot hide occluders in other chunks.
            selected_ids = None
            metal_result = None
            if metal is not None:
                selected_ids = (
                    all_point_ids if selected is None else np.asarray(selected, dtype="uint32")
                )
                metal_result = metal.process(selected_ids, f)
                projection = metal_result.projection
                if diagnostics:
                    reference = cpu_photo_decisions(xyz, normals, f, selected, CHUNK, "mixed")
                    candidate = (
                        metal_result.projection,
                        metal_result.depth_keys,
                        metal_result.exact_keys,
                        metal_result.blocker_keys,
                        metal_result.flags,
                    )
                    diagnostic_rows.append(
                        diagnostic_difference(reference, candidate, selected_ids, f.image_path)
                    )
            else:
                projection = np.empty((count, 4), dtype="float32")
                for start in range(0, count, CHUNK):
                    end = min(count, start + CHUNK)
                    point_chunk = slice(start, end) if selected is None else selected[start:end]
                    u, v, angle, d = project(xyz[point_chunk], f)
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
                    ids = (
                        np.flatnonzero(valid) + start
                        if selected is None
                        else selected[start:end][valid]
                    )
                    pix = v[valid].astype("int32") // 4 * w + u[valid].astype("int32") // 4
                    quant = np.rint(d[valid] * 1e6).astype("int64")
                    if len(ids):
                        if int(quant.max()) >= (sentinel - n) // n:
                            raise ValueError("Scene exceeds int64 depth encoding range")
                        np.minimum.at(depth_id, pix, quant * n + ids)
                filled = minimum_depth_keys(depth_id.reshape(h, w)).ravel()
                if diagnostics and backend == "cpu_float32":
                    selected_ids = (
                        np.arange(n, dtype="uint32")
                        if selected is None
                        else np.asarray(selected, dtype="uint32")
                    )
                    reference = cpu_photo_decisions(xyz, normals, f, selected, CHUNK, "mixed")
                    candidate = cpu_photo_decisions(xyz, normals, f, selected, CHUNK, "float32")
                    diagnostic_rows.append(
                        diagnostic_difference(reference, candidate, selected_ids, f.image_path)
                    )
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
                end = min(count, start + CHUNK)
                u, v, angle, d = projection[start:end].T
                valid = (
                    (metal_result.flags[start:end] & VALID) != 0
                    if metal_result is not None
                    else (
                        (d > 0.1)
                        & np.isfinite(u)
                        & np.isfinite(v)
                        & (u >= 0)
                        & (v >= 0)
                        & (u < cal.width - 1)
                        & (v < cal.height - 1)
                        & (angle < np.deg2rad(cal.max_incident_angle_deg))
                    )
                )
                ids = (
                    np.flatnonzero(valid) + start
                    if selected is None
                    else selected[start:end][valid]
                )
                positions = np.flatnonzero(valid) + start
                u = u[valid]
                v = v[valid]
                d = d[valid]
                angle = angle[valid]
                pix = v.astype("int32") // 4 * w + u.astype("int32") // 4
                if metal_result is None:
                    blocker_keys = filled[pix]
                    exact_keys = depth_id[pix]
                    visible, reliable, surface_rejected, incidence = visibility_decisions(
                        xyz,
                        normals,
                        ids,
                        d,
                        blocker_keys % n,
                        exact_keys,
                        n,
                        f,
                        "float32" if backend == "cpu_float32" else "mixed",
                    )
                else:
                    point_flags = metal_result.flags[positions]
                    visible = (point_flags & VISIBLE) != 0
                    surface_rejected = (point_flags & SURFACE_REJECTED) != 0
                rejected_chunk = int(surface_rejected.sum())
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
                if metal_result is not None:
                    ray = (xyz[ids] - f.center.astype("float32")) / d[usable, None]
                    incidence_for_score = np.einsum("ij,ij->i", normals[ids], -ray)
                else:
                    incidence_for_score = incidence[visible][usable]
                score = (
                    np.exp(-2 * angle[usable] ** 2)
                    / (d[usable] ** 2 + 0.25)
                    * np.maximum(incidence_for_score, 0.05) ** 2
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

            totals = list(executor.map(color_chunk, range(0, count, CHUNK)))
            inserted = sum(t[0] for t in totals)
            rejected = sum(t[1] for t in totals)
            rows.append(
                {
                    "image": str(f.image_path),
                    "projected_points": count,
                    "inserted": inserted,
                    "surface_patch_rejections": rejected,
                    "projection_cpu_rechecks": metal_result.projection_rechecks
                    if metal_result is not None
                    else 0,
                    "visibility_cpu_rechecks": metal_result.visibility_rechecks
                    if metal_result is not None
                    else 0,
                    "exact_depth_cpu_rechecks": metal_result.exact_depth_rechecks
                    if metal_result is not None
                    else 0,
                }
            )
            progress(index + 1, len(fs))
    for start in range(0, n, CHUNK):
        part = c[start : start + CHUNK]
        order = np.argsort(-part[:, :, 7], axis=1, kind="stable")
        c[start : start + CHUNK] = np.take_along_axis(part, order[:, :, None], axis=1)
    c.flush()
    metal_stats = metal.stats() if metal is not None else None
    if metal is not None:
        metal.close()
    if diagnostics:
        (dest / "collector-diagnostics.json").write_text(
            json.dumps({"backend": backend, "images": diagnostic_rows}, indent=2)
        )
    (dest / "meta.json").write_text(
        json.dumps(
            {
                "points": n,
                "candidates": K,
                "record_floats": 8,
                "images": rows,
                "chunk_points": CHUNK,
                "color_workers": workers,
                "collector": backend,
                "metal": metal_stats,
                "input_sha256": digest(geometry),
                "camera_sha256": digest(cameras),
                "calibration_sha256": digest(calibration),
                "backend": "Conservative 2 m fisheye voxel culling; persistent Metal projection/depth/visibility when selected; disjoint point-chunk CPU masking/color/ranking; deterministic packed uint64 depth keys; file-backed candidates",
            },
            indent=2,
        )
    )
