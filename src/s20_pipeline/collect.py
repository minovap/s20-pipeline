"""Bounded point chunks with full-image visibility and deterministic view ranking."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

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

CPU_DEPTH_SCRATCH_BUDGET = 128 << 20
CPU_PROJECT_SCRATCH_BYTES_PER_POINT = 128


def _load_geometry_arrays(geometry):
    points = map_points(read_ply_info(geometry))
    try:
        xyz = np.column_stack([points[k] for k in ("x", "y", "z")])
        normals = np.column_stack([points[k] for k in ("normal_x", "normal_y", "normal_z")])
    finally:
        # column_stack owns its data; no field views escape this scope. Closing
        # the source mapping releases its resident pages before collection.
        if isinstance(points, np.memmap):
            points._mmap.close()
    return xyz, normals


def _empty_observations(path, count, candidates):
    shape = (count, candidates, 8)
    # A newly created, truncated regular file reads as zero without eagerly
    # dirtying every mapped page. Never reuse an existing candidate file.
    with path.open("xb") as stream:
        stream.truncate(count * candidates * 8 * np.dtype("float32").itemsize)
    return np.memmap(path, dtype="float32", mode="r+", shape=shape)


def _empty_array(path, dtype, shape):
    dtype = np.dtype(dtype)
    with path.open("xb") as stream:
        stream.truncate(int(np.prod(shape, dtype=object)) * dtype.itemsize)
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


def _pread_exact(file_descriptor, size, offset):
    parts = []
    while size:
        part = os.pread(file_descriptor, size, offset)
        if not part:
            raise OSError("Unexpected end of candidate slot index")
        parts.append(part)
        offset += len(part)
        size -= len(part)
    return b"".join(parts)


def _load_mask(frame, mask_root):
    """Load one mask while validating its source image dimensions."""
    started = perf_counter()
    with Image.open(frame.image_path) as source:
        image_size = source.size
    if mask_root:
        path = mask_root / (frame.name + "_mask") / (frame.image_path.stem + ".png")
        with Image.open(path) as source:
            mask = np.asarray(source)
    else:
        calibration = frame.calibration
        mask = np.zeros((calibration.height, calibration.width), dtype="uint8")
    return image_size, mask, perf_counter() - started


def _load_image(frame):
    started = perf_counter()
    with Image.open(frame.image_path) as source:
        image = np.asarray(source.convert("RGB"))
    return image, perf_counter() - started


def _finalize_observations(dest, observations, frames, chunk, grid, workers):
    """Sort deferred ranks, group slots by photo, and sample final winners once."""
    n, candidates = observations.shape[:2]
    scores = observations[:, :, 7]
    photo_ids = observations[:, :, 6]
    uv = observations[:, :, :2]
    sort_started = perf_counter()
    per_photo = np.zeros(len(frames), dtype=np.int64)
    final_occupied = 0
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        part = observations[start:end]
        order = np.argsort(-part[:, :, 7], axis=1, kind="stable")
        observations[start:end] = np.take_along_axis(part, order[:, :, None], axis=1)
        occupied = scores[start:end] > 0
        final_occupied += int(np.count_nonzero(occupied))
        per_photo += np.bincount(
            photo_ids[start:end][occupied].astype(np.intp), minlength=len(frames)
        )
    sort_s = perf_counter() - sort_started

    offsets = np.r_[np.int64(0), np.cumsum(per_photo)]
    if final_occupied == 0:
        flush_started = perf_counter()
        observations.flush()
        return observations, {
            "final_sort_s": sort_s,
            "photo_bucket_s": 0.0,
            "final_image_decode_s": 0.0,
            "final_pack_s": 0.0,
            "flush_s": perf_counter() - flush_started,
            "final_occupied": 0,
            "represented_photos": 0,
            "slot_index_bytes": 0,
        }
    slot_dtype = np.uint32 if n * candidates <= np.iinfo(np.uint32).max else np.uint64
    slot_path = dest / "ranking-slots.bin"
    slot_ids = _empty_array(slot_path, slot_dtype, (final_occupied,))
    cursors = offsets[:-1].copy()
    bucket_started = perf_counter()
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        occupied = scores[start:end].ravel() > 0
        if not occupied.any():
            continue
        local_slots = np.flatnonzero(occupied).astype(slot_dtype)
        local_slots += np.asarray(start * candidates, dtype=slot_dtype)
        local_photos = photo_ids[start:end].ravel()[occupied]
        order = np.argsort(local_photos, kind="stable")
        local_slots = local_slots[order]
        local_photos = local_photos[order]
        boundaries = np.r_[0, np.flatnonzero(local_photos[1:] != local_photos[:-1]) + 1]
        for begin, finish in zip(boundaries, np.r_[boundaries[1:], len(local_photos)], strict=True):
            photo = int(local_photos[begin])
            destination = int(cursors[photo])
            count = int(finish - begin)
            slot_ids[destination : destination + count] = local_slots[begin:finish]
            cursors[photo] += count
    bucket_s = perf_counter() - bucket_started
    if not np.array_equal(cursors, offsets[1:]):
        raise RuntimeError("Final candidate photo buckets are incomplete")
    slot_ids.flush()
    slot_ids._mmap.close()

    pack_started = perf_counter()
    grid_width, grid_height = grid
    pack_chunk = min(chunk, 32768)

    def pack_photo(photo):
        frame = frames[photo]
        if per_photo[photo] == 0:
            return 0.0
        image, decode_s = _load_image(frame)
        calibration = frame.calibration
        if image.shape != (calibration.height, calibration.width, 3):
            raise ValueError("Image dimensions do not match calibration")
        for begin in range(int(offsets[photo]), int(offsets[photo + 1]), pack_chunk):
            finish = min(int(offsets[photo + 1]), begin + pack_chunk)
            itemsize = np.dtype(slot_dtype).itemsize
            raw_slots = _pread_exact(slot_file, (finish - begin) * itemsize, begin * itemsize)
            flat_slots = np.frombuffer(raw_slots, dtype=slot_dtype).astype(np.intp)
            ids = flat_slots // candidates
            slots = flat_slots % candidates
            u = uv[ids, slots, 0]
            v = uv[ids, slots, 1]
            x = u.astype("int32")
            y = v.astype("int32")
            p00 = image[y, x].astype("float32")
            p10 = image[y, x + 1].astype("float32")
            p01 = image[y + 1, x].astype("float32")
            p11 = image[y + 1, x + 1].astype("float32")
            a = (u - x)[:, None]
            b = (v - y)[:, None]
            color = (1 - a) * (1 - b) * p00 + a * (1 - b) * p10 + (1 - a) * b * p01 + a * b * p11
            gradient = np.maximum(abs(p10 - p00).max(1), abs(p01 - p00).max(1))
            observations[ids, slots, :3] = color
            observations[ids, slots, 3] = gradient
            observations[ids, slots, 4] = np.clip(
                u / calibration.width * grid_width - 0.5, 0, grid_width - 1
            )
            observations[ids, slots, 5] = np.clip(
                v / calibration.height * grid_height - 0.5, 0, grid_height - 1
            )
        return decode_s

    with slot_path.open("rb", buffering=0) as slot_stream:
        slot_file = slot_stream.fileno()
        with ThreadPoolExecutor(max_workers=min(workers, 3)) as executor:
            image_decode_s = sum(executor.map(pack_photo, range(len(frames))))
    pack_s = perf_counter() - pack_started
    flush_started = perf_counter()
    observations.flush()
    flush_s = perf_counter() - flush_started
    slot_path.unlink()
    return observations, {
        "final_sort_s": sort_s,
        "photo_bucket_s": bucket_s,
        "final_image_decode_s": image_decode_s,
        "final_pack_s": pack_s,
        "flush_s": flush_s,
        "final_occupied": final_occupied,
        "represented_photos": int(np.count_nonzero(per_photo)),
        "slot_index_bytes": final_occupied * np.dtype(slot_dtype).itemsize,
    }


def _depth_worker_count(count, chunk, workers, depth_bytes):
    if count == 0:
        return 0
    # Estimate concurrent NumPy scratch as well as private depth images; this
    # is not a cap on total process RSS. The one-worker baseline is required
    # even if one image exceeds this budget.
    estimated_scratch = depth_bytes + min(count, chunk) * CPU_PROJECT_SCRATCH_BYTES_PER_POINT
    return min(
        workers,
        4,
        (count + chunk - 1) // chunk,
        max(1, CPU_DEPTH_SCRATCH_BUDGET // estimated_scratch),
    )


def _cpu_projection_depth(
    xyz,
    frame,
    selected,
    projection,
    depth_id,
    chunk,
    workers,
    executor,
    projected_chunks=None,
):
    """Project fixed chunks in parallel, then merge exact per-worker depth minima.

    When projected_chunks is supplied, retain only valid projection rows and
    their projection values. This preserves the projection batch shapes while
    avoiding a second validity scan over every selected point.
    """
    n = len(xyz)
    count = n if selected is None else len(selected)
    worker_count = _depth_worker_count(count, chunk, workers, depth_id.nbytes)
    cal = frame.calibration
    width = (cal.width + 3) // 4
    sentinel = np.iinfo(np.int64).max
    id_dtype = "uint32" if n <= np.iinfo(np.uint32).max else "uint64"

    def project_stripe(worker):
        local_depth = depth_id if worker == 0 else np.full_like(depth_id, sentinel)
        for start in range(worker * chunk, count, worker_count * chunk):
            end = min(count, start + chunk)
            point_chunk = slice(start, end) if selected is None else selected[start:end]
            u, v, angle, d = project(xyz[point_chunk], frame)
            if projection is not None:
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
            ids = np.flatnonzero(valid) + start if selected is None else selected[start:end][valid]
            pix = v[valid].astype("int32") // 4 * width + u[valid].astype("int32") // 4
            if projected_chunks is not None:
                projected_chunks[start // chunk] = (
                    np.ascontiguousarray(ids, dtype=id_dtype),
                    u[valid],
                    v[valid],
                    angle[valid],
                    d[valid],
                )
            quant = np.rint(d[valid] * 1e6).astype("int64")
            if len(ids):
                if int(quant.max()) >= (sentinel - n) // n:
                    raise ValueError("Scene exceeds int64 depth encoding range")
                np.minimum.at(local_depth, pix, quant * n + ids)
        return local_depth

    if worker_count == 1:
        project_stripe(0)
    elif worker_count > 1:
        # At most four tasks/maps are live. All workers finish before merging
        # into worker zero's map or beginning any visibility/color work.
        depths = list(executor.map(project_stripe, range(worker_count)))
        for local_depth in depths[1:]:
            np.minimum(depth_id, local_depth, out=depth_id)
    return worker_count


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
    blocker_xyz = xyz[blocker]
    denom = np.einsum("ij,ij->i", bn, ray)
    if precision == "float32":
        center = center32
        plane_depth = np.einsum("ij,ij->i", bn, blocker_xyz - center) / np.where(
            abs(denom) > np.float32(0.05), denom, np.float32(1)
        )
        hit = center + ray * plane_depth[:, None]
        patch_distance = np.linalg.norm(hit - blocker_xyz, axis=1)
        exact_depth = (exact_keys // point_count).astype("float32") * np.float32(1e-6)
    elif precision == "mixed":
        center = frame.center
        plane_depth = np.einsum("ij,ij->i", bn, blocker_xyz - center) / np.where(
            abs(denom) > 0.05, denom, 1
        )
        hit = center + ray * plane_depth[:, None]
        patch_distance = np.linalg.norm(hit - blocker_xyz, axis=1)
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
    profile=False,
):
    if backend not in ("cpu", "cpu_float32", "metal"):
        raise ValueError(f"Unknown collector backend: {backend}")
    collect_started = perf_counter()
    phase_started = collect_started
    fs = load_camera_frames(cameras, calibration)
    setup_times = {"camera_load_s": perf_counter() - phase_started}
    K = 4
    GW = 8
    GH = 6
    CHUNK = chunk
    dest = output / "candidates"
    dest.mkdir(parents=True, exist_ok=False)
    phase_started = perf_counter()
    xyz, normals = _load_geometry_arrays(geometry)
    setup_times["geometry_load_s"] = perf_counter() - phase_started
    n = len(xyz)
    if n == 0 or not np.isfinite(xyz).all() or not np.isfinite(normals).all():
        raise ValueError("Empty or nonfinite geometry")
    phase_started = perf_counter()
    point_index = PhotoPointIndex(xyz, chunk=CHUNK)
    setup_times["point_index_s"] = perf_counter() - phase_started
    metal = None
    if backend == "metal":
        from .metal_collect import MetalCollector

        if metal_library is None or metal_kernels is None:
            raise ValueError("Metal collector requires its library and kernel source")
        metal = MetalCollector(xyz, normals, metal_library, metal_kernels, CHUNK)
        all_point_ids = np.arange(n, dtype="uint32")
    else:
        all_point_ids = None
    phase_started = perf_counter()
    c = _empty_observations(dest / "observations.bin", n, K)
    scores = c[:, :, 7]
    photo_ids = c[:, :, 6]
    uv = c[:, :, :2]
    setup_times["observation_init_s"] = perf_counter() - phase_started
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        ThreadPoolExecutor(max_workers=1) as image_executor,
    ):
        rows = []
        diagnostic_rows = []
        sentinel = np.iinfo(np.int64).max
        for index, f in enumerate(fs):
            photo_started = perf_counter()
            # Decode only while this photo's projection/depth work runs. Eager
            # look-ahead competes with the preceding photo's memory-heavy color
            # pass and was slower on the frozen scan.
            image_future = image_executor.submit(_load_mask, f, mask_root)
            index_started = photo_started
            selected = point_index.point_ids(f)
            index_s = perf_counter() - index_started
            count = n if selected is None else len(selected)
            projection = None
            cal = f.calibration
            w = (cal.width + 3) // 4
            h = (cal.height + 3) // 4
            depth_id = None
            filled = None
            # All potentially visible voxels contribute before ANY points are colored.
            # Point chunks therefore cannot hide occluders in other chunks.
            selected_ids = None
            metal_result = None
            projected_chunks = None
            depth_workers = 0
            project_started = perf_counter()
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
                depth_id = np.full(w * h, sentinel, dtype="int64")
                projected_chunks = [None] * ((count + CHUNK - 1) // CHUNK)
                depth_workers = _cpu_projection_depth(
                    xyz,
                    f,
                    selected,
                    None,
                    depth_id,
                    CHUNK,
                    workers,
                    executor,
                    projected_chunks,
                )
                projection_depth_s = perf_counter() - project_started
                minimum_started = perf_counter()
                filled = minimum_depth_keys(depth_id.reshape(h, w)).ravel()
                minimum_depth_s = perf_counter() - minimum_started
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
            if metal is not None:
                projection_depth_s = perf_counter() - project_started
                minimum_depth_s = 0.0
            image_wait_started = perf_counter()
            image_size, mask, image_decode_s = image_future.result()
            image_wait_s = perf_counter() - image_wait_started
            if image_size != (cal.width, cal.height) or mask.shape != (cal.height, cal.width):
                raise ValueError("Image/mask dimensions do not match calibration")

            def color_chunk(start):
                chunk_started = perf_counter()
                if metal_result is not None:
                    end = min(count, start + CHUNK)
                    u, v, angle, d = projection[start:end].T
                    valid = (metal_result.flags[start:end] & VALID) != 0
                    ids = (
                        np.flatnonzero(valid) + start
                        if selected is None
                        else selected[start:end][valid]
                    )
                    positions = np.flatnonzero(valid) + start
                    valid_count = int(np.count_nonzero(valid)) if profile else 0
                    u = u[valid]
                    v = v[valid]
                    d = d[valid]
                    angle = angle[valid]
                    pix = v.astype("int32") // 4 * w + u.astype("int32") // 4
                else:
                    projected = projected_chunks[start]
                    projected_chunks[start] = None
                    ids, u, v, angle, d = projected
                    positions = None
                    valid_count = len(ids) if profile else 0
                    pix = v.astype("int32") // 4 * w + u.astype("int32") // 4
                decision_started = perf_counter()
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
                decision_s = perf_counter() - decision_started if profile else 0.0
                if profile:
                    exact_depth = (
                        exact_keys.astype(np.int64) // n
                        if metal_result is None
                        else metal_result.exact_keys[positions].astype(np.int64) // n
                    )
                    passes_exact_depth = int(
                        np.count_nonzero(d <= exact_depth / 1e6 + 0.025 + 0.005 * d)
                    )
                else:
                    passes_exact_depth = 0
                rejected_chunk = int(surface_rejected.sum())
                visible_count = int(np.count_nonzero(visible)) if profile else 0
                ids = ids[visible]
                u = u[visible]
                v = v[visible]
                d = d[visible]
                angle = angle[visible]
                x = u.astype("int32")
                y = v.astype("int32")
                mask_started = perf_counter()
                usable = (
                    (mask[y, x] == 0)
                    & (mask[y + 1, x] == 0)
                    & (mask[y, x + 1] == 0)
                    & (mask[y + 1, x + 1] == 0)
                )
                mask_s = perf_counter() - mask_started if profile else 0.0
                usable_count = int(np.count_nonzero(usable)) if profile else 0
                ids = ids[usable]
                u = u[usable]
                v = v[usable]
                if metal_result is not None:
                    ray = (xyz[ids] - f.center.astype("float32")) / d[usable, None]
                    incidence_for_score = np.einsum("ij,ij->i", normals[ids], -ray)
                else:
                    incidence_for_score = incidence[visible][usable]
                rank_started = perf_counter()
                score = (
                    np.exp(-2 * angle[usable] ** 2)
                    / (d[usable] ** 2 + 0.25)
                    * np.maximum(incidence_for_score, 0.05) ** 2
                )
                score_count = len(score) if profile else 0
                slot = np.argmin(scores[ids], axis=1)
                take = score > scores[ids, slot]
                ids = ids[take]
                slot = slot[take]
                u = u[take]
                v = v[take]
                score = score[take]
                rank_s = perf_counter() - rank_started if profile else 0.0
                pack_started = perf_counter()
                scores[ids, slot] = score
                photo_ids[ids, slot] = index
                uv[ids, slot, 0] = u
                uv[ids, slot, 1] = v
                pack_s = perf_counter() - pack_started if profile else 0.0
                return {
                    "inserted": len(ids),
                    "surface_patch_rejections": rejected_chunk,
                    "valid": valid_count,
                    "passes_exact_depth": passes_exact_depth,
                    "visible": visible_count,
                    "mask_usable": usable_count,
                    "score_evaluated": score_count,
                    "worker_s": perf_counter() - chunk_started if profile else 0.0,
                    "decision_s": decision_s,
                    "mask_s": mask_s,
                    "rank_s": rank_s,
                    "pack_s": pack_s,
                }

            valid_projection_pairs = (
                0 if projected_chunks is None else sum(len(part[0]) for part in projected_chunks)
            )
            projection_intermediate_bytes = (
                0
                if projected_chunks is None
                else sum(array.nbytes for part in projected_chunks for array in part)
            )
            color_started = perf_counter()
            color_work = (
                range(0, count, CHUNK)
                if metal_result is not None
                else range(len(projected_chunks))
            )
            totals = list(executor.map(color_chunk, color_work))
            color_wall_s = perf_counter() - color_started
            # The last photo's compact projection would otherwise remain live
            # through final sorting and image materialization.
            projected_chunks = None
            inserted = sum(t["inserted"] for t in totals)
            rejected = sum(t["surface_patch_rejections"] for t in totals)
            rows.append(
                {
                    "image": str(f.image_path),
                    "projected_points": count,
                    "valid_projection_pairs": valid_projection_pairs,
                    "projection_intermediate_bytes": projection_intermediate_bytes,
                    "cpu_depth_workers": depth_workers,
                    "cpu_depth_buffer_bytes": max(1, depth_workers) * depth_id.nbytes
                    if depth_id is not None
                    else 0,
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
            if profile:
                selection = getattr(point_index, "last_selection", None) or {}
                selected_bytes = 0 if selected is None else selected.nbytes
                photo_storage_bytes = (
                    projection.nbytes if projection is not None else projection_intermediate_bytes
                ) + selected_bytes
                if depth_id is not None:
                    photo_storage_bytes += depth_id.nbytes + filled.nbytes
                elif metal_result is not None:
                    photo_storage_bytes += sum(
                        value.nbytes
                        for value in (
                            metal_result.depth_keys,
                            metal_result.exact_keys,
                            metal_result.blocker_keys,
                            metal_result.flags,
                        )
                    )
                rows[-1]["profile"] = {
                    "photo_wall_s": perf_counter() - photo_started,
                    "point_index_s": index_s,
                    "selection_path": selection.get("path"),
                    "occupied_voxels": selection.get("occupied_voxels", 0),
                    "retained_voxels": selection.get("retained_voxels", 0),
                    "projection_depth_s": projection_depth_s,
                    "minimum_depth_s": minimum_depth_s,
                    "occupied_depth_pixels": int(np.count_nonzero(depth_id != sentinel))
                    if depth_id is not None
                    else 0,
                    "image_mask_decode_s": image_decode_s,
                    "image_mask_wait_s": image_wait_s,
                    "color_wall_s": color_wall_s,
                    "color_worker_s": sum(t["worker_s"] for t in totals),
                    "valid": sum(t["valid"] for t in totals),
                    "passes_exact_depth": sum(t["passes_exact_depth"] for t in totals),
                    "visible": sum(t["visible"] for t in totals),
                    "mask_usable": sum(t["mask_usable"] for t in totals),
                    "score_evaluated": sum(t["score_evaluated"] for t in totals),
                    "decision_worker_s": sum(t["decision_s"] for t in totals),
                    "mask_worker_s": sum(t["mask_s"] for t in totals),
                    "rank_worker_s": sum(t["rank_s"] for t in totals),
                    "pack_worker_s": sum(t["pack_s"] for t in totals),
                    "photo_storage_bytes": photo_storage_bytes,
                }
            with (dest / "photo-progress.jsonl").open("a") as stream:
                stream.write(json.dumps(rows[-1]) + "\n")
            progress(index + 1, len(fs))
    metal_stats = metal.stats() if metal is not None else None
    if metal is not None:
        metal.close()
    voxel_order = getattr(point_index, "order", None)
    voxel_order_bytes = voxel_order.nbytes if voxel_order is not None else 0
    persistent_bytes = {
        "xyz": xyz.nbytes,
        "normals": normals.nbytes,
        "voxel_order": voxel_order_bytes,
        "voxel_starts": getattr(point_index, "starts", np.empty(0)).nbytes,
        "voxel_counts": getattr(point_index, "counts", np.empty(0)).nbytes,
        "voxel_centers": getattr(point_index, "centers", np.empty(0)).nbytes,
        "observations": c.nbytes,
    }
    voxel_order = None
    color_chunk = None
    projection = projected_chunks = selected = selected_ids = depth_id = filled = None
    mask = metal_result = None
    totals = image_future = None
    xyz = normals = all_point_ids = None
    del point_index, metal
    c, finalize = _finalize_observations(dest, c, fs, CHUNK, (GW, GH), workers)
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
                "profile": {
                    **setup_times,
                    **finalize,
                    "persistent_bytes": persistent_bytes,
                    "max_photo_storage_bytes": max(
                        (row.get("profile", {}).get("photo_storage_bytes", 0) for row in rows),
                        default=0,
                    ),
                    "collect_s": perf_counter() - collect_started,
                }
                if profile
                else None,
                "input_sha256": digest(geometry),
                "camera_sha256": digest(cameras),
                "calibration_sha256": digest(calibration),
                "backend": "Conservative 2 m fisheye voxel culling; persistent Metal projection/depth/visibility when selected; disjoint point-chunk CPU masking/ranking; final-only bounded photo sampling; deterministic packed uint64 depth keys; file-backed candidates",
            },
            indent=2,
        )
    )
