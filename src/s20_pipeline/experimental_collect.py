"""Opt-in approximate photo-to-point experiments.

These collectors deliberately trade exact candidate parity for speed.  They
are kept out of the production CLI until their visual and numerical quality is
understood on more than the frozen indoor scan.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .camera import load_camera_frames
from .collect import (
    _cpu_projection_depth,
    _empty_observations,
    _finalize_observations,
    _load_geometry_arrays,
    _load_mask,
    minimum_depth_keys,
)
from .cpu_visibility import CpuVisibility
from .storage import digest


@dataclass(frozen=True)
class CellIndex:
    """One actual source-point representative per spatial/normal cell."""

    size: float
    normal_split: bool
    representatives: np.ndarray
    inverse: np.ndarray
    counts: np.ndarray


def build_cell_index(xyz, normals, size, normal_split):
    """Return deterministic cells while retaining original source-point IDs."""
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Experimental cell size must be positive and finite")
    xyz = np.asarray(xyz)
    normals = np.asarray(normals)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or normals.shape != xyz.shape:
        raise ValueError("Experimental cells require matching n x 3 geometry and normals")
    lower = np.floor(xyz.min(axis=0).astype(np.float64) / size).astype(np.int64)
    cells = np.floor(xyz.astype(np.float64) / size).astype(np.int64) - lower
    shape = cells.max(axis=0).astype(object) + 1
    if np.prod(shape, dtype=object) > np.iinfo(np.int64).max // (8 if normal_split else 1):
        raise ValueError("Experimental cell coordinates exceed int64 key range")
    keys = (cells[:, 0] * int(shape[1]) + cells[:, 1]) * int(shape[2]) + cells[:, 2]
    if normal_split:
        octants = (
            (normals[:, 0] >= 0).astype(np.int64)
            | ((normals[:, 1] >= 0).astype(np.int64) << 1)
            | ((normals[:, 2] >= 0).astype(np.int64) << 2)
        )
        keys = keys * 8 + octants
    _, representatives, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    inverse_dtype = np.uint32 if len(representatives) <= np.iinfo(np.uint32).max else np.uint64
    representative_dtype = np.uint32 if len(xyz) <= np.iinfo(np.uint32).max else np.uint64
    return CellIndex(
        size,
        normal_split,
        representatives.astype(representative_dtype),
        inverse.astype(inverse_dtype),
        counts.astype(np.uint32),
    )


def _geometric_quality(xyz, normals, cells, frames, native):
    """Cheap valid-view quality for cell representatives; no occlusion or masks."""
    quality = np.zeros((len(cells.representatives), len(frames)), dtype=np.float32)
    points = xyz[cells.representatives]
    for photo, frame in enumerate(frames):
        projection = native.project_valid(points, frame)
        if not len(projection.positions):
            continue
        ids = cells.representatives[projection.positions]
        ray = (xyz[ids] - frame.center.astype(np.float32)) / projection.distance[:, None]
        incidence = np.einsum("ij,ij->i", normals[ids], -ray)
        usable = incidence > np.float32(0.05)
        score = (
            np.exp(np.float32(-2) * projection.angle[usable] ** 2)
            / (projection.distance[usable] ** 2 + np.float32(0.25))
            * np.maximum(incidence[usable], np.float32(0.05)) ** 2
        )
        quality[projection.positions[usable], photo] = score
    return quality


def select_keyframes(xyz, normals, frames, native, count=20):
    """Greedily cover surface cells with strong first and second viewpoints."""
    if count <= 0 or count > len(frames):
        raise ValueError("Experimental keyframe count must fit the camera table")
    cells = build_cell_index(xyz, normals, 0.20, True)
    quality = _geometric_quality(xyz, normals, cells, frames, native)
    row_max = quality.max(axis=1)
    np.divide(quality, row_max[:, None], out=quality, where=row_max[:, None] > 0)
    weights = np.sqrt(np.minimum(cells.counts, np.uint32(64))).astype(np.float32)
    best = np.zeros(len(cells.representatives), dtype=np.float32)
    second = np.zeros_like(best)
    chosen = []
    names = sorted({frame.name for frame in frames})
    quotas = {name: count // len(names) for name in names}
    for name in names[: count % len(names)]:
        quotas[name] += 1
    used = {name: 0 for name in names}
    for step in range(count):
        available_names = [name for name in names if used[name] < quotas[name]]
        preferred = available_names[step % len(available_names)]
        candidates = [
            index
            for index, frame in enumerate(frames)
            if frame.name == preferred and index not in chosen
        ]
        winner = None
        winner_value = -np.inf
        for photo in candidates:
            candidate = quality[:, photo]
            new_best = np.maximum(best, candidate)
            new_second = np.where(candidate >= best, best, np.maximum(second, candidate))
            value = float(np.sum(weights * (new_best + np.float32(0.35) * new_second)))
            if value > winner_value:
                winner = photo
                winner_value = value
        if winner is None:
            raise RuntimeError("Could not satisfy experimental keyframe camera balance")
        candidate = quality[:, winner]
        second = np.where(candidate >= best, best, np.maximum(second, candidate))
        best = np.maximum(best, candidate)
        chosen.append(winner)
        used[frames[winner].name] += 1
    covered = best > 0
    return np.asarray(sorted(chosen), dtype=np.uint32), {
        "cell_size_m": cells.size,
        "normal_split": cells.normal_split,
        "cells": len(cells.representatives),
        "selected": sorted(chosen),
        "camera_counts": used,
        "covered_cells": int(np.count_nonzero(covered)),
        "coverage_fraction": float(np.mean(covered)),
        "mean_best_normalized_quality": float(np.average(best, weights=weights)),
        "mean_second_normalized_quality": float(np.average(second, weights=weights)),
    }


def build_photo_shortlist(xyz, normals, frames, native, size=0.08, count=4):
    """Rank cameras analytically for normal-split surface cells."""
    cells = build_cell_index(xyz, normals, size, True)
    quality = _geometric_quality(xyz, normals, cells, frames, native)
    count = min(count, len(frames))
    partition = np.argpartition(quality, -count, axis=1)[:, -count:]
    scores = np.take_along_axis(quality, partition, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    photos = np.take_along_axis(partition, order, axis=1).astype(np.int16)
    scores = np.take_along_axis(scores, order, axis=1)
    photos[scores <= 0] = -1
    return cells, photos, {
        "cell_size_m": size,
        "normal_split": True,
        "cells": len(cells.representatives),
        "photos_per_cell": count,
        "mean_valid_shortlist": float(np.mean(np.count_nonzero(photos >= 0, axis=1))),
    }


def _insert_top4(scores, photos, positions, values, photo):
    """Chronological strict insertion matching the four-slot production policy."""
    if not len(positions):
        return 0
    slots = np.argmin(scores[positions], axis=1)
    take = values > scores[positions, slots]
    positions = positions[take]
    slots = slots[take]
    scores[positions, slots] = values[take]
    photos[positions, slots] = photo
    return int(np.count_nonzero(take))


def _sort_top4(scores, photos):
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(scores, order, axis=1), np.take_along_axis(photos, order, axis=1)


def _photo_targets(cell_index, cell_photos, photo):
    keep_cells = np.any(cell_photos == photo, axis=1)
    return np.flatnonzero(keep_cells[cell_index.inverse]).astype(np.uint32)


def _bucket_photo_targets(cell_index, cell_photos, photo_count, slots=1):
    """Expand assignments once and return stable original-ID buckets per photo."""
    assigned = cell_photos[cell_index.inverse, :slots].reshape(-1)
    point_ids = np.repeat(np.arange(len(cell_index.inverse), dtype=np.uint32), slots)
    valid = assigned >= 0
    assigned = assigned[valid]
    point_ids = point_ids[valid]
    order = np.argsort(assigned, kind="stable")
    point_ids = point_ids[order]
    counts = np.bincount(assigned, minlength=photo_count)
    offsets = np.empty(photo_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return [point_ids[offsets[photo] : offsets[photo + 1]] for photo in range(photo_count)]


def _project_targets(xyz, frame, target_ids, native, chunk, executor):
    count = len(xyz) if target_ids is None else len(target_ids)

    def project_chunk(start):
        end = min(count, start + chunk)
        point_chunk = slice(start, end) if target_ids is None else target_ids[start:end]
        result = native.project_valid(xyz[point_chunk], frame)
        ids = (
            result.positions + start
            if target_ids is None
            else target_ids[start:end][result.positions]
        )
        return (
            np.ascontiguousarray(ids, dtype=np.uint32),
            result.u,
            result.v,
            result.angle,
            result.distance,
        )

    return list(executor.map(project_chunk, range(0, count, chunk)))


def _insert_observations(native, observations, ids, u, v, score, photo, candidate_slots):
    if candidate_slots == 4:
        return native.insert(observations, ids, u, v, score, photo)
    current = observations[ids, 0, 7]
    take = score > current
    ids = ids[take]
    observations[ids, 0, 0] = u[take]
    observations[ids, 0, 1] = v[take]
    observations[ids, 0, 6] = photo
    observations[ids, 0, 7] = score[take]
    return int(np.count_nonzero(take))


def _process_photo(
    xyz,
    normals,
    frames,
    photo,
    target_ids,
    depth_proxy,
    observations,
    native,
    mask_root,
    workers,
    chunk,
    executor,
    destination_ids=None,
    rescue_observations=None,
    cached_depth=None,
    visibility=True,
    candidate_slots=4,
):
    """Build approximate depth, then exactly reproject and test dense targets."""
    frame = frames[photo]
    cal = frame.calibration
    width = (cal.width + 3) // 4
    height = (cal.height + 3) // 4
    sentinel = np.iinfo(np.int64).max
    depth_id = None
    if visibility:
        depth_id = (
            np.full(width * height, sentinel, dtype=np.int64)
            if cached_depth is None
            else np.array(cached_depth, dtype=np.int64, copy=True)
        )
    target_count = len(xyz) if target_ids is None else len(target_ids)
    projected_pairs = 0
    if visibility and depth_proxy is not None and cached_depth is None:
        _cpu_projection_depth(
            xyz, frame, depth_proxy, None, depth_id, chunk, workers, executor, projector=native
        )
        projected_pairs += len(depth_proxy)
    if visibility:
        projected_chunks = [None] * ((target_count + chunk - 1) // chunk)
        _cpu_projection_depth(
            xyz,
            frame,
            target_ids,
            None,
            depth_id,
            chunk,
            workers,
            executor,
            projected_chunks,
            native,
        )
    else:
        projected_chunks = _project_targets(
            xyz, frame, target_ids, native, chunk, executor
        )
    projected_pairs += target_count
    filled = minimum_depth_keys(depth_id.reshape(height, width)).ravel() if visibility else None
    image_size, mask, _ = _load_mask(frame, mask_root)
    if image_size != (cal.width, cal.height) or mask.shape != (cal.height, cal.width):
        raise ValueError("Image/mask dimensions do not match calibration")
    mask = native.prepare_mask(mask)
    if destination_ids is not None:
        destination_ids = np.asarray(destination_ids)

    def decide(projected):
        ids, u, v, angle, distance = projected
        if not len(ids):
            return 0, 0, 0
        if rescue_observations is not None:
            ray = (xyz[ids] - frame.center.astype(np.float32)) / distance[:, None]
            rescue_incidence = np.einsum("ij,ij->i", normals[ids], -ray)
            rescue_usable = rescue_incidence > np.float32(0.05)
            rescue_score = (
                np.exp(np.float32(-2) * angle[rescue_usable] ** 2)
                / (distance[rescue_usable] ** 2 + np.float32(0.25))
                * np.maximum(rescue_incidence[rescue_usable], np.float32(0.05)) ** 2
            )
            rescue_destinations = destination_ids[ids[rescue_usable]]
            _insert_top4(
                rescue_observations[0],
                rescue_observations[1],
                rescue_destinations,
                rescue_score,
                photo,
            )
        if visibility:
            pixels = v.astype(np.int32) // 4 * width + u.astype(np.int32) // 4
            result = native.decide(
                ids, u, v, distance, filled[pixels], depth_id[pixels], frame, mask
            )
            usable = (result.flags & np.uint8(8)) != 0
            incidence = result.incidence[usable]
        else:
            ray = (xyz[ids] - frame.center.astype(np.float32)) / distance[:, None]
            all_incidence = np.einsum("ij,ij->i", normals[ids], -ray)
            x = u.astype(np.int32)
            y = v.astype(np.int32)
            usable = (
                (all_incidence > np.float32(0.05))
                & (mask[y, x] == 0)
                & (mask[y + 1, x] == 0)
                & (mask[y, x + 1] == 0)
                & (mask[y + 1, x + 1] == 0)
            )
            incidence = all_incidence[usable]
        ids = ids[usable]
        u = u[usable]
        v = v[usable]
        angle = angle[usable]
        distance = distance[usable]
        score = (
            np.exp(np.float32(-2) * angle**2)
            / (distance**2 + np.float32(0.25))
            * np.maximum(incidence, np.float32(0.05)) ** 2
        )
        if destination_ids is None:
            inserted = _insert_observations(
                native, observations, ids, u, v, score, photo, candidate_slots
            )
        else:
            destinations = destination_ids[ids]
            if np.any(destinations < 0):
                raise RuntimeError("Proxy projection escaped its destination map")
            inserted = _insert_top4(
                observations[0], observations[1], destinations, score, photo
            )
        return len(projected[0]), int(np.count_nonzero(usable)), inserted

    totals = list(executor.map(decide, projected_chunks))
    return {
        "photo": photo,
        "image": str(frame.image_path),
        "target_points": target_count,
        "depth_proxy_points": 0 if depth_proxy is None else len(depth_proxy),
        "evaluated_pairs": projected_pairs,
        "valid_targets": sum(value[0] for value in totals),
        "usable_targets": sum(value[1] for value in totals),
        "inserted": sum(value[2] for value in totals),
        "occupied_depth_pixels": int(np.count_nonzero(depth_id != sentinel))
        if depth_id is not None
        else 0,
    }, depth_id


def _run_dense(
    xyz,
    normals,
    frames,
    mask_root,
    output,
    native,
    workers,
    chunk,
    photos,
    target_source,
    depth_proxy,
    strategy,
    strategy_meta,
    cached_depth_root=None,
    visibility=True,
    candidate_slots=4,
):
    dest = output / "candidates"
    dest.mkdir(parents=True, exist_ok=False)
    observations = _empty_observations(dest / "observations.bin", len(xyz), 4)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for done, photo in enumerate(photos, 1):
            target_ids = (
                None if target_source is None else target_source(int(photo))
            )
            cached = None
            if cached_depth_root is not None:
                cached = np.fromfile(cached_depth_root / f"{int(photo):04d}.i64", dtype="<i8")
            row, _ = _process_photo(
                xyz,
                normals,
                frames,
                int(photo),
                target_ids,
                depth_proxy,
                observations,
                native,
                mask_root,
                workers,
                chunk,
                executor,
                cached_depth=cached,
                visibility=visibility,
                candidate_slots=candidate_slots,
            )
            rows.append(row)
            print(json.dumps({"event": "photo", "done": done, "total": len(photos), **row}), flush=True)
    native.release_geometry()
    observations, finalize = _finalize_observations(
        dest, observations, frames, chunk, (8, 6), workers, native
    )
    observations.flush()
    observations._mmap.close()
    meta = {
        "points": len(xyz),
        "image_count": len(frames),
        "candidates": 4,
        "record_floats": 8,
        "images": rows,
        "collector": "experimental",
        "strategy": strategy,
        "strategy_meta": strategy_meta,
        "candidate_slots_used": candidate_slots,
        "dense_visibility": visibility,
        "profile": {
            **finalize,
            "target_pairs": sum(row["target_points"] for row in rows),
            "evaluated_pairs": sum(row["evaluated_pairs"] for row in rows),
            "mean_depth_occupancy": float(np.mean([row["occupied_depth_pixels"] for row in rows])),
        },
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _proxy_assignments(
    xyz,
    normals,
    frames,
    mask_root,
    native,
    workers,
    chunk,
    cell_index,
    depth_proxy,
    cache_root,
):
    """Choose photos on proxy surfels, optionally caching their depth maps."""
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=False)
    count = len(cell_index.representatives)
    visible_scores = np.zeros((count, 4), dtype=np.float32)
    visible_photos = np.full((count, 4), -1, dtype=np.int16)
    analytic_scores = np.zeros((count, 4), dtype=np.float32)
    analytic_photos = np.full((count, 4), -1, dtype=np.int16)
    original_to_cell = np.full(len(xyz), -1, dtype=np.int32)
    original_to_cell[cell_index.representatives] = np.arange(count, dtype=np.int32)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for photo in range(len(frames)):
            row, depth_id = _process_photo(
                xyz,
                normals,
                frames,
                photo,
                cell_index.representatives,
                depth_proxy,
                (visible_scores, visible_photos),
                native,
                mask_root,
                workers,
                chunk,
                executor,
                destination_ids=original_to_cell,
                rescue_observations=(analytic_scores, analytic_photos),
            )
            if cache_root is not None:
                depth_id.astype("<i8", copy=False).tofile(cache_root / f"{photo:04d}.i64")
            rows.append(row)
            print(
                json.dumps({"event": "proxy-photo", "done": photo + 1, "total": len(frames), **row}),
                flush=True,
            )
    visible_scores, visible_photos = _sort_top4(visible_scores, visible_photos)
    analytic_scores, analytic_photos = _sort_top4(analytic_scores, analytic_photos)
    visible_top1_before_rescue = visible_photos[:, 0] >= 0
    rescued = 0
    for slot in range(4):
        missing = visible_photos[:, slot] < 0
        if not np.any(missing):
            continue
        for analytic_slot in range(4):
            candidate = analytic_photos[:, analytic_slot]
            duplicate = np.any(visible_photos == candidate[:, None], axis=1)
            take = missing & (candidate >= 0) & ~duplicate
            visible_photos[take, slot] = candidate[take]
            visible_scores[take, slot] = analytic_scores[take, analytic_slot]
            rescued += int(np.count_nonzero(take))
            missing = visible_photos[:, slot] < 0
            if not np.any(missing):
                break
    assigned_top1 = visible_photos[:, 0] >= 0
    rescued_top1 = assigned_top1 & ~visible_top1_before_rescue
    assigned_points = int(np.sum(cell_index.counts[assigned_top1], dtype=np.int64))
    rescued_points = int(np.sum(cell_index.counts[rescued_top1], dtype=np.int64))
    return visible_photos, {
        "proxy_rows": rows,
        "proxy_shortlist_rescued_slots": rescued,
        "proxy_shortlist_rescue_fraction": rescued
        / max(1, int(np.count_nonzero(visible_photos >= 0))),
        "top1_rescued_cells": int(np.count_nonzero(rescued_top1)),
        "top1_rescue_fraction_cells": float(np.mean(rescued_top1)),
        "top1_rescued_dense_points": rescued_points,
        "top1_rescue_fraction_dense_points": rescued_points / max(1, assigned_points),
        "mean_transferred_photos": float(np.mean(np.count_nonzero(visible_photos >= 0, axis=1))),
    }


def collect_experimental(
    geometry: Path,
    cameras: Path,
    calibration: Path,
    mask_root: Path | None,
    output: Path,
    visibility_library: Path,
    strategy: str,
    workers: int = 8,
    chunk: int = 262144,
):
    """Run one of the three named approximate candidate experiments."""
    if strategy not in {"keyframes20", "cell1", "proxy8cm"}:
        raise ValueError(f"Unknown candidate experiment: {strategy}")
    started = perf_counter()
    frames = load_camera_frames(cameras, calibration)
    xyz, normals = _load_geometry_arrays(geometry)
    if len(xyz) > np.iinfo(np.uint32).max:
        raise ValueError("Experimental collector currently requires uint32 source IDs")
    native = CpuVisibility(xyz, normals, visibility_library)
    if strategy == "keyframes20":
        selected, strategy_meta = select_keyframes(xyz, normals, frames, native, 20)
        _run_dense(
            xyz,
            normals,
            frames,
            mask_root,
            output,
            native,
            workers,
            chunk,
            selected,
            None,
            None,
            strategy,
            strategy_meta,
        )
    elif strategy == "cell1":
        cells, shortlist, strategy_meta = build_photo_shortlist(
            xyz, normals, frames, native, 0.08, 4
        )
        strategy_meta["dense_occlusion"] = False
        strategy_meta["dense_rechecks_per_cell"] = 1
        target_buckets = _bucket_photo_targets(cells, shortlist, len(frames), 1)
        _run_dense(
            xyz,
            normals,
            frames,
            mask_root,
            output,
            native,
            workers,
            chunk,
            np.arange(len(frames), dtype=np.uint32),
            lambda photo: target_buckets[photo],
            None,
            strategy,
            strategy_meta,
            visibility=False,
            candidate_slots=1,
        )
    else:
        cells = build_cell_index(xyz, normals, 0.08, True)
        cache_root = None
        assignments, proxy_meta = _proxy_assignments(
            xyz,
            normals,
            frames,
            mask_root,
            native,
            workers,
            chunk,
            cells,
            None,
            cache_root,
        )
        target_buckets = _bucket_photo_targets(cells, assignments, len(frames), 1)
        strategy_meta = {
            "assignment_cell_size_m": cells.size,
            "normal_split": cells.normal_split,
            "assignment_cells": len(cells.representatives),
            "depth_proxy_cell_size_m": cells.size,
            "depth_proxy_normal_split": cells.normal_split,
            "depth_proxy_points": len(cells.representatives),
            "dense_rechecks_per_cell": 1,
            **proxy_meta,
        }
        _run_dense(
            xyz,
            normals,
            frames,
            mask_root,
            output,
            native,
            workers,
            chunk,
            np.arange(len(frames), dtype=np.uint32),
            lambda photo: target_buckets[photo],
            None,
            strategy,
            strategy_meta,
            None,
            visibility=False,
            candidate_slots=1,
        )
    meta_path = output / "candidates/meta.json"
    meta = json.loads(meta_path.read_text())
    proxy_rows = meta.get("strategy_meta", {}).get("proxy_rows", [])
    proxy_pairs = sum(row["evaluated_pairs"] for row in proxy_rows)
    meta["profile"]["proxy_evaluated_pairs"] = proxy_pairs
    meta["profile"]["evaluated_pairs"] += proxy_pairs
    geometry_sha256 = digest(geometry)
    camera_sha256 = digest(cameras)
    calibration_sha256 = digest(calibration)
    meta.update(
        {
            "geometry_sha256": geometry_sha256,
            "camera_sha256": camera_sha256,
            "calibration_sha256": calibration_sha256,
            "experimental": True,
        }
    )
    meta["wall_s"] = perf_counter() - started
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"strategy": strategy, "wall_s": meta["wall_s"]}), flush=True)
