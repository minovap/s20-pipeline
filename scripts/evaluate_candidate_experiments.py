#!/usr/bin/env python3
"""Compare approximate candidate/color outputs with the frozen exact control."""

import argparse
import hashlib
import json
from pathlib import Path

import laspy
import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def candidate_metrics(reference, candidate, points, chunk=262144):
    reference = np.memmap(reference, dtype="<f4", mode="r", shape=(points, 4, 8))
    candidate = np.memmap(candidate, dtype="<f4", mode="r", shape=(points, 4, 8))
    distribution = np.zeros(5, dtype=np.int64)
    reference_occupied = 0
    recalled = 0
    top1_shared = 0
    top1_equal = 0
    nonfinite = 0
    for start in range(0, points, chunk):
        end = min(points, start + chunk)
        expected = reference[start:end]
        actual = candidate[start:end]
        expected_occupied = expected[:, :, 7] > 0
        actual_occupied = actual[:, :, 7] > 0
        distribution += np.bincount(actual_occupied.sum(axis=1), minlength=5)
        reference_occupied += int(np.count_nonzero(expected_occupied))
        matches = (
            (expected[:, :, None, 6] == actual[:, None, :, 6])
            & expected_occupied[:, :, None]
            & actual_occupied[:, None, :]
        )
        recalled += int(np.count_nonzero(np.any(matches, axis=2)))
        shared = expected_occupied[:, 0] & actual_occupied[:, 0]
        top1_shared += int(np.count_nonzero(shared))
        top1_equal += int(np.count_nonzero(shared & (expected[:, 0, 6] == actual[:, 0, 6])))
        occupied_values = actual[actual_occupied]
        nonfinite += int(np.count_nonzero(~np.isfinite(occupied_values)))
    return {
        "candidate_count_distribution_0_to_4": distribution.tolist(),
        "reference_occupied_slots": reference_occupied,
        "reference_top4_photo_recall": recalled / max(1, reference_occupied),
        "shared_covered_points": top1_shared,
        "top1_photo_agreement": top1_equal / max(1, top1_shared),
        "nonfinite_occupied_values": nonfinite,
    }


def las_metrics(reference_root, candidate_root, point_count):
    reference_ids = np.fromfile(reference_root / "source-indices.u64", dtype="<u8")
    candidate_ids = np.fromfile(candidate_root / "source-indices.u64", dtype="<u8")
    if np.any(reference_ids[1:] <= reference_ids[:-1]) or np.any(
        candidate_ids[1:] <= candidate_ids[:-1]
    ):
        raise ValueError("Export source IDs must be strictly increasing")
    reference_mask = np.zeros(point_count, dtype=bool)
    candidate_mask = np.zeros(point_count, dtype=bool)
    reference_mask[reference_ids] = True
    candidate_mask[candidate_ids] = True
    shared_ids = np.flatnonzero(reference_mask & candidate_mask)
    reference_positions = np.searchsorted(reference_ids, shared_ids)
    candidate_positions = np.searchsorted(candidate_ids, shared_ids)
    reference_las = laspy.read(reference_root / "colorized.las")
    candidate_las = laspy.read(candidate_root / "colorized.las")
    reference_rgb = np.column_stack(
        [reference_las.red, reference_las.green, reference_las.blue]
    )[reference_positions].astype(np.float32) / np.float32(257)
    candidate_rgb = np.column_stack(
        [candidate_las.red, candidate_las.green, candidate_las.blue]
    )[candidate_positions].astype(np.float32) / np.float32(257)
    error = np.abs(reference_rgb - candidate_rgb).reshape(-1)
    reference_xyz = np.asarray(reference_las.xyz)[reference_positions]
    candidate_xyz = np.asarray(candidate_las.xyz)[candidate_positions]
    xyz_difference = np.linalg.norm(reference_xyz - candidate_xyz, axis=1)
    return {
        "reference_covered_points": len(reference_ids),
        "candidate_covered_points": len(candidate_ids),
        "coverage_fraction_of_geometry": len(candidate_ids) / point_count,
        "coverage_retained_from_reference": len(shared_ids) / len(reference_ids),
        "coverage_lost_points": int(np.count_nonzero(reference_mask & ~candidate_mask)),
        "coverage_gained_points": int(np.count_nonzero(candidate_mask & ~reference_mask)),
        "shared_points": len(shared_ids),
        "shared_xyz_max_difference_m": float(xyz_difference.max(initial=0)),
        "rgb_absolute_channel_mae_8bit": float(np.mean(error)),
        "rgb_absolute_channel_p50_8bit": float(np.percentile(error, 50)),
        "rgb_absolute_channel_p95_8bit": float(np.percentile(error, 95)),
        "rgb_absolute_channel_p99_8bit": float(np.percentile(error, 99)),
        "rgb_absolute_channel_p99_9_8bit": float(np.percentile(error, 99.9)),
        "rgb_channel_fraction_over_8": float(np.mean(error > 8)),
        "rgb_channel_fraction_over_16": float(np.mean(error > 16)),
        "rgb_channel_fraction_over_32": float(np.mean(error > 32)),
        "nonfinite_rgb_values": int(np.count_nonzero(~np.isfinite(candidate_rgb))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-seconds", type=float, default=7.134188625)
    parser.add_argument("--experiment", action="append", required=True)
    args = parser.parse_args()
    point_count = json.loads((args.reference / "candidates/meta.json").read_text())["points"]
    result = {
        "reference": str(args.reference),
        "point_count": point_count,
        "exact_candidate_median_s": args.baseline_seconds,
        "experiments": {},
    }
    for specification in args.experiment:
        name, value = specification.split("=", 1)
        root = Path(value)
        meta = json.loads((root / "candidates/meta.json").read_text())
        metrics = {
            "candidate_wall_s": meta["wall_s"],
            "candidate_speedup": args.baseline_seconds / meta["wall_s"],
            "candidate": candidate_metrics(
                args.reference / "candidates/observations.bin",
                root / "candidates/observations.bin",
                point_count,
            ),
            "color": las_metrics(args.reference / "export", root / "export", point_count),
            "strategy": meta["strategy_meta"],
            "collector_semantics": {
                "dense_visibility": meta.get(
                    "dense_visibility", meta["strategy"] == "keyframes20"
                ),
                "candidate_slots_used": meta.get(
                    "candidate_slots_used", 4 if meta["strategy"] == "keyframes20" else 1
                ),
            },
            "candidate_sha256": sha256(root / "candidates/observations.bin"),
            "colorized_sha256": sha256(root / "export/colorized.las"),
        }
        result["experiments"][name] = metrics
        print(json.dumps({"experiment": name, **metrics["color"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
