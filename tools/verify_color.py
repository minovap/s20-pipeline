"""Compare an independently rerun color result with a retained LAS golden fixture."""

import argparse
import json
from pathlib import Path

import laspy
import numpy as np
from s20_pipeline.storage import atomic_json, digest

p = argparse.ArgumentParser()
p.add_argument("result", type=Path)
p.add_argument("reference", type=Path)
p.add_argument("--report", type=Path, required=True)
a = p.parse_args()
count = 0
max_xyz = 0.0
max_rgb = 0
with laspy.open(a.result) as actual, laspy.open(a.reference) as expected:
    if actual.header.point_count != expected.header.point_count:
        raise ValueError("Point count differs")
    for x, y in zip(actual.chunk_iterator(262144), expected.chunk_iterator(262144)):
        max_xyz = max(
            max_xyz,
            float(np.max(abs(np.column_stack([x.x, x.y, x.z]) - np.column_stack([y.x, y.y, y.z])))),
        )
        max_rgb = max(
            max_rgb,
            int(
                np.max(
                    abs(
                        np.column_stack([x.red, x.green, x.blue]).astype("i4")
                        - np.column_stack([y.red, y.green, y.blue]).astype("i4")
                    )
                )
            ),
        )
        count += len(x)
result = {
    "points": count,
    "max_xyz_difference_m": max_xyz,
    "max_rgb_difference_16bit": max_rgb,
    "result_sha256": digest(a.result),
    "reference_sha256": digest(a.reference),
    "pass": max_xyz < 1e-10 and max_rgb == 0,
    "note": "LAS headers/offsets may differ; compares all decoded coordinates and color values in order.",
}
atomic_json(a.report, result)
print(json.dumps(result, indent=2))
if not result["pass"]:
    raise SystemExit(1)
