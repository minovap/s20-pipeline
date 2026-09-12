#!/usr/bin/env python3
"""Run one opt-in approximate photo-matching strategy on staged inputs."""

import argparse
from pathlib import Path

from s20_pipeline.experimental_collect import collect_experimental


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visibility-library", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("keyframes20", "cell1", "proxy8cm"), required=True
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=262144)
    parser.add_argument("--keyframe-percent", type=int, default=30)
    args = parser.parse_args()
    collect_experimental(
        args.geometry,
        args.cameras,
        args.calibration,
        args.masks,
        args.output,
        args.visibility_library,
        args.strategy,
        args.workers,
        args.chunk,
        keyframe_percent=args.keyframe_percent,
    )


if __name__ == "__main__":
    main()
