#!/usr/bin/env python3
"""Run unchanged exposure, blend, and export stages for an experiment."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from s20_pipeline.export import export
from s20_pipeline.exposure import Exposure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=262144)
    args = parser.parse_args()

    meta = json.loads((args.experiment / "candidates/meta.json").read_text())
    exposure = Exposure(args.experiment, meta["image_count"])
    exposure.solve("global")
    exposure.solve("local")
    blend = args.experiment / "blend"
    blend.mkdir()
    shutil.copyfile(args.experiment / "local/field.bin", blend / "field.bin")
    subprocess.run(
        [
            args.blend,
            args.kernels,
            args.experiment / "candidates/observations.bin",
            blend / "field.bin",
            blend / "colors.bin",
            blend / "metal.json",
        ],
        check=True,
    )
    export(
        args.geometry,
        args.experiment / "candidates/observations.bin",
        blend / "colors.bin",
        args.experiment / "export",
        args.chunk,
    )


if __name__ == "__main__":
    main()
