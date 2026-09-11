"""CLI/API boundary for native reconstruction; never invokes Windows programs."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .storage import digest, validate_destination


def source_identity(path):
    """Cheap identity for large source files: size and modification time.

    Source folders are assumed unchanged between runs, so the multi-gigabyte
    bag is not hashed; resume still notices a replaced or rewritten file.
    """
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def repository():
    # Source checkout is explicit; a wheel-only installation requires --native-dir.
    return Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(prog="s20", description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "inspect", help="Read capture metadata and bag index; no reconstruction"
    )
    inspect.add_argument("capture", type=Path)
    estimate = commands.add_parser("estimate", help="Provisional reference-machine estimate")
    estimate.add_argument("capture", type=Path)
    estimate.add_argument("--points", type=int)
    build = commands.add_parser("build", help="Build the macOS native workers")
    build.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    build.add_argument("--kiss-source", type=Path)
    for command in ("run", "colorize"):
        c = commands.add_parser(
            command,
            help="Raw capture to cloud"
            if command == "run"
            else "Color existing geometry with validated camera poses",
        )
        if command == "run":
            c.add_argument("capture", type=Path)
            c.add_argument(
                "--camera-clock",
                choices=["sensor-header"],
                help="Explicitly select raw sensor-header to native-origin clock",
            )
            c.add_argument(
                "--camera-convention",
                choices=["lidar-extrinsics"],
                help="Use recorded LiDAR-to-camera calibration; validate projections for each new device/recording type",
            )
        else:
            c.add_argument("--geometry", type=Path, required=True)
            c.add_argument("--cameras", type=Path, required=True)
            c.add_argument("--calibration", type=Path, required=True)
            c.add_argument(
                "--masks",
                type=Path,
                help="Reuse precomputed masks, hashing every file for provenance",
            )
        c.add_argument("--output", type=Path, required=True)
        c.add_argument(
            "--resources", choices=["interactive", "balanced", "throughput"], default="balanced"
        )
        c.add_argument("--cpu-threads", type=int)
        c.add_argument("--color-workers", type=int)
        c.add_argument("--memory-gb", type=float, default=16)
        c.add_argument("--chunk-points", type=int, default=262144)
        c.add_argument("--mask", choices=["person", "off"], default="person")
        c.add_argument("--mask-device", choices=["mps", "cpu"], default="mps")
        c.add_argument("--exposure", choices=["local", "global", "off"], default="local")
        c.add_argument("--blend", choices=["metal", "cpu"], default="metal")
        c.add_argument("--pose-refinement", action=argparse.BooleanOptionalAction, default=True)
        c.add_argument("--color", action=argparse.BooleanOptionalAction, default=True)
        c.add_argument("--frame-start", type=int)
        c.add_argument("--frame-end", type=int)
        c.add_argument("--max-pose-gap", type=float, default=0.5)
        c.add_argument("--native-dir", type=Path, default=repository() / "build")
        c.add_argument("--native-sources", type=Path, default=repository() / "native")
        c.add_argument(
            "--resume",
            action="store_true",
            help="Verify all input/code/output hashes before reusing completed stages",
        )
    return p


def prepare_job(args):
    from PIL import Image

    from .camera import load_camera_frames
    from .capture import inspect_capture
    from .runner import files_identity

    cores = os.cpu_count() or 1
    threads = {"interactive": min(2, cores), "balanced": min(8, cores), "throughput": cores}[
        args.resources
    ]
    workers = {
        "interactive": min(2, cores),
        "balanced": min(4, cores),
        "throughput": min(8, cores),
    }[args.resources]
    options = {
        name: getattr(args, name)
        for name in (
            "resources",
            "memory_gb",
            "chunk_points",
            "mask",
            "mask_device",
            "exposure",
            "blend",
            "pose_refinement",
            "color",
            "frame_start",
            "frame_end",
            "max_pose_gap",
        )
    }
    options.update(
        cpu_threads=args.cpu_threads if args.cpu_threads is not None else threads,
        color_workers=args.color_workers if args.color_workers is not None else workers,
    )
    if not 1 <= options["cpu_threads"] <= 128 or not 1 <= options["color_workers"] <= 128:
        raise ValueError("Thread counts must be 1–128")
    if (
        not 1 <= args.memory_gb <= 1024
        or not 1024 <= args.chunk_points <= 4194304
        or not 0 < args.max_pose_gap <= 10
    ):
        raise ValueError("Invalid memory, chunk or pose-gap limit")
    if (args.frame_start is None) != (args.frame_end is None):
        raise ValueError("Specify both frame-start and frame-end")
    if args.frame_start is not None and not 0 <= args.frame_start <= args.frame_end:
        raise ValueError("Invalid frame range")
    job = {
        "schema": 1,
        "mode": args.command,
        "output": str(args.output.resolve()),
        "options": options,
        "native": str(args.native_dir.resolve()),
        "native_sources": str(args.native_sources.resolve()),
    }
    sources = []
    identities = {}
    if args.command == "run":
        if args.color and (args.camera_clock is None or args.camera_convention is None):
            raise ValueError(
                "Color import requires --camera-clock sensor-header --camera-convention lidar-extrinsics. No clock/frame convention is guessed."
            )
        metadata = inspect_capture(args.capture)
        job.update(
            bag=metadata["bag"], calibration=metadata["calibration"], capture=metadata["capture"]
        )
        sources = [args.capture]
        identities = {p: source_identity(p) for p in [job["bag"], job["calibration"]]}
        job["clock"] = args.camera_clock
        job["camera_convention"] = args.camera_convention
    else:
        if not args.color:
            raise ValueError("--no-color is only valid for raw run")
        job.update(
            geometry=str(args.geometry.resolve()),
            cameras=str(args.cameras.resolve()),
            calibration=str(args.calibration.resolve()),
        )
        sources = [args.geometry, args.cameras, args.calibration]
        for key in ("geometry", "cameras", "calibration"):
            identities[job[key]] = source_identity(job[key])
        fs = load_camera_frames(args.cameras, args.calibration)
        if not fs:
            raise ValueError("No camera frames")
        for f in fs:
            sources.append(f.image_path)
            identities[str(f.image_path)] = source_identity(f.image_path)
            with Image.open(f.image_path) as im:
                if im.size != (f.calibration.width, f.calibration.height):
                    raise ValueError("Image dimensions do not match calibration")
        if args.masks:
            job["masks"] = str(args.masks.resolve())
            sources.append(args.masks)
            identities[job["masks"]] = files_identity(args.masks)
    if args.resume:
        if any(
            args.output.resolve().is_relative_to((s if s.is_dir() else s.parent).resolve())
            for s in sources
        ):
            raise ValueError("Output overlaps source")
    else:
        validate_destination(args.output, sources)
    needed = []
    if args.command == "run":
        needed = ["s20_reconstruct", "s20_geometry"] + (
            ["s20_refine_poses"] if args.pose_refinement else []
        )
    if args.color and args.blend == "metal":
        needed += ["s20_blend"]
    for name in needed:
        path = args.native_dir / name
        if not path.is_file():
            raise ValueError(f"Missing native worker {path}; run s20 build")
        identities[str(path.resolve())] = digest(path)
    job["source_identities"] = identities
    job["python_code"] = files_identity(Path(__file__).parent)
    # Compiled bytecode is environment dependent, not source identity.
    job["python_code"] = {k: v for k, v in job["python_code"].items() if k.endswith(".py")}
    job["native_code"] = files_identity(args.native_sources)
    return job


def main():
    args = parser().parse_args()
    try:
        if args.command == "build":
            command = [
                "cmake",
                "-S",
                str(repository() / "native"),
                "-B",
                str(repository() / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
            ]
            if args.kiss_source:
                command += ["-DKISS_SOURCE=" + str(args.kiss_source.resolve())]
            subprocess.run(command, check=True)
            subprocess.run(
                ["cmake", "--build", str(repository() / "build"), "-j", str(args.jobs)], check=True
            )
        elif args.command in ("inspect", "estimate"):
            from .capture import inspect_capture

            metadata = inspect_capture(args.capture)
            if args.command == "estimate":
                from .estimate import estimate
                from .telemetry import hardware

                metadata = estimate(metadata, hardware(), args.points)
            print(json.dumps(metadata, indent=2))
        else:
            from .runner import run

            run(prepare_job(args), args.resume)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"s20: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
