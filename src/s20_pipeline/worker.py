"""One isolated processing stage per process; used by CLI and future desktop app."""

import json
import subprocess
import sys
from pathlib import Path

from .storage import atomic_json


def progress(done, total, unit=None):
    row = {"event": "progress", "done": done, "total": total}
    if unit:
        row["unit"] = unit
    print(json.dumps(row), flush=True)


def execute(stage, job):
    out = Path(job["output"])
    cfg = job["options"]
    native = Path(job["native"])
    dest = out / stage
    geometry = Path(job["geometry"]) if job["mode"] == "colorize" else out / "geometry/filtered.ply"
    cameras = Path(job["cameras"]) if job["mode"] == "colorize" else out / "cameras/frames.json"
    calibration = Path(job["calibration"])

    def call(args):
        subprocess.run([str(a) for a in args], check=True)

    if stage == "decode":
        from .decode import extract

        extract(Path(job["bag"]), dest, progress)
    elif stage == "pack":
        from .pack import pack

        pack(out / "decode", calibration, dest, progress)
    elif stage == "tracking":
        call(
            [
                native / "s20_reconstruct",
                out / "pack/raw-native.bin",
                dest,
                cfg["cpu_threads"],
                0,
                0.12,
                0.005,
                1,
                1,
                "input",
                100,
                0,  # far-observation diagnostics: unused downstream
                1,
                "gyro",
                1,
                0,  # whole-scan voxel map: unused downstream, costs memory
            ]
        )
    elif stage == "pose_refinement":
        call([native / "s20_refine_poses", out / "tracking/observations", dest, cfg["cpu_threads"]])
    elif stage == "registered":
        from .geometry import stage_registered

        stage_registered(
            out / "tracking/observations",
            dest,
            out / "pose_refinement/poses.txt" if cfg["pose_refinement"] else None,
            progress,
        )
    elif stage == "geometry":
        command = [
            native / "s20_geometry",
            "--input",
            out / "registered",
            "--output",
            dest,
            "--kernels",
            Path(job["native_sources"]) / "geometry.metal",
            "--max-frames",
            -1,
        ]
        if cfg["frame_start"] is None:
            command += ["--all-frames"]
        else:
            command += ["--frame-start", cfg["frame_start"], "--frame-end", cfg["frame_end"]]
        call(command)
    elif stage == "photos":
        from .capture import extract_photos

        extract_photos(Path(job["bag"]), dest, progress)
    elif stage == "cameras":
        from .capture import prepare_cameras

        prepare_cameras(
            out / "photos/images.json",
            out / "registered/FrameOptPose.txt",
            calibration,
            dest / "frames.json",
            cfg["max_pose_gap"],
        )
    elif stage == "masks":
        from .masks import masks

        # In run mode the photo index is enough, so masks do not wait for poses.
        source = out / "photos/images.json" if job["mode"] == "run" else cameras
        masks(source, calibration, dest, cfg["mask_device"], cfg["cpu_threads"], progress)
    elif stage == "candidates":
        photo_matching = cfg.get("photo_matching", "exact")
        collector = cfg.get("collector", "cpu")
        mask_root = (
            Path(job["masks"])
            if job.get("masks")
            else (out / "masks" if cfg["mask"] == "person" else None)
        )
        if photo_matching == "keyframes":
            from .experimental_collect import collect_experimental

            if collector != "cpu":
                raise ValueError("Fast photo matching requires the CPU collector")
            collect_experimental(
                geometry,
                cameras,
                calibration,
                mask_root,
                out,
                native / "libs20_visibility.dylib",
                "keyframes20",
                cfg["color_workers"],
                cfg["chunk_points"],
                progress,
                keyframe_percent=cfg.get("keyframe_percent", 30),
            )
        elif photo_matching == "exact":
            from .collect import collect

            collect(
                geometry,
                cameras,
                calibration,
                mask_root,
                out,
                cfg["color_workers"],
                cfg["chunk_points"],
                progress,
                collector,
                native / "libs20_collector.dylib" if collector == "metal" else None,
                Path(job["native_sources"]) / "collector.metal"
                if collector == "metal"
                else None,
                cfg.get("collector_diagnostics", False),
                visibility_library=(native / "libs20_visibility.dylib")
                if collector == "cpu" and (native / "libs20_visibility.dylib").is_file()
                else None,
            )
        else:
            raise ValueError(f"Unknown photo matching mode: {photo_matching}")
    elif stage in ("global", "local"):
        from .camera import load_camera_frames
        from .exposure import Exposure

        Exposure(out, len(load_camera_frames(cameras, calibration))).solve(stage)
    elif stage == "blend":
        import numpy as np

        from .camera import load_camera_frames
        from .exposure import Exposure
        from .ply import read_ply_info

        dest.mkdir()
        ex = Exposure(out, len(load_camera_frames(cameras, calibration)))
        field = (
            np.load(out / cfg["exposure"] / "field.npy")
            if cfg["exposure"] != "off"
            else np.zeros((ex.image_count, 6, 8, 3), dtype="<f4")
        )
        field.tofile(dest / "field.bin")
        if cfg["blend"] == "metal":
            call(
                [
                    native / "s20_blend",
                    Path(job["native_sources"]) / "blend.metal",
                    out / "candidates/observations.bin",
                    dest / "field.bin",
                    dest / "colors.bin",
                    dest / "metal.json",
                ]
            )
        else:
            c = ex.candidates()
            n = read_ply_info(geometry).point_count
            rgb = np.memmap(dest / "colors.bin", mode="w+", dtype="<f4", shape=(n, 4))
            for start in range(0, n, cfg["chunk_points"]):
                rgb[start : start + cfg["chunk_points"]] = ex.reference(
                    c[start : start + cfg["chunk_points"]], field, robust=True
                )
                progress(min(n, start + cfg["chunk_points"]), n, "points")
            rgb.flush()
            atomic_json(dest / "cpu.json", {"backend": "NumPy reference", "points": n})
    elif stage == "export":
        from .export import export

        export(
            geometry,
            out / "candidates/observations.bin",
            out / "blend/colors.bin",
            dest,
            cfg["chunk_points"],
            progress,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    execute(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()))
