"""Provisional timing model. No core-count extrapolation from one measured machine."""


def estimate(metadata, host, points=None):
    frame_scale = metadata["lidar_frames"] / 671
    photos = metadata["photos"]
    point_scale = points / 5096684 if points is not None else frame_scale
    same_host = "M4 Max" in (host.get("cpu_model") or "") and host.get("logical_cpu_cores") == 16
    stages = {
        "geometry_preparation_s": 38.09 * frame_scale,
        "person_masks_s": 7.73 * photos / 58,
        "visibility_exposure_blend_export_s": 25.20 * point_scale * photos / 58,
    }
    seconds = sum(stages.values())
    return {
        "schema": 1,
        "estimated_seconds": seconds if same_host else None,
        "reference_machine_seconds": seconds,
        "stages_on_reference_machine": stages,
        "range_seconds": [seconds * 0.6, seconds * 2.5] if same_host else None,
        "confidence": "low; heuristic range, not a statistical confidence interval",
        "reference_hardware": "M4 Max, 16 CPU / 40 GPU cores, 64 GB unified memory",
        "assumptions": [
            "Same photo resolution, model weights cached, comparable geometry density.",
            "Color collector currently projects every point into every selected photo (points × photos).",
            "Photo extraction/import I/O is not independently calibrated; total is incomplete.",
            "Point count inferred from LiDAR frames unless supplied; frame rate/density may differ.",
            "No assumption that twice as many CPU/GPU cores halves runtime.",
        ],
        "candidate_file_bytes": int(point_scale * 5096684 * 128),
        "warning": "5–20× longer captures can make this collector 25–400× slower if both points and photos scale. Spatial image selection is required before claiming linear scaling.",
    }
