"""LR-ASPP MobileNet V3 person masks; weights provisioned separately."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .camera import load_camera_frames


@dataclass(frozen=True)
class MaskInput:
    name: str
    image_path: Path


def mask_inputs(source, calibration):
    """Images to mask: either the photo index (a list) or calibrated camera frames."""
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [MaskInput(item["camera"], Path(item["image"]).resolve()) for item in payload]
    return [MaskInput(f.name, f.image_path) for f in load_camera_frames(source, calibration)]


def masks(cameras, calibration, dest, device="mps", workers=8, progress=lambda *args: None):
    import torch
    import torchvision
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )

    dest.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(workers)
    torch.set_num_interop_threads(1)
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; select CPU masks explicitly")
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    model = lraspp_mobilenet_v3_large(weights=weights).eval().to(device)
    transform = weights.transforms()
    person = weights.meta["categories"].index("person")
    rows = []
    sync_wall = 0.0
    preprocess_wall = 0.0
    save_wall = 0.0
    fs = mask_inputs(cameras, calibration)
    if not fs:
        raise ValueError("No images to mask")
    with torch.inference_mode():
        for start in range(0, len(fs), 4):
            batch = fs[start : start + 4]
            tick = time.perf_counter()
            images = [Image.open(f.image_path).convert("RGB") for f in batch]
            x = torch.stack([transform(im) for im in images]).to(device)
            preprocess_wall += time.perf_counter() - tick
            if device == "mps":
                torch.mps.synchronize()
            tick = time.perf_counter()
            labels = model(x)["out"].argmax(1)
            if device == "mps":
                torch.mps.synchronize()
            sync_wall += time.perf_counter() - tick
            tick = time.perf_counter()
            for f, im, label in zip(batch, images, labels.cpu().numpy()):
                # Keep the model's predicted person pixels; no glass-class claim.
                mask = Image.fromarray((label == person).astype("uint8") * 255).resize(
                    im.size, Image.Resampling.NEAREST
                )
                target = dest / (f.name + "_mask") / (f.image_path.stem + ".png")
                target.parent.mkdir(exist_ok=True)
                mask.save(target)
                rows.append(
                    {
                        "image": str(f.image_path),
                        "mask": str(target),
                        "excluded_pixels": int(np.count_nonzero(mask)),
                        "pixels": im.width * im.height,
                        "input_shape": list(x.shape[1:]),
                    }
                )
            save_wall += time.perf_counter() - tick
            progress(len(rows), len(fs))
    result = {
        "device": device,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "model": str(weights),
        "mask_semantics": "255 = predicted person, 0 = usable. No window/glass/sky classes.",
        "batch_size": 4,
        "torch_threads": workers,
        "images": rows,
        "synchronized_forward_wall_s": sync_wall,
        "preprocess_wall_s": preprocess_wall,
        "save_and_readback_wall_s": save_wall,
        "gpu_timing_scope": "Synchronized model-forward wall time, not hardware GPU command duration.",
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory()
        if device == "mps"
        else None,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()
        if device == "mps"
        else None,
    }
    (dest / "result.json").write_text(json.dumps(result, indent=2))
