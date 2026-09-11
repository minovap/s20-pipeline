# Source provenance and dependencies

This folder contains source and documentation for a separate repository. It does not contain raw captures, photographs, Studio binaries, disassemblies, decompiler projects, benchmark clouds, private machine paths or model weights. The original `import-scan-from-device` checkout remains intact outside it.

| Component | Origin |
|---|---|
| Raw decoding and calibrated packing | Our native S20 extraction/packer scripts, made independent of workspace layout. |
| Tracking and optional pose correction | Our C++ LiDAR/IMU engine and frozen-plane pose refinement. Uses KISS-ICP core registration/map primitives; not Studio SLAM/BA. |
| Metal geometry host | Preserved profiled implementation from the accepted benchmark generation, with the original v4 Metal kernel from importer commit `d90e900`. This deliberately does not silently switch to the later v5 filter. |
| Camera and PLY readers | Source copied from our cloned importer, with Studio-pose loading removed from the production camera module. |
| Photo candidate collector | Accepted eight-worker visibility/tangent-plane collector; paths, masks and photo count parameterized. |
| Exposure fit and consensus | Our robust local/global exposure fit and deterministic consensus blend; no Studio color values used for generation. |
| MPS masks | Torchvision LR-ASPP MobileNet V3 Large person segmentation, CPU or MPS. Weights downloaded separately. |

The Metal geometry kernel is a native port informed by recovered CUDA/PTX semantics. It is **not represented as an independently clean-room implementation**. Keep that provenance visible. No new license is assigned to the whole repository here; the owner can choose repository/distribution terms with that history in mind. Preparing this folder for a private Git commit is not a claim about public redistribution rights.

Third-party notices retained in `licenses/`:

- KISS-ICP, MIT, pinned commit `b16835283aee62f7d5e2bdf6c1c3bb2930de74ff`: [upstream](https://github.com/PRBonn/kiss-icp).
- Mindkosh point-cloud colorization, MIT, conceptual starting point for early photo projection/depth selection, commit `509b8f1a3e414244ee22d646f6b6db178b5b9e48`: [upstream](https://github.com/Mindkosh/colorize-lidar-pointcloud).

The CMake dependency graph also uses Eigen, Sophus, TBB and robin-map under their upstream terms. The Python dependency manifest and lock record the tested packages. A packaged binary distribution must carry the dependency notices appropriate to what it includes; the current repository is a source checkout, not a finished signed app installer.
