# S20 native pipeline

A standalone macOS pipeline for raw SHARE S20 captures: LiDAR/IMU decoding → native tracking and deskew → optional pose correction → Metal geometry filtering → raw JPEG extraction → calibrated camera poses → MPS person masks → visibility-aware photo selection → local exposure correction → consensus blending → colored LAS.

The default color algorithm is the accepted visibility/consensus implementation. No Windows executable, Studio masks, Studio colors or Studio poses are needed to run it. The Metal geometry implementation was informed by recovered Studio kernel behavior; its provenance is documented in [PROVENANCE.md](docs/PROVENANCE.md).

**Status: tested engineering release candidate, not a broadly certified production release.** This repository removes the experiment-folder dependencies and fixed photo count. See [VALIDATION.md](docs/VALIDATION.md) for exactly what passed, and the remaining release gates. Large scans are not yet fully out of core. The garden recording's camera convention is not validated by the indoor test.

## Install and build

Requires Apple Silicon macOS, Xcode command-line tools/Metal SDK, CMake ≥3.24, Python 3.11–3.13 and sufficient local storage. Build dependencies are fetched on first build; mask weights download on first use. Provision both before working offline. Capture data stays outside Git.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[masks,dev]'
s20 build --jobs 8
pytest -q
build/s20_geometry --self-test --kernels native/geometry.metal
```

KISS-ICP is pinned to a commit in CMake. The original cloned importer remains separate and is not modified. An existing matching KISS checkout can be used with `s20 build --kiss-source /path/to/kiss-icp/cpp/kiss_icp`.

## Process a capture

```sh
s20 inspect /Volumes/SD_CARD/Test
s20 estimate /Volumes/SD_CARD/Test
s20 run /Volumes/SD_CARD/Test \
  --output /path/to/outputs/test-run \
  --camera-clock sensor-header \
  --camera-convention lidar-extrinsics \
  --resources throughput
```

The input must contain one `all_*.bag` and `info/calibration.yaml`. The capture is opened read-only, and the output must be separate. No clock offset, Studio epoch, image count, local path or historical trim interval is built into the importer. The two camera flags explicitly select the supported convention; they do not establish calibration accuracy for an untested recording. Out-of-coverage images and long pose gaps are rejected and recorded.

The result is `export/colorized.las`; intermediate products, source identities, per-stage receipts and `events.jsonl` live in the run directory. With `--no-color`, the result is `geometry/filtered.ply`, and camera-convention flags are unnecessary. Native coordinates are not automatically surveyed coordinates.

Ctrl-C terminates the worker process group. To continue an interrupted run, repeat the **same command** with `--resume`. Inputs (by size and modification time), code, native binary and completed-output hashes must match; resource limits such as `--memory-gb` and `--resources` may change. Incomplete stage directories are retained under `incomplete/`; completed data is never silently overwritten. Only one process may own a run.

## Supported options

| Flag | Default | Effect |
|---|---|---|
| `--resources interactive/balanced/throughput` | balanced | CPU budgets: up to 2/8/all logical cores; color workers up to 2/4/8. Throughput matches the accepted eight-worker color configuration on the test Mac. |
| `--cpu-threads N`, `--color-workers N` | From profile | Explicit limits. More workers can increase memory and contention. |
| `--memory-gb N` | 16 GB | Process-tree RSS ceiling; sampled every 0.5 s. Stops a stage exceeding it. Not a total unified-memory or Metal-driver reservation. |
| `--chunk-points N` | 262144 | Temporary CPU color chunk size. Geometry and full-image projection still retain scene-sized arrays. |
| `--mask person/off` | person | LR-ASPP person exclusion; off can retain people/operators. No glass-class claim. |
| `--mask-device mps/cpu` | mps | Same segmentation model on selected backend. No silent CPU fallback. |
| `--exposure local/global/off` | local | Spatial per-photo offsets, per-photo constant offsets, or no exposure correction. Consensus blending remains enabled. |
| `--blend metal/cpu` | metal | Metal consensus kernel or independent NumPy reference. |
| `--no-pose-refinement` | Refinement enabled | Skip the native frozen-plane correction; keep tracked poses. |
| `--no-color` | Color enabled | Stop after full Metal geometry processing. |
| `--frame-start A --frame-end B` | All frames | Explicit inclusive geometry trim by native frame ID. |
| `--max-pose-gap S` | 0.5 s | Reject photo interpolation across larger gaps. |

Experimental inertial tracking and the tiny bilinear-fit variant are not exposed in this release; they remain in the archived research workspace. There is one supported geometry filter and one consensus kernel, rather than several duplicated experiment pipelines.

To color existing geometry with externally validated native camera poses:

```sh
s20 colorize --geometry /data/filtered.ply \
  --cameras /data/camera_frames.json --calibration /data/calibration.yaml \
  --masks /data/person-masks --output /data-new/color-run \
  --resources throughput
```

Omit `--masks` to generate masks, or select `--mask off`. PLY requires XYZ and oriented normal fields. Image identities, transforms and calibrated dimensions are validated. The exposure grid is intentionally fixed at 8×6 and four candidate views; image count is dynamic.

## Desktop app

`apps/desktop/` holds S20 Studio, a Tauri 2 + React app around this pipeline: projects that collect raw scan folders, build-pipeline style progress for each run, and a point cloud viewer with axis-locked views, orientation calibration, box slicing and LAS export of slices. See [apps/desktop/README.md](apps/desktop/README.md).

## Repository layout

- `src/s20_pipeline/`: import, configuration, stage orchestration, projection, exposure, export and telemetry.
- `native/`: C++ tracking/pose correction, Objective-C++ Metal hosts and Metal kernels.
- `tests/`: correctness and failure-path tests; GPU parity runs when native binaries are built.
- `tools/`: full-cloud golden comparison and repository checks.
- `apps/desktop/`: the S20 Studio desktop app.
- `docs/`: desktop app plan, event contract, performance model, provenance and validation.
- `licenses/`: retained third-party license notices.

Generated outputs, model weights, build trees, environments, camera data and historical comparisons are excluded. The existing comparison viewer and all accepted outputs remain in the workspace's separate `outputs/` folder.
