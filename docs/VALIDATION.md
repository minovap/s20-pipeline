# Validation and release status

Validated locally on 11 September 2026, Apple M4 Max, 16 CPU cores / 40 GPU cores / 64 GB unified memory.

## Accepted color regression

Ran the new standalone color command using the existing validated geometry, camera poses and person masks. Compared every output point against the accepted visibility-consensus LAS:

- 5,027,200 points, same order and count.
- Every 16-bit RGB channel exactly equal.
- Maximum decoded XYZ difference 5.33×10⁻¹⁵ m (floating-point representation of different LAS offsets).
- Candidate binary SHA-256 exactly equal to the accepted eight-worker candidate binary.
- LAS file bytes differ because the new streaming writer uses centered offsets; this is not a color or spatial change.

This is the quality regression for the code extraction. It does not claim that a newly tracked full raw run has identical geometry to the historical trimmed experiment.

## Original SD-card capture, fresh full run

Read `/Volumes/SD_CARD/Test`, writing to a separate workspace output folder. No reused decoded frames, photos, masks or Studio processing were used. Model weights were already cached.

| Result | Value |
|---|---:|
| Raw bag | 583,032,095 bytes; 69.356884 s bag duration |
| Raw LiDAR / IMU | 671 frames / 13,414 samples |
| Raw photos extracted | 64 |
| Native poses retained | 669 |
| Cameras accepted | 62; first pair outside native pose coverage |
| Metal geometry | 6,136,485 points |
| Colored LAS | 6,024,829 points |
| Processing-stage wall sum | 85.625 s |
| First stage start through completion | 87.247 s, including intermediate output hashing |
| Peak sampled process-tree RSS | 7.40 GB, geometry stage |
| Final LAS SHA-256 | `6c0c41373ddd5bea23c8800c240080a086bb8bd3ed5923fe81e022158b393ac1` |

These timings exclude initial bag inspection/input hashing and software/model downloads. They include worker startup and the 0.5-second orchestration polling interval. This run used all supported frames, not the old 100–569 comparison trim, and more photos than the old 58-photo test.

| Stage | Wall seconds |
|---|---:|
| Decode | 0.518 |
| Pack | 1.029 |
| Tracking | 8.834 |
| Pose correction | 1.568 |
| Registered scan staging | 5.239 |
| Metal geometry | 31.595 |
| Raw photos | 0.521 |
| Cameras | 0.520 |
| MPS masks | 7.263 |
| Visibility candidates | 23.844 |
| Global exposure | 1.038 |
| Local exposure | 1.577 |
| Metal blend | 1.039 |
| LAS export | 1.039 |

Bag and calibration SHA-256 values were unchanged after processing. Repeating the command with `--resume` verified and reused all 14 completed stages, without reprocessing. Full-resolution comparison of the new all-frame cloud against Studio is not part of the accepted-color regression and has not been visually signed off.

## CPU resource limit check

Replayed full Metal geometry with its CPU worker limit set to two. All 6,136,485 XYZ positions, normals, intensities, capture times and frame IDs were exactly equal to the unrestricted run. Diagnostic ray-noise scores differed by at most 7.63×10⁻⁶; the GPU reduction is not claimed byte-deterministic. CPU resource profiles now also cap the geometry worker pool and BLAS threads.

## Automated checks

12 tests currently cover exact integer depth keys above double precision; dynamic photo counts; empty exposure overlap; consensus rejection of a conflicting best view; source/output symlink containment; unknown-hardware estimate behavior; changed-output resume rejection; dynamic CPU/Metal parity and GPU address validation; cross-chunk foreground occlusion; camera clock/gap rejection; and process-group cancellation. GPU tests run when the Metal worker is built; they are explicitly skipped otherwise. The native geometry self-test also passed.

Python lint and formatting, package installation, CMake native builds and source-only repository checks are part of the local validation. CI configuration repeats portable checks and macOS native builds; GPU parity/self-tests require a physical Metal-capable Mac; no remote CI result is claimed before the repository is pushed.

## Remaining production release gates

The source is organized for a commit and the supported indoor path runs end to end. Calling it a universally production-ready scanner application would overstate the evidence. Before a public/stable release:

- Validate additional devices/recordings, especially the garden optical/clock convention. Do not adopt the historical diagnostic −25° correction.
- Validate 5×–20× captures, bound geometry/projection/exposure memory, and implement spatial photo selection. Current complexity is points × photos.
- Add split/multiple-bag capture support, calibrated disk-space estimates, and deliberate out-of-disk/power-loss/large-memory-pressure fault tests.
- Package the worker/weights, add native progress events and platform GPU/system-pressure collectors, then sign and test the desktop installer.
- Establish dependency redistribution terms and a release/update policy appropriate to the recorded source provenance.

The app layout and framework decision are in [APP_PLAN.md](APP_PLAN.md). Historical estimates and prioritized CPU/GPU work are in [PERFORMANCE.md](PERFORMANCE.md).
