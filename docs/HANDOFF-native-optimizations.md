# Handoff: native pipeline optimizations, round two

For a fresh agent session. Everything below is verified as of 11 September 2026, commit `cb79c1e` on `main` of https://github.com/minovap/s20-pipeline (private). Read this whole file before touching code.

## What this repository is

`s20-pipeline` turns a raw SHARE S20 LiDAR capture (a ROS bag plus calibration) into a colored LAS point cloud. The desktop app in `apps/desktop` (Tauri 2 + React) drives it. Stages run as separate worker processes launched by `src/s20_pipeline/runner.py` in two lanes: geometry (`decode → pack → tracking → pose_refinement → registered → geometry`) and color (`photos → masks → cameras → candidates → global → local → blend → export`). Stage code lives in `src/s20_pipeline/worker.py`; native workers in `native/` (`main.cpp` = tracker, `geometry.mm` = Metal filter, `refine_poses.cpp`, `blend.mm`).

The user is the only developer. He runs scans through the app while you work, so:

- **Never edit `native/` or `src/` while a run is in progress.** The pipeline hashes code and binaries into `job.json`; a change mid-run does not break the run, but a Rust change under `apps/desktop/src-tauri` triggers `tauri dev` to rebuild and restart the app, which kills the running pipeline with "Broken pipe". Check with `pgrep -fl s20_pipeline.cli` and `cat "~/Documents/S20 Projects/*/runs/*/state.json"` before editing.
- Develop native changes in a git worktree (see Build) and merge when validated.
- Resume tolerates code changes since `608f317`, so the user can resume old runs after you rebuild.

## Build

`cmake` is not on PATH; it is `/opt/homebrew/bin/cmake`. Network fetches of dependencies hang, so point CMake at the sources already on disk:

```sh
git worktree add -b <branch> /tmp/s20-next main
cd /tmp/s20-next
/opt/homebrew/bin/cmake -S native -B build -DCMAKE_BUILD_TYPE=Release \
  -DKISS_SOURCE="/Users/olcay/Documents/ChatGPT/Garden design/pipeline/dependencies/kiss-icp/cpp/kiss_icp" \
  -DFETCHCONTENT_SOURCE_DIR_SOPHUS="/Users/olcay/Documents/ChatGPT/Garden design/s20-pipeline/build/_deps/sophus-src" \
  -DFETCHCONTENT_SOURCE_DIR_TESSIL="/Users/olcay/Documents/ChatGPT/Garden design/s20-pipeline/build/_deps/tessil-src" \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
/opt/homebrew/bin/cmake --build build -j 8 --target s20_reconstruct s20_geometry
```

Syntax-check `geometry.mm` quickly with `clang++ -x objective-c++ -std=c++20 -fobjc-arc -fsyntax-only native/geometry.mm`. The main checkout's `build/` is already configured; `/opt/homebrew/bin/cmake --build build -j 8` there rebuilds the real binaries after merging. Python: `.venv/bin/python -m pytest -q` (24 tests, around a second; GPU parity tests run when binaries exist), `.venv/bin/python -m ruff check --fix` and `ruff format` before committing. `build/s20_geometry --self-test --kernels native/geometry.metal` must pass.

## Benchmark inputs and method

| Input | Path | Size |
|---|---|---|
| Small indoor scan, raw | `/Volumes/SD_CARD/Test` (SD card, may be unmounted) | 671 frames, 64 photos |
| Small scan, packed for the tracker | `/Users/olcay/Documents/ChatGPT/Garden design/outputs/runs/s20-desktop-validation-20260911/Test-2026-09-11T15-50-42-075Z/pack/raw-native.bin` | 12.5 M points |
| Small scan, baseline full run receipts | same folder, `receipts/*.json` | |
| Big garden scan, packed | `~/Documents/S20 Projects/Test/runs/2026-09-11 19-22-39/pack/raw-native.bin` | 10,972 frames, 4.4 GB |
| Big scan, registered cloud for geometry | `~/Documents/S20 Projects/Test/runs/2026-09-11 19-22-39/registered/` | 160 M points |
| Big scan, geometry output to compare against | the run folder `2026-09-11 21-xx` in the same `runs/` directory, or rerun `build/s20_geometry` on the registered folder | 121,778,659 kept, 13,141,047 noise, 30 subfiles |

Do not modify anything under `~/Documents/S20 Projects`; write benchmark outputs to a scratch directory. Run tracking directly with the worker's exact arguments (see `worker.py`, stage `tracking`): `s20_reconstruct <pack> <out> <threads> 0 0.12 0.005 1 1 input 100 0 1 gyro 1 0`. Run geometry with `s20_geometry --input <registered> --output <out> --kernels native/geometry.metal --max-frames -1 --all-frames`. Run the registered conversion with `from s20_pipeline.geometry import stage_registered`.

Measure memory with a sampling loop, not `/usr/bin/time -l` (it reports a different figure than the pipeline's psutil sampler):

```sh
"$BIN" ... & pid=$!; peak=0; while kill -0 $pid 2>/dev/null; do r=$(ps -o rss= -p $pid | tr -d ' '); [ "$r" -gt "$peak" ] && peak=$r; sleep 10; done; echo "peak $((peak/1000000)) GB"
```

Run one benchmark at a time; concurrent runs distort wall time (a contaminated tracker run measured 721 s against 340 s clean).

### Validation rule

Every optimization must leave outputs identical:

- Tracking: `trajectory.txt` position difference 0.000 mm against the previous binary (they are bit-identical today; `np.loadtxt` both and compare).
- Geometry: `filter_stats.json` `kept_points`, `noise_points`, `ray_subfile_count` equal. Note that float atomics on the GPU already make scores order-dependent, so equality of counts is the accepted check.
- Registered: `diff -rq` of the output folders is empty.
- Full pipeline on the small scan: `export/result.json` `colored_points` equals 6,024,829.

### Numbers so far

| Stage, big scan | Before this week | Now |
|---|---|---|
| tracking | 411 s, 32 GB | 340 s, 17 GB |
| geometry | 3,643 s, 56 GB (failed at a 32 GB limit) | 451 s, 17 GB |
| registered | 90 s, one core | parallel, not yet measured on the big scan |
| candidates | never completed on the big scan | expected hours: 122 M points × 1,094 photos |

Small scan end to end: 157 s before, 92 s now. Details in `docs/PERFORMANCE.md`.

## The five tasks

Do them in this order. Each is independent; validate and commit each one separately.

### 1. Cull points per photo in the color collector — DONE (11 September 2026)

**Completed and validated.** The user selected the small indoor scan as the benchmark instead of a full large-scan run. The original and optimized `observations.bin` files pass `cmp`; direct collection measured **23.76 s before → 22.92 s after**, with sampled peak RSS **2.65 GB → 2.97 GB**. This compact scan takes the contiguous full-cloud fallback, so the timing difference is not evidence of a substantial culling speedup. A fresh full pipeline run exported exactly **6,024,829 colored points**. All **24 tests** and the Metal geometry self-test passed. A separate six-photo large-cloud diagnostic also produced byte-identical observations. The full large benchmark was stopped at the user's request; no full large-scan timing or 10–50× speedup is claimed. See `docs/PERFORMANCE.md` for details.

Implementation notes: the current collector already caches projections, so it projects once per photo rather than twice. It has a minimum distance (`d > 0.1`) but **no maximum range**; adding a far cutoff would change candidates. The implementation preserves this unlimited range. Its 2 m voxel index bounds all eight corners with a one-voxel halo, then uses conservative interval bounds through the fisheye polynomial and image rectangle. Testing only whether individual corners project inside the image is not conservative. Original point IDs, packed depth ties, the candidate layout and progress calls are preserved.

File: `src/s20_pipeline/collect.py`, function `collect`. Before this task, the loop `for index, f in enumerate(fs)` projected **all n points into every photo**, cached the projection to build the photo's depth buffer (`depth_id`, quarter resolution, `np.minimum.at` of packed depth and point id), and reused it in `color_chunk` per 262,144-point chunk. On the big scan that was 122 M × 1,094 projections.

Plan:

- Once, before the photo loop, bucket points into a coarse grid (2 m voxels is plenty): `np.floor(xyz / 2)` → sort indices by voxel key, keep voxel bounds. Memory is one int64 per point plus the permutation.
- Per photo, conservatively bound each voxel's 8 corners against the fisheye image and take only points of voxels that pass. Keep the existing `d > 0.1` test; there is no maximum range to reuse. Then project only those for the depth buffer and for coloring.
- The depth buffer must still see every point that could occlude, so the frustum test must be conservative (expand by one voxel and bound the full box, not just individual projected corners). Because a culled point could not have projected into the image, the depth buffer and the chosen candidates are unchanged.
- Keep `observations.bin` layout (`n × 4 × 8` float32) and the progress calls untouched; `blend.mm` reads that file.

Validation: use the small scan as the acceptance and performance benchmark, per the user's updated instruction. `observations.bin` must be byte-identical before and after (`cmp`), and the full pipeline must preserve the exported colored-point count. The original proposed large-scan benchmark and 10–50× speedup expectation are superseded.

### 2. Metal projection and depth test for the collector

Only after task 1. Use the **small indoor scan** as the acceptance and performance benchmark. Implement this task in the checkpoints below; mark the whole task done only after both Metal checkpoints pass validation.

#### First: validate an all-float32 CPU reference

Do not assume the current NumPy collector uses float32 throughout. `project()` uses float32 for the native float32 geometry, but `plane_depth`, `hit`, and `patch_distance` in `collect.py` become float64 through the camera center; the decoded exact-depth calculation also uses float64. An all-float32 implementation is worth testing before deciding that mixed precision must be preserved.

- Freeze the current CPU reference and its small-scan `observations.bin`. Reuse identical geometry, camera frames, calibration, photos, masks, photo order and collector settings for every comparison.
- Create a CPU experimental variant with explicitly float32 floating-point visibility calculations, including camera-center arithmetic, plane intersections, patch distances and depth comparisons. Keep packed depth keys and point IDs as exact integers. Do not change thresholds, quantization or ranking rules.
- Compare the experimental `observations.bin` with the frozen reference using `cmp`. Byte identity is the primary output acceptance check for this experiment. Also compare per-photo validity and visibility flags: the observation file retains only the top four candidates and cannot expose every intermediate visibility decision.
- If both output and decision parity pass, use the validated float32 variant as the implementation reference for the Metal port, while retaining the original golden output. Record that this validates the small scan and boundary fixtures, rather than proving equivalence for every possible input.
- If comparisons differ, identify the first differing photo, point and decision. Preserve the required CPU calculations, or develop and validate a targeted CPU recheck for numerically ambiguous cases. Do not silently relax thresholds or replace the golden output to obtain a pass.

#### 2A. Metal projection and deterministic depth buffers

Move per-photo fisheye projection, depth-buffer construction and the neighborhood minimum to Metal. Keep visibility, masking, color sampling and candidate ranking on the CPU for this checkpoint so projection/depth disagreements can be isolated.

- Reproduce `project()` from `src/s20_pipeline/camera.py`, including transform conventions, polynomial evaluation, axis handling, validity bounds and radial distance. Start with fast math disabled; do not copy geometry's fast-math setting when reusing its host structure.
- Preserve `np.rint(d * 1e6)` depth quantization, the packed `(quantized depth, original point ID)` ordering, sentinel behavior and range checks. Task 1's culling must not renumber original point IDs. Check the target device's atomic capabilities and use a deterministic multi-pass reduction if necessary; do not replace exact integer tie-breaking with floating-point atomics.
- Build the depth buffer from all retained points for the photo before testing any point's visibility. Preserve the exact radius-3 neighborhood minimum over packed keys. Chunked processing must retain global occluders across chunks.
- Validate projected-pixel validity, exact depth keys, winning point IDs and neighborhood blocker IDs against the CPU reference before proceeding to 2B.

#### 2B. Metal visibility

Move the visibility calculation in `color_chunk` to Metal only after 2A passes parity. Reproduce the ray/normal incidence, plane intersection, patch-reliability and depth-threshold decisions using the precision strategy validated above. Keep masking, bilinear color sampling, scoring and top-four candidate selection on the CPU while validating this checkpoint.

Require identical per-photo visibility flags and retained photo IDs. If GPU arithmetic changes a decision, diagnose it rather than accepting it merely because the final colored-point count happens to match. A CPU recheck for ambiguous cases is acceptable only when its decision parity and total runtime are measured.

#### Host and backend integration

Use a persistent Objective-C++ host, following `native/geometry.mm` for `MetalRayFilter`, buffer management and dispatch structure. Compile kernels once, load geometry and normals once within the memory budget, and reuse buffers across photos. Update camera parameters and selected point IDs per photo. Retain the per-launch `@autoreleasepool`; avoid launching a new process or copying the entire cloud for each photo.

Add `--collector cpu|metal` through `cli.py` and the candidates stage in `worker.py`, following the existing `--blend` selection pattern. Keep CPU as the default until Metal passes the correctness and performance checks. Record the selected backend and required native/source identities in run metadata. Preserve Task 1's culling, `observations.bin` layout (`n × 4 × 8` float32), original point IDs and progress calls.

#### Validation and performance acceptance

- Provide an optional diagnostic comparison mode for per-photo projection validity, depth winners, neighborhood blocker IDs and visibility flags. Keep these diagnostics separate from the production observation-file layout.
- Compare `observations.bin` with the frozen CPU output. Aim for byte identity. Small Metal differences in projected `u, v` are acceptable only within explicitly documented tolerances and with identical validity/visibility decisions, candidate occupancy and retained photo IDs in their ranked slots. Report score and color differences as well; unchanged photo IDs alone do not establish color parity.
- Exercise image/pixel boundaries, equal-depth ties, neighborhood boundaries, near-axis projections, empty views, culling, chunk boundaries and each visibility threshold. Compare decisions against the original CPU reference, not only the experimental float32 variant.
- Run the full small-scan pipeline and require exactly **6,024,829 colored points**. Report any exported color differences from the CPU result.
- Benchmark with the same frozen small-scan inputs: one warm-up for each backend, then at least three alternating CPU/Metal measurements, one run at a time. Compare median total candidates-stage wall time, sampled peak RSS and Metal buffer allocation. Report kernel time separately; total stage time must include host initialization, transfers, allocations and output writing. Do not regenerate geometry between paired collector benchmarks.
- Promote Metal to the default only after decision/output acceptance passes and median total collector time improves. Update `docs/PERFORMANCE.md` with the measurements and limitations. A full garden benchmark is not required for this task.

### 3. Parallelize the density kd-tree in geometry

File: `native/geometry.mm`, function `assign_density_levels`. It groups points by 1 m block (`kBlockSize`), and for each block with more than 7 points builds a kd-tree (`build_kd_tree`) over the block plus its neighbors and runs `nearest_other` for every point to get the mean nearest-neighbor distance. Blocks are independent. Wrap the per-range loop in `dispatch_apply` over ranges (GCD is already used in `list_ray_subfiles` and `extract_ray_subfile`; blocks capture by const copy, so take `.data()` pointers before the block). Write per-range results into preallocated arrays indexed by point; no shared mutable state. It was 43 s single-threaded on the big scan. Validate with equal `filter_stats.json` counts and identical `density_level_counts`.

### 4. Write geometry output incrementally

File: `native/geometry.mm`, `run_partitioned_filter` and `write_ply`. Today each subfile's kept points are appended to `combined` (points, frame ids, times, scores, normals) and the PLY is written at the end from that copy, about 5 GB extra on the big scan. Instead open the PLY at the start, write the header with a placeholder vertex count, stream each subfile's kept records (the `PlyPoint` struct, packed) as they finish, and seek back to patch the count. `write_stats` needs only the aggregate counters. The order of points in the PLY must stay the same (subfile order, then index order), and `result.kept_points` must still be right. Validate with `cmp` of `filtered.ply` before and after on the small scan.

### 5. Exact-size voxel storage in the KISS-ICP local map

The tracker's remaining 17 GB is the local map: `kiss_icp::VoxelHashMap` at `/Users/olcay/Documents/ChatGPT/Garden design/pipeline/dependencies/kiss-icp/cpp/kiss_icp/core/VoxelHashMap.cpp`. In `AddPoints` each new voxel does `voxel_points.reserve(max_points_per_voxel_)` (20 points × 24 bytes) whether the voxel ever fills. Change it to grow naturally (`push_back` without the reserve, or reserve 4) so memory follows actual occupancy. This is the vendored copy that `KISS_SOURCE` points at, so the change is local; note it in `docs/PROVENANCE.md`. Registration results must not change (the map contents are identical, only capacity differs): validate with the trajectory comparison on both scans and the memory trace on the big scan.

## Things that bit us, so you do not repeat them

- `newBufferWithBytes` copies; command buffers are autoreleased and retain their buffers. Every Metal launch in `geometry.mm` now runs inside its own `@autoreleasepool`; keep it that way in new code.
- `reserve(size + n)` before each append is quadratic. The loader now reserves once from PCD headers.
- Objective-C blocks capture C++ locals by const copy; index through a raw pointer.
- `clang` warnings about float conversion are errors only in KISS's own flags, not in the `s20_*` targets.
- The pipeline's psutil sampler (every 0.5 s) can miss short spikes; use the `ps` loop for peak RSS.
- `stage_names` order in `runner.py` and `STAGES` in `apps/desktop/src/types.ts` must list the same ids; the app forecasts steps from the latter.

## Commit and push

Commit each task separately with a message that states the measured before and after. Push to `origin main`. Attribution footer used in this repository:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
```

Update `docs/PERFORMANCE.md` with the new rows when a task changes a number in the table above.
