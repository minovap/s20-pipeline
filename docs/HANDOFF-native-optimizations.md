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

Syntax-check `geometry.mm` quickly with `clang++ -x objective-c++ -std=c++20 -fobjc-arc -fsyntax-only native/geometry.mm`. The main checkout's `build/` is already configured; `/opt/homebrew/bin/cmake --build build -j 8` there rebuilds the real binaries after merging. Python: `.venv/bin/python -m pytest -q` (17 tests, under a second; GPU parity tests run when binaries exist), `.venv/bin/python -m ruff check --fix` and `ruff format` before committing. `build/s20_geometry --self-test --kernels native/geometry.metal` must pass.

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

### 1. Cull points per photo in the color collector (largest win)

File: `src/s20_pipeline/collect.py`, function `collect`. Today the loop `for index, f in enumerate(fs)` projects **all n points into every photo** twice: once to build the photo's depth buffer (`depth_id`, quarter resolution, `np.minimum.at` of packed depth and point id) and once in `color_chunk` per 262,144-point chunk. On the big scan that is 122 M × 1,094 projections per pass.

Plan:

- Once, before the photo loop, bucket points into a coarse grid (2 m voxels is plenty): `np.floor(xyz / 2)` → sort indices by voxel key, keep voxel bounds. Memory is one int64 per point plus the permutation.
- Per photo, test each voxel's 8 corners against the camera frustum and a maximum range (the collector already limits by `d`; use the same limit) and take only points of voxels that pass. Then project only those for the depth buffer and for coloring.
- The depth buffer must still see every point that could occlude, so the frustum test must be conservative (expand by one voxel). Because a culled point could not have projected into the image, the depth buffer and the chosen candidates are unchanged.
- Keep `observations.bin` layout (`n × 4 × 8` float32) and the progress calls untouched; `blend.mm` reads that file.

Validate on the small scan: `observations.bin` must be byte-identical before and after (`cmp`). Then run the big scan's candidates stage and record the time. Expect 10 to 50 times faster.

### 2. Metal projection and depth test for the collector

Only after task 1. Move the per-photo projection, depth-buffer build and visibility test into a Metal kernel driven from a small Objective-C++ host, following the structure of `native/geometry.mm` (`MetalRayFilter`, `make_buffer`, `dispatch` with its per-launch `@autoreleasepool`). Keep the existing NumPy path selectable for parity testing, like `--blend cpu` does for the blend stage (see `worker.py` stage `blend` and `cli.py` `--blend`). The kernel must reproduce `project()` from `src/s20_pipeline/camera.py` (fisheye model) and the visibility rule in `color_chunk` exactly; use float32 throughout as NumPy does. Validate by comparing `observations.bin` with the NumPy path on the small scan; small float differences in `u, v` are acceptable only if the chosen photo ids and the visible flags are identical.

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
