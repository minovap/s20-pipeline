# Performance and scaling

These historical measurements are from the small indoor scan on an M4 Max with 16 CPU cores, 40 GPU cores and 64 GB unified memory. They are not predictions for an arbitrary scan or computer.

| Stage | Historical wall time | Resource character |
|---|---:|---|
| Native geometry preparation (grouped) | 38.09 s | Raw I/O, CPU tracking/pose processing and Metal geometry; not a single homogeneous GPU stage. |
| Native MPS person masks | 7.73 s | GPU inference plus CPU image decode/transform and mask save. Model weights cached. |
| Visibility candidates, 8 workers | 21.74 s | About 2.14 average CPU cores and 2.47 GB peak RSS; serial full-image depth pass limits utilization. |
| Global exposure fit | 0.64 s | CPU sparse solve. |
| Local exposure fit | 1.43 s | CPU sparse solve. |
| Metal blend process | 0.15 s | Kernel around 14.56 ms in separate warm tests; roughly 0.734 GB allocated Metal buffers. |
| Color export | 1.24 s | CPU quantization and file I/O. |
| Native color group | 25.20 s | Candidate collection dominates. |

Earlier visibility color took 42.20 s; the eight-worker optimization preserved every candidate and the accepted LAS. Studio-component masking/color on the Mac took 155.17 + 65.18 = 220.35 s, versus 7.73 + 25.20 = 32.93 s for native photo stages. The previously quoted ~71.02 s native total was a sum of measured groups; extracted photos were reused, so it was **not a measured complete raw-to-color run**. The new repository validation records its own complete run, including fresh import and orchestration, separately.

## Larger scans

Current candidate storage is 128 bytes per geometry point. At 5.1 million points it is about 652 MB; at 20× it is about 13.0 GB. Geometry, normals, the voxel index, per-photo projection and temporary arrays add memory. Disk-backed candidates bound one allocation, not the entire pipeline.

Projection now uses a conservative 2 m voxel index per photo. Its cost follows the points in the retained fisheye bounds; worst-case views still project the entire cloud. If points and selected photos both grow 5×, the work can still grow 25×; at 20×, 400×. Do not extrapolate the 25.20-second color stage linearly or promise that more cores solve this.

Prioritized improvements:

1. **Spatial photo selection — implemented.** A sorted 2 m point-voxel index bounds the complete polynomial fisheye image with a one-voxel halo. Both depth and color passes use the retained points, preserving all possible occluders and original point-ID ties. Views retaining at least 90% of points use contiguous full-cloud slices to avoid gather overhead. There is no maximum-distance cutoff in the existing collector, so none was added. The small-scan observations are byte-identical; see the measurements below.
2. **GPU projection, depth and visibility — implemented as an opt-in backend.** The persistent Metal collector uses deterministic packed depth/ID ties and targeted CPU repair for the established mixed-precision decisions. It is faster on the indoor scan, but the CPU backend remains the default because tiny projection differences can cross downstream consensus thresholds; see the measurements below.
3. **Streaming geometry and observations.** Decode and process frame windows, spill observation tiles, avoid loading the entire raw recording into the C++ front end. Bound map/submap lifetime. Existing tracking is sequential across time; parallelize registration residuals and local reductions rather than independent time chunks with no pose continuity.
4. **Overlap useful work.** Decode/transform the next photo while inference runs, pipeline file reads with CPU work, and process independent pose residuals/geometry blocks in parallel. Budget all work against unified memory; benchmark with and without overlap.
5. **Bound exposure fitting.** Stratified samples per overlap edge and connected-component handling; do not let graph rows grow without a cap. Preserve held-out quality evaluation. The sparse solve is not currently the bottleneck, so a GPU port is lower priority.
6. **Stream GPU blending and export.** Chunk the Metal host's buffers and LAS export; GPU blend is already short and should follow projection work in priority. The new LAS writer already bounds export temporaries. Add LAZ only with an explicit supported compressor and measured CPU tradeoff.
7. **LOD rendering.** Shared tile cache and point budget across comparison panes; suspend unnecessary rendering while full processing runs.

## Measurement contract

Each run saves immutable input identities, configuration and native/source identities; per-stage wall time, OS-reported child CPU time, sampled process-tree RSS, and explicit GPU measurements where available. Wall times include worker startup and the orchestration polling interval, but exclude completed-output hashing. Source hashing/import preflight occurs before processing stage events and must be included separately when reporting user-visible end-to-end time.

CPU core equivalents = CPU seconds / wall seconds. Per-process RSS is not unified GPU usage. Metal allocation overlaps mapped/unified memory, and MPS synchronized forward wall time is not hardware GPU command duration. No tool in this package currently reports system GPU occupancy. Missing values are null.

Large-scan release acceptance must cover at least 5× and 20× captures, image-pair rejection accuracy, memory pressure, cancellation and restart. Estimate intervals should widen when selected points/photos, memory pressure or computer hardware differ from the measured reference. The CLI's current range is an explicit heuristic, not a statistical interval.


## Large-scan measurements, 11 September 2026

An 18-minute garden capture (10,972 frames, 160 M registered points, 1,094 photos) on the M4 Max with 64 GB. Peak RSS sampled every 20 s from the process; "before" is the release before the changes below, "after" is the same commit as this note. Outputs are identical on both scans: same trajectory to the millimetre, same kept and noise point counts.

| Stage | Before | After | What changed |
|---|---|---|---|
| Track motion and deskew | 411 s, 32 GB | 340 s, 17 GB | Frames stream from the packed file instead of being preloaded (4.4 GB); the whole-scan voxel map and far-observation file, which no later stage read, are no longer built; observation records are written on a background thread. The remaining memory is the KISS-ICP local map. |
| Register scan frames | 90 s on one core | frames convert in parallel | Per-frame PCD conversion in a process pool. |
| Filter geometry | failed past 32 GB; a run with no limit took 3,643 s at 56 GB | 451 s, 17 GB | The loader reserved the exact size before every frame and so reallocated and copied the whole cloud 10,950 times; it now reserves once from the PCD headers. Every kernel launch drains an autorelease pool, so a subfile's GPU buffers are freed when it finishes instead of at exit. Rays stream to the GPU in 16 M point batches, and subfiles are extracted one at a time with GCD instead of all at once. |

The small indoor test scan (671 frames, 12.5 M points) is unchanged in time and slightly lower in memory; the fixes only matter when the cloud is large.

Not changed: registration itself, the per-block density kd-tree in geometry (43 s here, single threaded), and the 50 m subfile size.


## Collector voxel culling, 11 September 2026

At the user's request, the small indoor scan is the acceptance and performance benchmark. The full garden candidates benchmark was stopped; no full garden timing or 10–50× speedup is claimed.

Task 1 uses conservative 2 m voxel bounds for the existing fisheye projection. It preserves the full-image depth buffer, original point IDs, exact depth ties, candidate record layout and progress calls. The original collector already cached projections and had no maximum range; neither an extra projection pass nor a far clipping plane was removed.

Sequential direct collector runs on the M4 Max, eight color workers and 262,144-point chunks, with process RSS sampled every 0.5 s:

| Input | Before | After | Validation |
|---|---:|---:|---|
| Indoor scan: 6,136,485 points, 62 photos | 23.76 s; 2.65 GB peak RSS | 22.92 s; 2.97 GB peak RSS | `cmp` passed for all observations. The dense-view fallback projects the full cloud here; this small timing difference is not evidence of a substantial speedup. |

The indoor candidate SHA-256 is `48f0dee4f5f4e1ada3dbc39ff915c2d3bd23d97d68bb9b7ca48f668e0b2c0d37` before and after. A fresh raw-to-export indoor run retained exactly **6,024,829 colored points**. All 24 tests and the native Metal geometry self-test passed. Regression coverage includes arbitrary rotations, nonmonotonic distortion, image intersections without visible voxel corners, distant points, masks, empty views, depth ties and occluders across chunk boundaries.

An additional six-photo comparison on all **121,379,229 garden points** also passed `cmp`. The culled path projected 511,417,663 point/photo pairs versus 728,275,374 without culling. Profiling placed voxel selection around 0.19 s per photo; these broad initial views retain roughly two-thirds of the cloud. The profiled six-photo runs took 57.59 s with culling and 53.13 s without, including one-time index construction and sorting/writing the entire 15.5 GB observation file. These short profiled runs are parity diagnostics, not full-scan speedup measurements.

The garden geometry was regenerated from the preserved registered frames because the historical large filtered cloud was unavailable. It contains 121,379,229 kept points, 13,963,472 noise points and 30 ray subfiles; do not equate it with the historical 121,778,659-point output. The collector optimization changes no geometry code.


## Metal projection, depth and visibility collector, 11 September 2026

Task 2 adds `--collector metal` beside the unchanged default `--collector cpu`. It keeps Task 1's point selection and original point IDs, then uses a persistent Objective-C++ host and Metal kernels for fisheye projection, packed depth construction, the exact radius-three neighborhood minimum and visibility decisions. Metal safe/precise math is selected and fast math is disabled. The M4 Max does not expose the required 64-bit atomic minimum, so depth keys use two deterministic 32-bit atomic passes: first the high word, then the lowest low word among keys matching the winning high word.

The all-float32 CPU experiment did not preserve the reference. Its observation file differed in 10 rows, and per-photo flags changed in 59 of 62 photos. The first retained visibility mismatch was photo 1, point 1,120,961: the mixed reference flags were 11 and the all-float32 flags were 7. The production Metal path therefore preserves exact CPU float32 camera-space distance/angle and rechecks numerically ambiguous projection and mixed-precision visibility cases on the CPU. Repairs are chunked. Error bounds cover projection/pixel edges, depth quantization, dot-product cancellation, denominator conditioning, plane intersections, patch distance, exact-depth comparison, incidence, float64-to-float32 camera-center rounding and large world-coordinate spacing. These are conservative engineering bounds validated by the tests and scan below, not a proof for arbitrary inputs.

The frozen input is the same 6,136,485-point geometry, 62 camera frames, calibration, extracted photos, masks, photo order, eight workers and 262,144-point chunks used by the CPU golden file. The CPU golden SHA-256 is `48f0dee4f5f4e1ada3dbc39ff915c2d3bd23d97d68bb9b7ca48f668e0b2c0d37`.

Final diagnostic results across all 62 photos:

- Zero differences in projected validity, packed depth keys, exact pixel winners, radius-three neighborhood blocker keys, or reliability, surface-rejection and final visibility flags.
- Candidate occupancy and ranked photo IDs are identical. Gradient values are identical. Ninety-three scores differ by at most `1.8626451e-9`.
- Grid coordinates differ by at most `1.9073486e-6` cells. Candidate RGB differs by at most `0.0905762` of one 8-bit channel level.
- CPU repairs covered 12,159,333 projection-boundary cases, 1,054,487 visibility cases and 155,575,674 valid-point exact distance/angle cases. The first photo rechecked 120,485 projection cases and 16,507 visibility cases out of 6,136,485 selected points.
- The Metal collector allocated 466,382,952 bytes of buffers. Its 124 command-buffer launches used a median 0.572 seconds of measured GPU command time. The separate 883,689,552-byte Metal allocation reported by the blend stage is a different process and is not simultaneous collector memory.

The final full colorize validation reused the frozen geometry, cameras, photos, masks and calibration. It exported exactly **6,024,829 colored points** with identical membership, order and XYZ coordinates. The output is not byte-identical in color: 602,848 points and 848,877 of 18,074,487 16-bit RGB channel values differ. Of the changed channels, the 50th and 90th percentiles are 1 code value, the 99th is 3 and the 99.9th is 7; 334 points have at least one channel difference greater than 7. Two points cross a downstream consensus/reference threshold by more than 4,096 code values, and the maximum is 7,819 code values, equivalent to about 30.4 levels on an 8-bit scale. The Metal result reports 5,607,856 multi-view-blended points versus 5,607,857 for CPU. Because these sparse threshold effects are larger than numerical noise, Metal remains opt-in.

Performance was measured in separate processes on the Apple M4 Max. Inputs were warmed first, then CPU and Metal were alternated three times, one run at a time. Wall time includes host initialization, transfers, allocation, masking/color/ranking work and writing `observations.bin`. RSS was sampled every 0.1 seconds. GPU command time is reported separately above.

| Backend | Candidate-stage wall samples | Median wall | Median process CPU | Median sampled RSS |
|---|---|---:|---:|---:|
| CPU | 24.287 s, 24.370 s, 24.013 s | 24.287 s | 55.003 s | 2.826 GB |
| Metal | 17.818 s, 17.956 s, 18.045 s | 17.956 s | 35.812 s | 3.062 GB |

Metal reduces median wall time by **26.1%** (`1.353×`) and process CPU time by **34.9%**. Median sampled RSS increases by 236 MB, or **8.37%**. The speedup is mostly lower CPU array work rather than kernel execution; with only about 0.572 seconds in GPU commands, more kernel tuning has little remaining leverage on this scan. Higher-value future work is reducing CPU candidate packing/temporary traffic and bounding retained buffers on large clouds.

Regression fixtures cover equal-depth ties, neighborhood edges, near-axis projection, integer mask boundaries, empty/reused selections, every visibility threshold, large exactly representable coordinates, conditioned plane intersections and sub-threshold camera-center rounding. No full garden timing or memory validation was performed. The collector requires macOS 15 and Metal language 3.2; use `--collector cpu` on unsupported systems or when exact downstream color reproduction is required.


## CPU collector memory and depth parallelism, 12 September 2026

The exact CPU path now releases the source PLY mapping after copying geometry, creates the sparse observation file without eagerly dirtying every page, stores the voxel permutation and leaf starts as `uint32` when the point count permits, and gathers sparse voxel selections without allocating a full-cloud boolean mask. Projection and packed-depth construction use up to four fixed worker stripes with private depth images under a 128 MiB scratch budget; the images are merged only after all stripes finish. Point order, packed depth/ID ties, visibility arithmetic and candidate replacement order are unchanged.

The frozen indoor scan was warmed, then the previous and changed collectors were alternated three times in separate processes. Each run used 6,136,485 points, 62 photos, eight color workers and 262,144-point chunks. RSS was sampled every 0.5 seconds.

| Collector | Candidate-stage wall samples | Median wall | Median sampled RSS |
|---|---|---:|---:|
| Previous CPU | 23.948 s, 24.390 s, 25.366 s | 24.390 s | 2.808 GB |
| Changed CPU | 18.014 s, 18.712 s, 19.337 s | 18.712 s | 2.537 GB |

This is a **23.3% wall-time reduction** (`1.303×`) and a **9.7% RSS reduction** (about 271 MB). All three changed observation files passed `cmp` against the frozen golden file. A separate full first-photo diagnostic over all 6,136,485 points found zero differences in projection validity, packed depth keys, exact winners, neighborhood blockers, reliability/surface decisions or final visibility. The lightweight suite passed 42 tests with 8 Metal/native tests skipped in the isolated worktree.

This change does not meet the later fivefold candidate-stage target. Against the 24.390-second previous median, that target requires an end-to-end result at or below 4.878 seconds on the same frozen input. Further work must reduce projection/visibility/ranking traffic rather than treating parallel depth construction alone as the final design.


## Deferred final-only photo sampling, 12 September 2026

The CPU collector now postpones RGB decoding, bilinear sampling, gradient calculation and grid-coordinate calculation until the four final photo winners are known. During chronological selection it stores only each accepted winner's exact float32 `u`, `v`, photo ID and score in their eventual canonical record fields. Strict score comparison, first-minimum-slot replacement and the final stable descending sort are unchanged. Mask loading is overlapped with the same photo's projection/depth pass.

This is an **in-place deferred-sampling layout**, not a separate compact 64-byte-per-point allocation: `observations.bin` remains a 128-byte-per-point mapping. Final occupied slots are counted and grouped by photo through a temporary slot index of 4 bytes per occupied slot when `4N` fits `uint32`, with a `uint64` fallback. The writable index is flushed and closed before three finalizer workers consume it through bounded positional reads. Sampling chunks are capped at 32,768 slots. Geometry, normals, the voxel index and last-photo temporaries are released before final materialization. These worker and chunk values are measured bounded defaults, not universal optima.

Three sequential unprofiled runs on the same frozen indoor input were compared with the preceding exact CPU checkpoint:

| Collector | Candidate-stage wall samples | Median wall | Median sampled RSS |
|---|---|---:|---:|
| Preceding CPU checkpoint | 18.014 s, 18.712 s, 19.337 s | 18.712 s | 2.537 GB |
| Deferred final sampling | 17.263 s, 16.958 s, 16.827 s | 16.958 s | 2.548 GB |

The change is **9.37% faster** than the preceding checkpoint (`1.103×`) with an 11 MB (`0.45%`) median RSS increase. All three output files passed byte-for-byte comparison with the frozen golden observations. Combined with the preceding patch, median wall time is **30.5% lower** than the 24.390-second pre-task baseline (`1.438×`), while sampled RSS remains about 260 MB lower.

Profiling counted 59,513,401 accepted chronological insertions but only 23,555,511 final occupied slots, so final-only sampling avoids materializing 60.4% of records that would later be overwritten. Photo progress reaches 100% after selection and before the bounded final sort, grouping and materialization pass; the stage itself is not complete until that pass and the final flush finish.

A follow-up visibility cleanup reuses the gathered blocker coordinates for both plane depth and patch distance, and stops gathering projected integer pixels after masking no longer needs them. Three exact runs measured 16.491 s, 16.753 s and 16.724 s (16.724 s median) with 2.525 GB median sampled RSS. That is a further **1.38% wall-time reduction** and about **23 MB lower RSS** than the deferred-sampling checkpoint; all three outputs again matched the frozen golden file byte for byte.


## Valid-only projection intermediate, 12 September 2026

The CPU depth pass now retains each original projection chunk as valid point IDs plus exact float32 `u`, `v`, angle and distance arrays. After the unchanged full-photo depth barrier, visibility consumes those chunks directly. This removes the second validity pass over every selected point while preserving projection batch shapes, point order and exact depth contributions. Workers clear their uniquely owned chunk entry as soon as it is consumed.

The representation costs 20 bytes per valid pair instead of 16 bytes per selected pair for the former dense projection. On the frozen scan, 155,575,674 of 380,462,070 selected point-photo pairs were valid, so aggregate retained projection data falls from 6.087 GB to 3.112 GB across the 62 sequential photos. The largest single-photo intermediate was 96.7 MB. Pixel IDs are deliberately recomputed from retained `u/v`; retaining them raised the representation to 24 bytes per valid pair and increased the peak for dense photos.

| Collector | Candidate-stage wall samples | Median wall | Median sampled RSS |
|---|---|---:|---:|
| Reused-gather checkpoint | 16.491 s, 16.753 s, 16.724 s | 16.724 s | 2.525 GB |
| Valid-only projection | 15.869 s, 15.972 s, 15.790 s | 15.869 s | 2.549 GB |

This is another **5.12% wall-time reduction** (`1.054×`). The 24.6 MB (`0.98%`) sampled-RSS increase is within the variability of the final file-backed materialization peak; during the photo loop, sampled RSS peaked between 2.075 and 2.129 GB. All three output files matched the frozen golden observations byte for byte. From the original 24.390-second CPU baseline, the combined exact changes are **34.9% faster** (`1.537×`).


## Native exact visibility and mask fusion, 12 September 2026

The exact CPU collector uses a separate stateless C++ library, when available, to fuse mixed-precision visibility decisions and the four-pixel mask test. The native call borrows the existing geometry and compact projection arrays; it creates no geometry copy and performs no nested threading. Float32 ray, denominator and incidence arithmetic, double plane/hit/patch arithmetic, float32 relative depth tolerance and chronological NumPy ranking remain unchanged. Floating-point contraction is disabled. The NumPy implementation remains the fallback when the library is absent, and the selected library is included in run provenance and resume change detection.

Reliability short-circuits are exact: a denominator at or below the float32 `0.15` threshold cannot produce a reliable patch, and a plane depth at or below `0.1` cannot pass reliability. The frozen scan skipped 7,499,664 of 155,575,674 plane calculations at the denominator gate; its conditioned plane-depth gate skipped none. The main gain comes from fused scalar work and eliminating NumPy geometry-gather/ray/hit/patch/mask temporaries rather than from this 4.8% shortcut alone.

| Collector | Candidate-stage wall samples | Median wall |
|---|---|---:|
| Valid-only NumPy visibility | 15.869 s, 15.972 s, 15.790 s | 15.869 s |
| Native visibility and mask | 12.981 s, 12.834 s, 13.018 s | 12.981 s |

The native fusion is **18.2% faster** than the preceding checkpoint (`1.222×`). Profiling reduced visibility/color wall time from 5.249 s to 2.170 s and summed decision worker time from 18.884 s to 3.158 s. A final run after ABI hardening measured 12.863 s and remained byte-identical. Photo-loop sampled RSS was 1.95–2.04 GB; whole-stage sampled peaks varied with final file-backed materialization and are not used to claim an RSS improvement. Releasing the native wrapper before finalization is important because it owns references to both 12-byte-per-point geometry arrays.

Validation includes exact float32 incidence bits, threshold neighbors around positive and negative `0.15`, translated centers, unusual non-byte masks, concurrent calls, native-versus-NumPy collector integration and four byte-for-byte full frozen-scan comparisons. The combined median improvement from the original 24.390-second CPU baseline is **46.8%** (`1.879×`).


## Native chronological rank insertion, 12 September 2026

After NumPy computes the unchanged float32 score, the native CPU library now scans the point's four stored scores, selects the first strict minimum and writes accepted `u`, `v`, photo ID and score directly into the canonical record. It preserves the chronological `new_score > minimum` rule and disjoint per-photo point ownership. This removes the `N×4` advanced-index score copy, slot/take arrays, accepted-value copies and four separate NumPy scatter assignments without changing transcendental arithmetic.

| Collector | Candidate-stage wall samples | Median wall |
|---|---|---:|
| Native visibility/mask only | 12.981 s, 12.834 s, 13.018 s | 12.981 s |
| Native rank insertion | 11.918 s, 12.096 s, 11.996 s | 11.996 s |

This is a further **7.59% wall-time reduction** (`1.082×`). Profiled color wall time fell from 2.170 s to 1.521 s; summed rank worker time fell from 4.832 s to 1.736 s and the separate scatter timer fell from 1.652 s to zero. All three observation files match the frozen golden file byte for byte. Tests cover first-minimum ties, overwrite history, infinity and NaN inputs, exact insertion counts and untouched fields. The combined median improvement from the original 24.390-second CPU baseline is **50.8%** (`2.033×`).


## Native final photo buckets, 12 September 2026

Final occupied candidate slots are now grouped by photo with one ascending native scan of points and their four slots. Each occupied flat slot ID is appended at its photo's prefix cursor. This preserves the previous stable ascending slot order within every photo while removing per-chunk occupied arrays, stable photo sorts, boundary searches and Python scatter loops. Cursor bounds and final counts are checked, and both uint32 and uint64 slot-index mappings remain supported. The NumPy implementation remains the no-library fallback.

| Collector | Candidate-stage wall samples | Median wall | Median sampled RSS |
|---|---|---:|---:|
| Native rank insertion | 11.918 s, 12.096 s, 11.996 s | 11.996 s | 2.720 GB |
| Native photo buckets | 11.215 s, 11.206 s, 11.367 s | 11.215 s | 2.490 GB |

This is another **6.51% wall-time reduction** (`1.070×`). The profiled bucket phase fell from 0.750 s to 0.045 s. All three full observation files matched the frozen golden file byte for byte, and bounded-finalizer tests exercise both native and fallback grouping. The combined median improvement from the original 24.390-second CPU baseline is **54.0%** (`2.175×`).
