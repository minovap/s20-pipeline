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

Current candidate storage is 128 bytes per geometry point. At 5.1 million points it is about 652 MB; at 20× it is about 13.0 GB. Geometry, normals, full-image projection and temporary arrays add memory. Disk-backed candidates bound one allocation, not the entire pipeline.

More seriously, projection is currently O(points × selected photos). If both grow 5×, the work can grow 25×; at 20×, 400×. Do not extrapolate the 25.20-second color stage linearly or promise that more cores solve this.

Prioritized improvements:

1. **Spatial photo selection.** Build a camera/scene spatial index and conservative fisheye view bounds. Consider only cameras that can see each tile. Preserve occluders outside the tile through full-frustum depth or an appropriate halo; a color chunk alone cannot define visibility. Validate candidate identity/quality on occlusion boundaries.
2. **GPU projection and depth reduction.** Project points, reject invalid/masked pixels, build nearest-depth buffers and rank views in Metal. These independent point operations are the main remaining acceleration opportunity. Preserve deterministic packed depth/ID ties; query integer-atomic capabilities, or use a two-pass minimum reduction if required. Compare all records to CPU references.
3. **Streaming geometry and observations.** Decode and process frame windows, spill observation tiles, avoid loading the entire raw recording into the C++ front end. Bound map/submap lifetime. Existing tracking is sequential across time; parallelize registration residuals and local reductions rather than independent time chunks with no pose continuity.
4. **Overlap useful work.** Decode/transform the next photo while inference runs, pipeline file reads with CPU work, and process independent pose residuals/geometry blocks in parallel. Budget all work against unified memory; benchmark with and without overlap.
5. **Bound exposure fitting.** Stratified samples per overlap edge and connected-component handling; do not let graph rows grow without a cap. Preserve held-out quality evaluation. The sparse solve is not currently the bottleneck, so a GPU port is lower priority.
6. **Stream GPU blending and export.** Chunk the Metal host's buffers and LAS export; GPU blend is already short and should follow projection work in priority. The new LAS writer already bounds export temporaries. Add LAZ only with an explicit supported compressor and measured CPU tradeoff.
7. **LOD rendering.** Shared tile cache and point budget across comparison panes; suspend unnecessary rendering while full processing runs.

## Measurement contract

Each run saves immutable input identities, configuration and native/source identities; per-stage wall time, OS-reported child CPU time, sampled process-tree RSS, and explicit GPU measurements where available. Wall times include worker startup and the orchestration polling interval, but exclude completed-output hashing. Source hashing/import preflight occurs before processing stage events and must be included separately when reporting user-visible end-to-end time.

CPU core equivalents = CPU seconds / wall seconds. Per-process RSS is not unified GPU usage. Metal allocation overlaps mapped/unified memory, and MPS synchronized forward wall time is not hardware GPU command duration. No tool in this package currently reports system GPU occupancy. Missing values are null.

Large-scan release acceptance must cover at least 5× and 20× captures, image-pair rejection accuracy, memory pressure, cancellation and restart. Estimate intervals should widen when selected points/photos, memory pressure or computer hardware differ from the measured reference. The CLI's current range is an explicit heuristic, not a statistical interval.
