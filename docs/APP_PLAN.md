# Desktop app plan

Start with a local macOS desktop app around this pipeline. The UI manages jobs; separate Python/C++ workers own computation. Reuse the existing React/Three.js comparison view for the first app, then add tiled point-cloud storage and level of detail before advertising 20× scan support.

## Framework choice

| Framework | Fit with our code | Point-cloud view | Main tradeoff |
|---|---|---|---|
| **Tauri 2 + React + Three.js/Potree** | Reuses the current React viewer. Rust shell owns file access, process lifecycle and events; packaged Python/C++ worker runs as a sidecar. | Existing Three.js `Points` for small clouds; octree/LOD viewer for large outputs. | Best first implementation given existing UI. Uses platform WebViews, so validate macOS renderer performance and IPC paths on actual hardware. |
| **SwiftUI + MetalKit** | Native file picker, macOS integration and a Swift process bridge to the same worker contract. | Custom `MTKView` renderer with GPU culling and a tiled source. | Best Mac-only rendering/control option; requires a new renderer and UI rather than directly reusing React. |
| **Electron + React + Three.js/Potree** | Maximum reuse of web UI and mature subprocess integration. | Predictable bundled Chromium renderer plus LOD. | Larger application/runtime memory footprint, competing with compute for unified memory. |
| **Qt 6 Quick + C++** | Direct integration with native workers; Python can remain a separate process. | Custom Quick 3D geometry, or a dedicated point renderer. | Strong native/cross-platform option, but adds QML/C++ UI work and a separate packaging stack. |

Recommendation: **Tauri + React for the first app**, keeping the job/event API independent of the shell. If the product is permanently Mac-only and maximum renderer control is the priority, choose SwiftUI + MetalKit instead. A cross-platform shell does not make the current Metal/MPS backend cross-platform.

Official references: [Tauri sidecars](https://v2.tauri.app/develop/sidecar/), [Three.js Points](https://threejs.org/docs/pages/Points.html), [Potree](https://github.com/potree/potree), [Apple MTKView](https://developer.apple.com/documentation/metalkit/mtkview), [Electron process model](https://www.electronjs.org/docs/latest/tutorial/process-model), [Qt Quick 3D geometry](https://doc.qt.io/qt-6/qquick3dgeometry.html). Framework recommendations above are engineering judgments based on our existing code and these capabilities.

## Relative implementation effort

For this codebase, using the first Tauri app as a 1× baseline:

| Route | Rough effort | Main work |
|---|---:|---|
| Tauri + existing React/Three.js | 1× | Native shell, worker packaging, job screens, event bridge and file access. |
| SwiftUI + embedded existing web viewer | 1.2–1.5× | Rebuild controls in SwiftUI and bridge the web viewer to the native job model. |
| SwiftUI + native Metal viewer | 2–3× | Above plus a renderer, camera controls, picking, comparison panes, LOD and GPU cache management. |

These are engineering scope estimates, not measured development times or a schedule commitment. Basic forms and progress are straightforward in either framework; renderer replacement drives the difference. Large-scan pipeline improvements, tiled cloud storage, correctness validation and distribution work are shared costs. Both shells run the same native processing engine, so choosing SwiftUI alone does not accelerate reconstruction. The multiplier decreases if shared backend work dominates the project.

## Screen structure

```text
┌ Capture: Test                     [Open capture folder…] ┐
│ S20 · MID-360 · 69.36 s bag · 60.5 s device metadata     │
│ 583 MB bag · 671 LiDAR frames · 13,414 IMU · 64 photos   │
│ Left / Right: 3504×4672 · calibration present            │
│ Clock: needs preflight · local coordinates              │
├ Output ──────────────────────── Resources ───────────────┤
│ Folder […]  Geometry + color   Interactive/Balanced/Max  │
│ Exposure: Local                CPU threads [8]          │
│ Person masks [on]              Color workers [4]        │
│ Pose correction [on]           Memory limit [16 GB]     │
│ Advanced ▸ frame range, CPU reference blend, masks off   │
│ Estimate: range + confidence + assumptions [details]     │
│                                  [Start full pipeline] │
├ Steps ──────────────────────────────────────────────────┤
│ ✓ Decode   ✓ Pack   ▶ Track 320/670   ○ Pose correction  │
│ ○ Metal geometry   ○ Photos   ○ Cameras   ○ Masks       │
│ ○ Visibility   ○ Exposure   ○ Blend   ○ Export           │
│ CPU 2.1 core equivalents · RSS 2.5 GB · Metal 0.7 GB     │
│ GPU busy: unavailable · GPU command time: 15 ms          │
│                          [Cancel] [Open logs]            │
├ Result / Compare ───────────────────────────────────────┤
│ Point budget [2 M]  Linked cameras [on]  Reference […]    │
│ Native output                    Studio reference       │
└─────────────────────────────────────────────────────────┘
```

This is a layout proposal. Progress and telemetry values illustrate the presentation; the current CLI emits numeric progress only where the stage exposes counts. Other stages initially show elapsed time and an indeterminate indicator, never fabricated completion percentages.

## Capture import and preflight

1. Native folder picker selects the camera capture. Inspect bag index and metadata without extracting all payloads. Show actual file bytes, bag duration and device-reported duration separately. Display sensor model, topic counts, photo count and calibrated resolutions; point count becomes known after raw decoding.
2. Check supported layout, calibration matrices, timestamp overlap, source readability, output separation, output disk space, backend availability and model-weight availability. Keep sources read-only, including SD cards. Do not infer calibration from a filename/date or reuse a different recording's clock map.
3. Show image coverage and rejected-pose-gap counts once cameras are prepared. Offer an image-on-geometry inspection before accepting a new camera convention. The original garden recording is not covered by the indoor optical validation.
4. Display GNSS/RTK availability as metadata, without labelling native local coordinates surveyed. Provide a separate future registration workflow.

## Jobs and progress

The Rust/Swift/Qt shell launches a worker with an argument array, not an interpolated shell command. Parse the versioned JSONL event stream in [EVENTS.md](EVENTS.md). Keep stdout/stderr logs on disk and limit UI log retention. Emit stage status, elapsed time, counts when known, per-stage resource history and cache decisions. Show total ETA as the sum of remaining stage estimates; update rates as processing proceeds.

Initial release implements one processing job at a time. Pause means finish the current atomic stage and wait before the next; immediate cancel terminates the process group and preserves incomplete work. Resume verifies source/configuration/code/output hashes. Persist the queue in the app, not in UI component state. The CLI already supports cancel/resume; stage-boundary pause and a persistent multi-job queue remain app work.

## Resource controls

- **Interactive:** fewer CPU workers, keep headroom for the desktop and viewer; lower preview point budget/frame rate during compute.
- **Balanced:** conservative CPU workers and memory limit; default for unknown machines.
- **Maximum throughput:** use all available tracking/pose CPU threads and up to eight color workers by default, with explicit overrides. The measured collector used only about 2.14 average CPU cores despite eight workers because depth projection remains serial.
- Do not present a guarantee of “100% GPU” or “100% CPU”. No artificial load or busy waiting. Metal already dispatches parallel work when available; its kernels can finish while the CPU is still preparing input.
- A future **GPU throughput** option can batch masks, overlap transfer/preprocessing with inference, and queue independent GPU blocks within a unified-memory budget. Implement and measure these first; do not add a switch with no actual effect.
- Keep CPU worker pools, BLAS and model threads within one resource policy to avoid nested oversubscription. On unified memory, RSS and Metal buffer allocation overlap; do not add them together and call the sum total usage.

Telemetry shows process CPU core equivalents, system CPU percentage if collected, RSS/peak RSS, system available memory and observed Metal allocation. Display hardware GPU command duration separately from synchronized MPS forward wall time. System GPU utilization stays “unavailable” unless a supported collector is installed; never show a missing measurement as 0%.

## Time estimation

Use the staged model in [PERFORMANCE.md](PERFORMANCE.md). Start with a range, explain uncalibrated I/O and hardware, and replace estimates with measured local rates. Features: raw bytes, LiDAR frames/returns, photo count and resolution, selected point–photo pairs, export bytes, model cache status and memory pressure. Hardware descriptors include chip family, CPU cores, memory and storage; GPU core count alone is not a valid speed multiplier.

The current estimator is intentionally conservative about claims: it provides a numeric local estimate only on the measured 16-core M4 Max family, a reference-machine number elsewhere, and low confidence. It does not yet inspect actual GPU model/core count or storage throughput. Learning from run history and backend-specific GPU probing are app milestones.

## Large-cloud rendering

The current viewer uploads full cloud buffers for each selected result. Replace this with spatial tiles/octree LOD and an on-disk index. Keep a shared scene origin for float precision, bounded GPU/CPU caches, a screen-space point budget, frustum culling and cancellation of obsolete tile requests. Use transferable/binary buffers and allowlisted local asset paths; never send millions of points as JSON over sidecar IPC.

LOD sampling is for display only: exported geometry remains full resolution. Selection, measurement and comparison must retain source IDs and dataset transforms. Synchronized comparison cameras should use the same LOD policy and point size so density differences are not mistaken for spatial errors. Studio output remains an optional reference dataset, not an input to reconstruction.

## Delivery order

1. Tauri shell, capture inspector, options, job lifecycle and event-driven progress; reuse current small-cloud viewer.
2. Package Python/native binaries and pinned weights; signed macOS build, shutdown/cancel/recovery testing and disk-space preflight.
3. Octree/tiled outputs and viewer LOD, then the 5×/10×/20× memory/performance acceptance suite.
4. Spatial photo selection, GPU projection/depth selection, bounded exposure graph and pipelined masks. Update the estimator from measured local stage rates.
