# S20 Studio desktop

A working Tauri 2 + React desktop app around the native pipeline. It opens raw capture folders, shows bag/device metadata and a provisional time estimate, runs configurable processing, streams stage progress/resources, cancels and resumes jobs, and displays bounded point-cloud previews with a linked-camera reference pane.

## Start locally

Install/build the engine first, following the repository README. Then:

```sh
cd apps/desktop
npm ci
npm run desktop
```

To build a local macOS application:

```sh
npm run tauri -- build --debug --bundles app
```

The app bundle is under `src-tauri/target/debug/bundle/macos/S20 Studio.app`. It currently uses the repository's `.venv` and `build` directory. Engine settings can select a different installed checkout. **This is a development application for this Mac, not a self-contained signed distribution.** Python, native executables and model weights are not yet bundled inside the `.app`.

Use **Choose raw capture** to select `/Volumes/SD_CARD/Test`, then choose a separate output parent. A new child folder is generated for every run. Select Max to use the throughput profile. Cancel preserves completed stages; Resume last run verifies the original options/inputs/code before reusing them. A changed pipeline requires a fresh run.

## Inspector

Open an uncompressed LAS or binary PLY. The Python bridge streams deterministic samples into a cached binary buffer, with a maximum of two million displayed points per pane. Raw LAS coordinates are rebased before converting to float32. File data crosses IPC as binary, not millions of JSON records. Photo colors are displayed directly through the adapted comparison shader, without a second sRGB conversion.

The renderer adapts the existing Three.js/OrbitControls comparison approach: shared camera, scissored panes and on-demand drawing. Generic bounds replace the indoor hard-coded camera presets. Display sampling leaves exported files untouched. Reference clouds retain their coordinates; the app does not align unrelated coordinate frames automatically.

This first version reads the source cloud to create a sample. It is **not an octree/LOD system**, and generating a preview for a very large cloud can take time. Tiled out-of-core viewing is the next rendering milestone.

## Resource and progress contract

Python/native workers own processing. Rust launches argument arrays (no shell), forwards JSONL events and handles process exit. User Cancel sends SIGINT to the pipeline parent; the pipeline terminates its worker process group. Closing/quitting while processing asks the user to cancel first. Auxiliary inspectors/previews are terminated on exit.

Numeric progress appears when a stage exposes counts; other stages are indeterminate with elapsed time. Resource events show CPU core equivalents, sampled RSS and available memory. GPU command time is distinct from system GPU occupancy, which remains unavailable. Current estimates use the reference pipeline model; they are not calibrated for every option/hardware combination.

## Validation

- The native app was used to inspect the original Test capture and launch the full pipeline.
- Cancel was tested during Metal geometry. Resume reused five completed stages and completed all 14 stages, then loaded a preview of the 6.02-million-point colored result.
- Python tests cover bounded preview size, large-coordinate rebasing, RGB values, cache reuse and preservation of source bytes.
- Frontend tests cover optional-stage selection and missing telemetry values.
- Rust tests cover literal path arguments, option validation and geometry-only/resume flags.

Run `npm test`, `npm run build`, `cargo test --manifest-path src-tauri/Cargo.toml`, and the repository Python test suite. The lockfiles pin npm/Cargo dependencies. The latest local npm audit reported no known vulnerabilities.

Next: package the runtime/weights, add native count events and per-option ETA calibration, preserve jobs across application upgrades, implement tiled LOD, and add a calibrated registration workflow for reference clouds in different frames.
