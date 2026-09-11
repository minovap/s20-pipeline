# S20 Studio desktop

Tauri 2 + React desktop app around the native pipeline. Work is organised in
projects: add raw S20 scan folders to a project, process them, and open the
resulting point clouds in the viewer to slice and export them.

## Start locally

Install and build the engine first, following the repository README. Then:

```sh
cd apps/desktop
npm ci
npm run desktop
```

`start.command` does the same from Finder. To build a local macOS application:

```sh
npm run tauri -- build --debug --bundles app
```

The app uses the repository's `.venv` and `build` directory as its engine. A
different checkout can be chosen in Settings. This is a development build for
this Mac: Python, native binaries and model weights are not bundled.

## Workflow

**Projects.** Projects live as folders under a projects root (default
`~/Documents/S20 Projects`, changeable in Settings). Each holds `project.json`,
`runs/` and `exports/`. Every run is an ordinary pipeline output directory, so
the app reconstructs run history from the pipeline's own `job.json`,
`state.json`, `receipts/` and `events.jsonl`.

**Scans.** A scan is a raw capture folder referenced in place. It is inspected
once when added and only ever read. If the drive is unmounted the scan shows a
warning and cannot be processed until it is back.

**Runs.** New run shows four choices (result type, remove people, exposure,
performance) plus memory limit and pose refinement under Advanced. While a run
processes, the steps are listed like a build pipeline: done steps show their
time, the running step shows a live counter and the list keeps it centred,
the header shows the total elapsed time. A failed or cancelled run shows the
error under the step, can reveal that step's log, and can be resumed.

**Viewer.** The canvas fills the window. Perspective or axis-locked
orthographic views (Top, Front, Side, with a flip and a scale bar). Keys 7, 1,
3 and 5 switch views, `f` fits, `s` toggles slicing.

**Slices.** In an axis view, press Slice and drag a rectangle. The rectangle
becomes an axis-aligned box spanning the parent's full depth; a slice of a
slice is the intersection, so the Top view cuts a footprint and the Front view
then trims its height. Slices nest under their source cloud in the right-hand
list. Checked items are shown, and shown is what gets exported. The panel
edits the selected slice's bounds numerically. Slices are stored in
`project.json`.

**Export.** "Export checked" writes a LAS file into the project's `exports/`
folder, reading the full-resolution source, not the preview. One checked item
asks for a file name. Several ask whether to write one combined file (a point
inside more than one slice is written once) or separate files. Right-click a
slice for Export, Rename, Slice from here and Delete. Exports reappear in the
list as point clouds.

## Review without Tauri

`npm run dev` then open `http://127.0.0.1:1420/?mock&screen=projects` (or
`screen=project`, `project&select=new`, `running`, `viewer&view=top`) to review
the UI in a browser against fake data.

## Tests

```sh
npm test                                   # stage selection, durations, slice geometry
npm run build                              # type check and bundle
cargo test --manifest-path src-tauri/Cargo.toml   # argument building, run reconstruction
../../.venv/bin/python -m pytest -q ../../tests/test_desktop.py   # previews and slice export
```
