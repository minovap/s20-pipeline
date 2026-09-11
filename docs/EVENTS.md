# Worker and event contract, schema 1

The public process entry points are `s20 inspect`, `s20 estimate`, `s20 run`, and `s20 colorize`. `s20 build` provisions native executables. Internal `python -m s20_pipeline.worker STAGE JOB_JSON` is an implementation detail: launch the public CLI so it applies path protection, locks, provenance and failure handling.

`inspect` and `estimate` emit one JSON object. Processing stdout is newline-delimited JSON; diagnostics are on stderr. Persisted copies are in `events.jsonl`. The app must ignore unknown fields/events for forward compatibility.

```json
{"schema":1,"event":"stage_started","time_unix":1789139700.0,"stage":"candidates"}
{"schema":1,"event":"progress","time_unix":1789139701.0,"stage":"candidates","done":12,"total":62}
{"schema":1,"event":"resources","time_unix":1789139701.2,"stage":"candidates","rss_bytes":2600000000,"cpu_seconds":4.8,"cpu_core_equivalents":2.1,"system_available_memory_bytes":28000000000,"gpu_utilization_percent":null}
```

Other events: `stage_completed`, `stage_cached`, `completed`, `failed`, `cancelled`. Terminal stage events carry wall time, OS child CPU seconds, average core equivalents and sampled peak process-tree RSS. Metal blend receipts additionally contain hardware command time, device name and buffer allocation. Detailed mask and geometry receipts live inside their stage folders. CPU seconds are not percentages: 8 CPU seconds over 2 seconds wall means 4 average core equivalents.

`progress` currently reports photo or point counts for extraction, masks, candidate collection and CPU blending. Native tracking/geometry still expose their detailed counts in logs. Show indeterminate progress for those stages until a native event bridge is added. A completed stage is complete even if its last short-lived progress record fell between polling reads.

Run data:

- `job.json`: effective flags, external input hashes, Python/native source identities and executable hashes; used for resume equality.
- `hardware.json`: OS, CPU, core counts and memory. GPU utilization is explicitly unavailable.
- `state.json`: atomically updated status/current stage/error.
- `receipts/STAGE.json`: elapsed/resources and hashes of completed stage outputs.
- `logs/STAGE.log`: worker stdout/stderr, including native messages and tracebacks.
- `incomplete/`: retained failed/interrupted stage files when resuming.

One process owns `.run.lock`. Ctrl-C cancels the worker's entire process group. The memory ceiling is sampled process-tree RSS; it can overshoot briefly and cannot account for every Metal driver allocation. A production app should also monitor system memory pressure and storage. No auto-resume after a source/config/code change: start a new run so results remain attributable.

Counts and metadata are not trusted geometry. The camera record schema contains a proper camera-to-world rotation, a camera center, calibrated fisheye parameters and a raw-derived timestamp. Explicit camera convention selection plus timestamp overlap is a precondition, not an independent optical calibration proof.
