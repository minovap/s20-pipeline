"""Stage events, cancellation, memory limits and content-checked resume."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

from .storage import atomic_json, digest
from .telemetry import hardware, sample


def stage_names(job):
    names = []
    if job["mode"] == "run":
        names = ["decode", "pack", "tracking"]
        if job["options"]["pose_refinement"]:
            names += ["pose_refinement"]
        names += ["registered", "geometry"]
        if not job["options"]["color"]:
            return names
        names += ["photos", "cameras"]
    if job["options"]["mask"] == "person" and not job.get("masks"):
        names += ["masks"]
    names += ["candidates"]
    if job["options"]["exposure"] != "off":
        names += ["global"]
    if job["options"]["exposure"] == "local":
        names += ["local"]
    return names + ["blend", "export"]


def files_identity(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def stop(child):
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


RESOURCE_OPTIONS = ("memory_gb", "cpu_threads", "color_workers", "resources", "chunk_points")
CODE_KEYS = ("python_code", "native_code", "source_identities")


def resume_identity(job):
    """Inputs and options that determine outputs. Resource limits only change speed and may differ on resume."""
    options = {k: v for k, v in job["options"].items() if k not in RESOURCE_OPTIONS}
    inputs = {
        k: v
        for k, v in job.get("source_identities", {}).items()
        if not k.endswith(("s20_reconstruct", "s20_refine_poses", "s20_geometry", "s20_blend"))
    }
    return {
        **{k: v for k, v in job.items() if k not in CODE_KEYS},
        "options": options,
        "inputs": inputs,
    }


def code_changes(previous, job):
    """Code or binaries that differ from the run's first attempt. Completed stage outputs are still verified by hash."""
    changed = []
    for key in ("python_code", "native_code"):
        before, after = previous.get(key, {}), job.get(key, {})
        changed += sorted(f for f in set(before) | set(after) if before.get(f) != after.get(f))
    before, after = previous.get("source_identities", {}), job.get("source_identities", {})
    changed += sorted(
        Path(f).name
        for f in set(before) | set(after)
        if f.endswith(("s20_reconstruct", "s20_refine_poses", "s20_geometry", "s20_blend"))
        and before.get(f) != after.get(f)
    )
    return changed


DEPENDENCIES = {
    "decode": [],
    "pack": ["decode"],
    "tracking": ["pack"],
    "pose_refinement": ["tracking"],
    "registered": ["tracking", "pose_refinement"],
    "geometry": ["registered"],
    "photos": [],
    "cameras": ["photos", "registered"],
    "masks": ["photos"],
    "candidates": ["geometry", "cameras", "masks"],
    "global": ["candidates"],
    "local": ["global"],
    "blend": ["candidates", "global", "local"],
    "export": ["blend"],
}
GEOMETRY_LANE = {"decode", "pack", "tracking", "pose_refinement", "registered", "geometry"}


def lane(stage):
    return "geometry" if stage in GEOMETRY_LANE else "color"


def dependencies(stage, names):
    return [d for d in DEPENDENCIES.get(stage, []) if d in names]


def ready(names, done, running):
    """Stages that can start now: inputs finished and their lane idle. One per lane, in pipeline order."""
    busy = {lane(s) for s in running}
    out = []
    for stage in names:
        if stage in done or stage in running or lane(stage) in busy:
            continue
        if all(d in done for d in dependencies(stage, names)):
            out.append(stage)
            busy.add(lane(stage))
    return out


def blocked(names, done, running):
    """First unfinished stage of each idle lane together with the inputs it still needs."""
    busy = {lane(s) for s in running}
    out = {}
    for stage in names:
        if stage in done or stage in running or lane(stage) in busy or lane(stage) in out:
            continue
        missing = [d for d in dependencies(stage, names) if d not in done]
        if missing:
            out[lane(stage)] = (stage, missing)
    return out.values()


def _run(job, resume=False):
    out = Path(job["output"])
    receipt = out / "job.json"
    changed_code = []
    if resume:
        previous = json.loads(receipt.read_text()) if receipt.is_file() else None
        if previous is None or resume_identity(previous) != resume_identity(job):
            raise ValueError("Resume requires identical inputs and options")
        changed_code = code_changes(previous, job)
        atomic_json(receipt, job)
    else:
        atomic_json(receipt, job)
    (out / "receipts").mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    atomic_json(out / "hardware.json", hardware())

    def emit(event, **data):
        row = {"schema": 1, "event": event, "time_unix": time.time(), **data}
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (out / "events.jsonl").open("a") as stream:
            stream.write(line + "\n")

    state = {"status": "running", "stage": None}

    def save():
        atomic_json(out / "state.json", state)

    save()
    if changed_code:
        # Finished stages keep their verified outputs; remaining stages run with the new code.
        emit("resume_notice", changed=changed_code)
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": str(min(4, job["options"]["cpu_threads"])),
            "VECLIB_MAXIMUM_THREADS": str(min(4, job["options"]["cpu_threads"])),
            "OMP_NUM_THREADS": str(job["options"]["cpu_threads"]),
            "S20_CPU_THREADS": str(job["options"]["cpu_threads"]),
        }
    )
    names = stage_names(job)
    done = set()
    running = {}
    waited = set()

    def start(stage):
        dest = out / stage
        if dest.exists():
            failed = out / "incomplete"
            failed.mkdir(exist_ok=True)
            dest.rename(failed / f"{stage}-{uuid.uuid4().hex[:8]}")
        state.update(stage=stage)
        save()
        emit("stage_started", stage=stage)
        log_path = out / "logs" / f"{stage}.log"
        log = log_path.open("w")
        child = subprocess.Popen(
            [sys.executable, "-m", "s20_pipeline.worker", stage, str(receipt)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        running[stage] = {
            "child": child,
            "log": log,
            "log_path": log_path,
            "proc": psutil.Process(child.pid),
            "tick": time.monotonic(),
            "peak": 0,
            "cpu_seconds": 0.0,
            "last_sample": None,
            "cursor": 0,
        }

    def finish(stage, item):
        item["log"].close()
        child = item["child"]
        if child.returncode:
            raise RuntimeError(f"{stage} exited {child.returncode}; see {item['log_path']}")
        # Keep hashing outside processing wall timing.
        wall = time.monotonic() - item["tick"]
        dest = out / stage
        report = {
            "stage": stage,
            "wall_s": wall,
            "sampled_peak_tree_rss_bytes": item["peak"],
            "cpu_seconds": item["cpu_seconds"],
            "average_cpu_cores": item["cpu_seconds"] / wall if wall else None,
            "sampling_interval_s": 0.5,
            "gpu_utilization_percent": None,
            "outputs": files_identity(dest),
        }
        if (dest / "metal.json").exists():
            report["metal"] = json.loads((dest / "metal.json").read_text())
        atomic_json(out / "receipts" / f"{stage}.json", report)
        emit(
            "stage_completed",
            stage=stage,
            **{k: v for k, v in report.items() if k not in ("stage", "outputs")},
        )
        done.add(stage)

    try:
        while True:
            for stage in ready(names, done, running):
                report_path = out / "receipts" / f"{stage}.json"
                dest = out / stage
                if resume and report_path.exists():
                    report = json.loads(report_path.read_text())
                    if not dest.is_dir() or report["outputs"] != files_identity(dest):
                        raise ValueError(f"Completed stage changed: {stage}. Use a fresh run.")
                    emit("stage_cached", stage=stage)
                    done.add(stage)
                    break  # re-evaluate readiness with the cached stage counted
                start(stage)
            else:
                if not running and len(done) == len(names):
                    break
                for stage, missing in blocked(names, done, running):
                    if stage not in waited:
                        waited.add(stage)
                        emit("stage_waiting", stage=stage, waiting_for=missing)
                for stage, item in list(running.items()):
                    child = item["child"]
                    try:
                        value = sample(item["proc"])
                        item["peak"] = max(item["peak"], value["rss_bytes"])
                        item["cpu_seconds"] = max(item["cpu_seconds"], value["cpu_seconds"])
                        now = time.monotonic()
                        last = item["last_sample"]
                        value["cpu_core_equivalents"] = (
                            max(0, value["cpu_seconds"] - last[1]) / (now - last[0])
                            if last
                            else None
                        )
                        item["last_sample"] = (now, value["cpu_seconds"])
                        emit("resources", stage=stage, **value)
                        if value["rss_bytes"] > job["options"]["memory_gb"] * 1e9:
                            state.update(stage=stage)
                            raise MemoryError(f"{stage} exceeded configured process RSS budget")
                    except psutil.NoSuchProcess:
                        pass
                    with item["log_path"].open() as reader:
                        reader.seek(item["cursor"])
                        for line in reader:
                            try:
                                value = json.loads(line)
                                if value.get("event") == "progress":
                                    emit(
                                        "progress",
                                        stage=stage,
                                        done=value["done"],
                                        total=value["total"],
                                    )
                            except (ValueError, AttributeError):
                                pass
                        item["cursor"] = reader.tell()
                    if child.poll() is not None:
                        del running[stage]
                        state.update(stage=stage)
                        finish(stage, item)
                if running:
                    time.sleep(0.5)
                continue
            continue
        state.update(status="completed")
        save()
        emit("completed", output=str(out))
    except BaseException as error:
        for item in running.values():
            stop(item["child"])
            item["log"].close()
        state.update(
            status="cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        save()
        emit(state["status"], stage=state["stage"], message=str(error))
        raise


def run(job, resume=False):
    out = Path(job["output"])
    out.mkdir(parents=True, exist_ok=resume)
    with (out / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this run")
        try:
            return _run(job, resume)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
