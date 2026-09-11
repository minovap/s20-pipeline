"""Stage events, cancellation, memory limits and content-checked resume."""

import fcntl
import json
import os
import resource
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


def resume_identity(job):
    """Everything that determines outputs. Resource limits only change speed and may differ on resume."""
    options = {k: v for k, v in job["options"].items() if k not in RESOURCE_OPTIONS}
    return {**job, "options": options}


def _run(job, resume=False):
    out = Path(job["output"])
    receipt = out / "job.json"
    if resume:
        previous = json.loads(receipt.read_text()) if receipt.is_file() else None
        if previous is None or resume_identity(previous) != resume_identity(job):
            raise ValueError("Resume requires identical inputs, options, code and binaries")
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
    env = os.environ.copy()
    env.update(
        {
            "OPENBLAS_NUM_THREADS": str(min(4, job["options"]["cpu_threads"])),
            "VECLIB_MAXIMUM_THREADS": str(min(4, job["options"]["cpu_threads"])),
            "OMP_NUM_THREADS": str(job["options"]["cpu_threads"]),
            "S20_CPU_THREADS": str(job["options"]["cpu_threads"]),
        }
    )
    try:
        for stage in stage_names(job):
            dest = out / stage
            report_path = out / "receipts" / f"{stage}.json"
            if resume and report_path.exists():
                report = json.loads(report_path.read_text())
                if not dest.is_dir() or report["outputs"] != files_identity(dest):
                    raise ValueError(f"Completed stage changed: {stage}. Use a fresh run.")
                emit("stage_cached", stage=stage)
                continue
            if dest.exists():
                failed = out / "incomplete"
                failed.mkdir(exist_ok=True)
                dest.rename(failed / f"{stage}-{uuid.uuid4().hex[:8]}")
            state.update(stage=stage)
            save()
            emit("stage_started", stage=stage)
            usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
            tick = time.monotonic()
            peak = 0
            cpu = 0
            last_sample = None
            log_path = out / "logs" / f"{stage}.log"
            with log_path.open("w") as log:
                child = subprocess.Popen(
                    [sys.executable, "-m", "s20_pipeline.worker", stage, str(receipt)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                proc = psutil.Process(child.pid)
                cursor = 0
                try:
                    while child.poll() is None:
                        try:
                            value = sample(proc)
                            peak = max(peak, value["rss_bytes"])
                            cpu = max(cpu, value["cpu_seconds"])
                            now = time.monotonic()
                            value["cpu_core_equivalents"] = (
                                (
                                    max(0, value["cpu_seconds"] - last_sample[1])
                                    / (now - last_sample[0])
                                )
                                if last_sample
                                else None
                            )
                            last_sample = (now, value["cpu_seconds"])
                            emit("resources", stage=stage, **value)
                            if value["rss_bytes"] > job["options"]["memory_gb"] * 1e9:
                                raise MemoryError(f"{stage} exceeded configured process RSS budget")
                        except psutil.NoSuchProcess:
                            pass
                        with log_path.open() as reader:
                            reader.seek(cursor)
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
                            cursor = reader.tell()
                        time.sleep(0.5)
                except BaseException:
                    stop(child)
                    raise
                if child.returncode:
                    raise RuntimeError(f"{stage} exited {child.returncode}; see {log_path}")
            # Keep hashing outside processing wall timing.
            wall = time.monotonic() - tick
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu_exact = (
                usage_after.ru_utime
                + usage_after.ru_stime
                - usage_before.ru_utime
                - usage_before.ru_stime
            )
            report = {
                "stage": stage,
                "wall_s": wall,
                "sampled_peak_tree_rss_bytes": peak,
                "cpu_seconds": cpu_exact,
                "average_cpu_cores": cpu_exact / wall,
                "sampling_interval_s": 0.5,
                "gpu_utilization_percent": None,
                "outputs": files_identity(dest),
            }
            if (dest / "metal.json").exists():
                report["metal"] = json.loads((dest / "metal.json").read_text())
            atomic_json(report_path, report)
            emit(
                "stage_completed",
                stage=stage,
                **{k: v for k, v in report.items() if k not in ("stage", "outputs")},
            )
        state.update(status="completed")
        save()
        emit("completed", output=str(out))
    except BaseException as error:
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
