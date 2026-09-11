"""Unprivileged host metadata and process samples; unavailable GPU counters stay null."""

import os
import platform
import subprocess

import psutil


def hardware():
    def sysctl(key):
        try:
            return subprocess.check_output(
                ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": sysctl("machdep.cpu.brand_string")
        if platform.system() == "Darwin"
        else platform.processor(),
        "logical_cpu_cores": os.cpu_count(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "available_memory_bytes": psutil.virtual_memory().available,
        "gpu_utilization_percent": None,
        "gpu_note": "Metal command timings are reported by native stages. System GPU utilization is unavailable without an additional platform collector.",
    }


def sample(process):
    rss = 0
    cpu = 0
    for p in [process] + process.children(recursive=True):
        try:
            rss += p.memory_info().rss
            t = p.cpu_times()
            cpu += t.user + t.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return {
        "rss_bytes": rss,
        "cpu_seconds": cpu,
        "system_available_memory_bytes": psutil.virtual_memory().available,
        "gpu_utilization_percent": None,
    }
