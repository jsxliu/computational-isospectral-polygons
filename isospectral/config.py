# This file keeps parallel searches from overwhelming the computer. It gives
# each worker one numerical-library thread by default, chooses the appropriate
# multiprocessing method for the operating system, checks available memory,
# and recommends a worker count based on both CPU and RAM.

import multiprocessing as mp
import os
import subprocess
import sys


OS_CPU = os.cpu_count() or 1

# Default numerical libraries to one thread per worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from numba import njit  # noqa: E402, F401 (ignore two warnings).


# Choose how worker processes start on this operating system.
def best_mp_context():
    if os.name == "nt" or sys.platform == "darwin":
        return mp.get_context("spawn")
    return mp.get_context("fork")


# Provide settings for worker pools.
def pool_kwargs():
    return {"mp_context": best_mp_context()}


# Read RAM in GiB and use available RAM where possible.
def get_available_memory_gib():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", encoding="utf-8") as meminfo:
                values = {
                    key.strip(): value.strip()
                    for key, value in (
                        line.split(":", 1) for line in meminfo if ":" in line
                    )
                }
            for key in ("MemAvailable", "MemTotal"):
                if key in values:
                    return int(values[key].split()[0]) / (1024 * 1024)

        if sys.platform == "darwin":
            total = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], timeout=2,
            )
            return int(total.strip()) / (1024 ** 3)

        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.available_physical / (1024 ** 3)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


# Choose a worker count based on CPU count and RAM.
def recommended_workers(*, reserve_gib=2.0, per_worker_gib=0.6):
    cpu_budget = max(1, OS_CPU - 1)
    ram_gib = get_available_memory_gib()
    if ram_gib is None:
        return cpu_budget
    ram_budget = max(1, int((ram_gib - reserve_gib) / per_worker_gib))
    return min(cpu_budget, ram_budget)
