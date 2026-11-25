"""D26: system load logged every 10 s while rgb renders (Windows; no extra dependencies).

Each line of <block>/load_log.jsonl holds:
- UTC time and seconds since the job started;
- CPU % over the last interval (GetSystemTimes);
- RAM used and free in MB (GlobalMemoryStatusEx);
- GPU utilisation % and GPU memory used / total in MiB (nvidia-smi, where available);
- the number of python.exe and node.exe processes (tasklist), to show other jobs on the machine.
A reading that fails is recorded as null with the error, never skipped.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

INTERVAL_S = 10.0


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]


class _MEMSTATUS(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _ft(f: _FILETIME) -> int:
    return (f.hi << 32) | f.lo


def cpu_times() -> tuple[int, int]:
    """(busy, total) in 100 ns units since boot, all processors."""
    idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        raise OSError("GetSystemTimes failed")
    total = _ft(kernel) + _ft(user)  # kernel time includes idle time
    return total - _ft(idle), total


def memory_mb() -> dict[str, float]:
    m = _MEMSTATUS()
    m.dwLength = ctypes.sizeof(_MEMSTATUS)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"ram_used_mb": (m.ullTotalPhys - m.ullAvailPhys) / 2**20, "ram_free_mb": m.ullAvailPhys / 2**20}


def gpu() -> dict[str, float]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    u, used, tot = (float(x) for x in out.split(","))
    return {"gpu_util_pct": u, "gpu_mem_used_mib": used, "gpu_mem_total_mib": tot}


def process_counts() -> dict[str, int]:
    out = subprocess.run(["tasklist", "/fo", "csv", "/nh"], capture_output=True, text=True, timeout=10).stdout.lower()
    names = [ln.split('","')[0].strip('"') for ln in out.splitlines() if ln.strip()]
    return {"python_procs": sum(1 for n in names if n == "python.exe"), "node_procs": sum(1 for n in names if n == "node.exe")}


class LoadLogger:
    """Background thread writing one reading every INTERVAL_S seconds; start() and stop() bracket a job."""

    def __init__(self, path: Path, interval_s: float = INTERVAL_S):
        self.path, self.interval = Path(path), interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n = 0
        self.t0 = time.time()

    def _reading(self, prev: tuple[int, int] | None) -> tuple[dict[str, Any], tuple[int, int] | None]:
        rec: dict[str, Any] = {"utc": datetime.now(timezone.utc).isoformat(), "t_s": round(time.time() - self.t0, 1), "errors": []}
        cur = None
        try:
            cur = cpu_times()
            rec["cpu_pct"] = round(100.0 * (cur[0] - prev[0]) / max(1, cur[1] - prev[1]), 1) if prev else None
        except Exception as exc:  # noqa: BLE001 - recorded
            rec["cpu_pct"] = None
            rec["errors"].append(f"cpu: {exc}")
        for name, fn in (("memory", memory_mb), ("gpu", gpu), ("processes", process_counts)):
            try:
                rec.update(fn())
            except Exception as exc:  # noqa: BLE001 - recorded
                rec["errors"].append(f"{name}: {exc}")
        return rec, cur

    def _run(self) -> None:
        prev = None
        with open(self.path, "a", encoding="utf-8") as f:
            while True:
                rec, prev = self._reading(prev)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self.n += 1
                if self._stop.wait(self.interval):
                    rec, prev = self._reading(prev)
                    rec["final"] = True
                    f.write(json.dumps(rec) + "\n")
                    self.n += 1
                    return

    def start(self) -> "LoadLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> int:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        return self.n


def summarise(path: Path) -> dict[str, Any]:
    """Statistics of a load log (for perf.json and the reports)."""
    rows = [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()] if Path(path).exists() else []
    out: dict[str, Any] = {"readings": len(rows), "errors": sum(len(r.get("errors", [])) for r in rows)}
    for k in ("cpu_pct", "ram_free_mb", "gpu_util_pct", "gpu_mem_used_mib", "python_procs", "node_procs"):
        v = sorted(r[k] for r in rows if r.get(k) is not None)
        if v:
            out[k] = {"median": v[len(v) // 2], "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "max": v[-1], "min": v[0]}
    return out
