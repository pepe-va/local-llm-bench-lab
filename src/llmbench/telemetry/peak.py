"""Exact peak resident memory of a child process, from `/proc/<pid>/status` (`VmHWM`).

Why not `wait4`'s `ru_maxrss` (ADR-0012): on Linux it keeps the high-water mark of the
address space the child had *before* `exec` — a copy of the Python parent — so a 13 MB
llama-bench launched from a 300 MB Python process reports 313 MB. `VmHWM` belongs to the
post-exec address space and only grows, so polling it is exact up to the last read.
"""

import threading
from pathlib import Path

_KIB_PER_MIB = 1024


def read_vmhwm_mb(pid: int) -> float | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / _KIB_PER_MIB
    return None


class PeakMemory:
    """Polls `VmHWM` of a pid until stopped (or the process exits)."""

    def __init__(self, pid: int, interval_s: float = 0.02) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="llb-peak-mem", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = read_vmhwm_mb(self.pid)
            if value is None:  # process gone (or not Linux)
                return
            self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)
            self._stop.wait(self.interval_s)

    def stop(self) -> float | None:
        self._stop.set()
        self._thread.join()
        return self.peak_mb
