"""Point-in-time hardware readings. Each reader returns None when unavailable."""

import shutil
import subprocess
from pathlib import Path

import psutil

from llmbench.telemetry import throttle

_THERMAL = Path("/sys/class/thermal")
_PREFERRED_ZONES = ("cpu", "soc", "x86_pkg", "k10temp", "coretemp")
_MB = 1024 * 1024


def temperature_c(*, use_rpi: bool = True) -> float | None:
    if use_rpi and (t := _vcgencmd_temp()) is not None:
        return t
    return _sysfs_temp()


def _sysfs_temp() -> float | None:
    preferred: list[float] = []
    others: list[float] = []
    for zone in sorted(_THERMAL.glob("thermal_zone*")):
        try:
            value = int((zone / "temp").read_text()) / 1000
            kind = (zone / "type").read_text().strip().lower()
        except (OSError, ValueError):
            continue
        (preferred if any(p in kind for p in _PREFERRED_ZONES) else others).append(value)
    readings = preferred or others
    if readings:
        return max(readings)
    return _psutil_temp()


def _psutil_temp() -> float | None:
    sensors = getattr(psutil, "sensors_temperatures", None)
    if sensors is None:
        return None
    try:
        groups = sensors()
    except OSError:
        return None
    values = [t.current for name, ts in groups.items() for t in ts if _is_cpu(name)]
    return max(values) if values else None


def _is_cpu(name: str) -> bool:
    return any(p in name.lower() for p in _PREFERRED_ZONES)


def cpu_freq_mhz() -> float | None:
    try:
        f = psutil.cpu_freq()
    except (OSError, NotImplementedError):
        return None
    return float(f.current) if f else None


def ram_used_mb() -> float:
    return psutil.virtual_memory().used / _MB


def ram_total_mb() -> float:
    return psutil.virtual_memory().total / _MB


def process_rss_mb(pid: int) -> float | None:
    try:
        return psutil.Process(pid).memory_info().rss / _MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def cpu_governor() -> str | None:
    try:
        return Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError:
        return None


def throttled_raw() -> int | None:
    out = _vcgencmd("get_throttled")
    return throttle.parse(out) if out else None


def _vcgencmd_temp() -> float | None:
    out = _vcgencmd("measure_temp")  # temp=48.3'C
    if not out:
        return None
    try:
        return float(out.split("=", 1)[1].split("'", 1)[0])
    except (IndexError, ValueError):
        return None


def _vcgencmd(*args: str) -> str | None:
    exe = shutil.which("vcgencmd")
    if exe is None:
        return None
    try:
        p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None
