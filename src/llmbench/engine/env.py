"""Machine snapshot stored with every run, plus pre-flight checks."""

import os
import platform
import socket
from pathlib import Path
from typing import Any

from llmbench import __version__
from llmbench.config.experiment import Preflight
from llmbench.telemetry import sources, throttle


class PreflightError(Exception):
    pass


def snapshot() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "system": platform.system(),
        "kernel": platform.release(),
        "os": _os_release(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_total_mb": round(sources.ram_total_mb()),
        "governor": sources.cpu_governor(),
        "python": platform.python_version(),
        "llb_version": __version__,
    }


def preflight(cfg: Preflight, *, use_rpi: bool) -> list[str]:
    """Raise on hard failures; return warnings worth showing."""
    warnings: list[str] = []
    gov = sources.cpu_governor()
    if cfg.require_governor is not None:
        if gov is None:
            warnings.append("CPU governor not readable; `require_governor` not enforced")
        elif gov != cfg.require_governor:
            raise PreflightError(
                f"CPU governor is {gov!r}, experiment requires {cfg.require_governor!r} "
                f"(e.g. `sudo cpupower frequency-set -g {cfg.require_governor}`)"
            )
    if use_rpi and (raw := sources.throttled_raw()) is not None:
        state = throttle.decode(raw)
        if state.is_throttled_now and cfg.fail_on_throttled:
            raise PreflightError(f"machine is throttled right now: {', '.join(state.now)}")
        if state.has_occurred:
            warnings.append(f"throttling occurred since boot: {', '.join(state.occurred)}")
    return warnings


def _os_release() -> str | None:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return None


def _cpu_model() -> str | None:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor() or None
    model = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "Model":  # Raspberry Pi board name
            return value.strip()
        if key == "model name" and model is None:
            model = value.strip()
    return model
