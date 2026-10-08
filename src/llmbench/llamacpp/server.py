"""llama-server process lifecycle: start on a free local port, wait until healthy, stop.

The process is reaped with `wait4` for its CPU time; its peak RSS is read from `VmHWM`
(ADR-0012).
"""

import os
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import httpx

from llmbench.llamacpp.argv import Argv
from llmbench.telemetry.peak import read_vmhwm_mb

_POLL_S = 0.2
_STOP_GRACE_S = 15.0


class ServerError(Exception):
    pass


@dataclass(frozen=True)
class ServerStats:
    exit_code: int | None
    max_rss_mb: float | None
    cpu_user_s: float | None
    cpu_sys_s: float | None


@dataclass
class LlamaServer:
    """Context manager around one llama-server process."""

    make_argv: Callable[[int], Argv]
    log_path: Path
    env: dict[str, str] = field(default_factory=dict[str, str])
    startup_timeout_s: float = 600
    on_start: Callable[[int], None] | None = None

    argv: list[str] = field(default_factory=list[str], init=False)
    url: str = field(default="", init=False)
    load_s: float = field(default=0.0, init=False)
    props: dict[str, Any] = field(default_factory=dict[str, Any], init=False)
    stats: ServerStats | None = field(default=None, init=False)
    _proc: subprocess.Popen[bytes] | None = field(default=None, init=False)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def pid(self) -> int:
        assert self._proc is not None
        return self._proc.pid

    def start(self) -> None:
        port = free_port()
        self.argv = self.make_argv(port).args
        self.url = f"http://127.0.0.1:{port}"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("wb")
        t0 = time.monotonic()
        try:
            self._proc = subprocess.Popen(
                self.argv, stdout=log, stderr=subprocess.STDOUT, env=os.environ | self.env
            )
        except OSError as e:
            raise ServerError(f"cannot start {self.argv[0]}: {e}") from None
        finally:
            log.close()
        if self.on_start is not None:
            self.on_start(self._proc.pid)
        self._wait_healthy(t0)
        self.load_s = time.monotonic() - t0
        self.props = httpx.get(f"{self.url}/props", timeout=10).json()

    def _wait_healthy(self, t0: float) -> None:
        assert self._proc is not None
        with httpx.Client(timeout=2) as client:
            while True:
                code = self._proc.poll()
                if code is not None:
                    raise ServerError(f"llama-server exited with code {code}\n{self.log_tail()}")
                try:
                    if client.get(f"{self.url}/health").status_code == 200:  # noqa: PLR2004
                        return
                except httpx.TransportError:
                    pass  # not listening yet
                if time.monotonic() - t0 > self.startup_timeout_s:
                    self.stop()
                    raise ServerError(
                        f"llama-server not healthy after {self.startup_timeout_s:.0f}s\n"
                        + self.log_tail()
                    )
                time.sleep(_POLL_S)

    def metrics(self) -> dict[str, float]:
        """Prometheus `/metrics` as a flat dict (`llamacpp:` prefix removed)."""
        text = httpx.get(f"{self.url}/metrics", timeout=10).text
        return parse_prometheus(text)

    def peak_rss_mb(self) -> float | None:
        """Peak resident memory since start (`VmHWM`; exact, see ADR-0012)."""
        return read_vmhwm_mb(self._proc.pid) if self._proc is not None else None

    def stop(self) -> ServerStats:
        """Idempotent: the first call reaps the process, later calls return the same stats."""
        if self.stats is not None:
            return self.stats
        proc = self._proc
        if proc is None:
            return ServerStats(None, None, None, None)
        self._proc = None
        if proc.returncode is not None:
            self.stats = ServerStats(proc.returncode, None, None, None)
            return self.stats
        peak = read_vmhwm_mb(proc.pid)
        proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + _STOP_GRACE_S
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid != 0:
                break
            if time.monotonic() > deadline:
                proc.kill()
                _, status, usage = os.wait4(proc.pid, 0)
                break
            time.sleep(_POLL_S)
        proc.returncode = os.waitstatus_to_exitcode(status)
        self.stats = ServerStats(
            proc.returncode,
            peak,
            usage.ru_utime,
            usage.ru_stime,
        )
        return self.stats

    def log_tail(self, n: int = 20) -> str:
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            out[name.removeprefix("llamacpp:")] = float(value)
        except ValueError:
            continue
    return out
