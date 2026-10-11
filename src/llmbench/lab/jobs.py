"""Long `llb` commands started from the lab (`run`, `llama build`), one at a time.

Each runs as a child process in its own session, so stopping the lab never interrupts a
measurement. A small state file lets a restarted lab find a job that is still running.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

JobKind = Literal["run", "judge", "report", "build", "other"]


class LabBusyError(Exception):
    """Something else is using the machine (HTTP 409)."""


@dataclass
class Job:
    kind: JobKind
    title: str
    argv: list[str]
    log: Path
    pid: int
    started_at: str
    experiment: str | None = None
    returncode: int | None = None
    _proc: subprocess.Popen[bytes] | None = field(default=None, repr=False)

    @property
    def running(self) -> bool:
        if self._proc is not None:
            self.returncode = self._proc.poll()
            return self.returncode is None
        return _alive(self.pid)  # adopted from an earlier lab: no exit code to read

    def cancel(self) -> None:
        """Ctrl-C for the whole job: `llb run` saves the current run as interrupted."""
        if self.running:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.pid, signal.SIGINT)

    def tail(self, n: int = 60) -> str:
        try:
            data = self.log.read_bytes()[-20_000:]
        except OSError:
            return ""
        lines = data.decode(errors="replace").replace("\r", "\n").splitlines()
        return "\n".join(line for line in lines[-n:])

    def to_json(self) -> dict[str, Any]:
        running = self.running
        return {
            "kind": self.kind,
            "title": self.title,
            "started_at": self.started_at,
            "experiment": self.experiment,
            "running": running,
            "returncode": self.returncode,
            "state": "running" if running else _ended(self.returncode),
            "log": str(self.log),
            "tail": self.tail(),
        }


def _ended(code: int | None) -> str:
    if code is None:
        return "finished"
    return {0: "ok", 130: "interrupted"}.get(code, "failed")


def _alive(pid: int) -> bool:
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return b"llmbench" in cmdline


class Jobs:
    def __init__(self, home: Path) -> None:
        self.home = home
        self.current: Job | None = self._adopt()

    @property
    def _state_path(self) -> Path:
        return self.home / "job.json"

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.running

    def start(
        self,
        kind: JobKind,
        title: str,
        args: list[str],
        *,
        cwd: Path,
        experiment: str | None = None,
    ) -> Job:
        """Run `llb <args>` in the background, with its output in a log file."""
        if self.busy:
            assert self.current is not None
            raise LabBusyError(f"{self.current.title} is still running")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        log = self.home / "logs" / f"{stamp}-{kind}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, "-m", "llmbench", *args]
        with log.open("wb") as out:
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=os.environ | {"COLUMNS": "120", "NO_COLOR": "1"},
            )
        job = Job(
            kind, title, argv, log, proc.pid,
            datetime.now(UTC).isoformat(timespec="seconds"), experiment, None, proc,
        )  # fmt: skip
        self.current = job
        state = {
            "kind": kind, "title": title, "argv": argv, "log": str(log), "pid": job.pid,
            "started_at": job.started_at, "experiment": experiment,
        }  # fmt: skip
        self._state_path.write_text(json.dumps(state))
        return job

    def _adopt(self) -> Job | None:
        try:
            data: dict[str, Any] = json.loads(self._state_path.read_text())
            job = Job(
                data["kind"], data["title"], data["argv"], Path(data["log"]), int(data["pid"]),
                data["started_at"], data.get("experiment"),
            )  # fmt: skip
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return job if job.running else None
