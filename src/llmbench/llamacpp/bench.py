"""Run `llama-bench -o jsonl` and parse its output."""

import json
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any

from llmbench.results.models import Measurement, TestKind


class BenchError(Exception):
    pass


@dataclass(frozen=True)
class BenchOutput:
    measurements: list[Measurement]
    stderr: str
    duration_s: float
    max_rss_mb: float | None = None
    """Exact peak resident memory of llama-bench, from the kernel (`ru_maxrss`)."""
    cpu_user_s: float | None = None
    cpu_sys_s: float | None = None


def parse_jsonl(text: str) -> list[Measurement]:
    out: list[Measurement] = []
    for n, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            row: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as e:
            raise BenchError(f"invalid JSONL at line {n}: {e}") from None
        out.append(_measurement(row))
    return out


def _measurement(row: dict[str, Any]) -> Measurement:
    n_prompt, n_gen = int(row["n_prompt"]), int(row["n_gen"])
    kind: TestKind = "pg" if n_prompt and n_gen else ("pp" if n_prompt else "tg")
    return Measurement(
        test=kind,
        n_prompt=n_prompt,
        n_gen=n_gen,
        n_depth=int(row.get("n_depth", 0)),
        avg_ts=float(row["avg_ts"]),
        stddev_ts=float(row["stddev_ts"]),
        avg_ns=int(row["avg_ns"]),
        stddev_ns=int(row["stddev_ns"]),
        samples_ts=[float(x) for x in row.get("samples_ts", [])],
        raw=row,
    )


def run_bench(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    timeout_s: float,
    on_start: Callable[[int], None] | None = None,
) -> BenchOutput:
    """Run llama-bench and reap it with `wait4` to get its exact peak RSS and CPU time."""
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ | (env or {}),
        )
    except OSError as e:
        raise BenchError(f"cannot start {argv[0]}: {e}") from None
    if on_start is not None:
        on_start(proc.pid)

    out: list[str] = []
    err: list[str] = []
    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, out), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err), daemon=True),
    ]
    for r in readers:
        r.start()
    timer = threading.Timer(timeout_s, proc.kill)
    timer.start()
    try:
        status, usage = _wait4(proc)
    finally:
        timer.cancel()
    for r in readers:
        r.join()
    duration = time.monotonic() - t0
    stdout, stderr = "".join(out), "".join(err)

    code = os.waitstatus_to_exitcode(status)
    if code == -signal.SIGKILL and duration >= timeout_s:
        raise BenchError(f"timed out after {timeout_s:.0f}s")
    if code != 0:
        raise BenchError(_describe_failure(code, stderr))
    measurements = parse_jsonl(stdout)
    if not measurements:
        raise BenchError("llama-bench produced no results\n" + _tail(stderr))
    return BenchOutput(
        measurements,
        stderr,
        duration,
        max_rss_mb=_maxrss_mb(usage.ru_maxrss),
        cpu_user_s=usage.ru_utime,
        cpu_sys_s=usage.ru_stime,
    )


def _wait4(proc: subprocess.Popen[str]) -> tuple[int, resource.struct_rusage]:
    try:
        _, status, usage = os.wait4(proc.pid, 0)
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        os.wait4(proc.pid, 0)
        proc.returncode = -signal.SIGINT
        raise
    proc.returncode = os.waitstatus_to_exitcode(status)  # tell Popen it has been reaped
    return status, usage


def _drain(stream: IO[str] | None, sink: list[str]) -> None:
    if stream is not None:
        sink.append(stream.read())
        stream.close()


def _maxrss_mb(maxrss: int) -> float:
    """ru_maxrss is KiB on Linux, bytes on macOS."""
    return maxrss / (1024 * 1024) if sys.platform == "darwin" else maxrss / 1024


def _describe_failure(code: int, stderr: str) -> str:
    if code == -signal.SIGKILL:
        reason = "killed (SIGKILL; likely out of memory)"
    elif code < 0:
        reason = f"killed by signal {-code}"
    else:
        reason = f"exit code {code}"
    return f"llama-bench failed: {reason}\n{_tail(stderr)}"


def _tail(text: str, n: int = 15) -> str:
    return "\n".join(text.strip().splitlines()[-n:])
