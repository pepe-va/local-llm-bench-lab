"""Wait until the machine is cool *and stable* before the next run."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from llmbench.config.experiment import Cooldown


@dataclass(frozen=True)
class CooldownResult:
    waited_s: float
    start_temp_c: float | None
    end_temp_c: float | None
    timed_out: bool = False
    skipped: str | None = None
    """Why no wait happened ("disabled", "no temperature sensor"), if so."""


def wait_for_cooldown(
    cfg: Cooldown,
    read_temp: Callable[[], float | None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    on_wait: Callable[[float, float], None] | None = None,
) -> CooldownResult:
    if not cfg.enabled:
        return CooldownResult(0, None, None, skipped="disabled")
    start_temp = read_temp()
    if start_temp is None:
        return CooldownResult(0, None, None, skipped="no temperature sensor")

    t0 = now()
    temp: float | None = start_temp
    cool_since: float | None = None
    while True:
        elapsed = now() - t0
        if temp is not None and temp <= cfg.max_temp_c:
            cool_since = cool_since if cool_since is not None else now()
            if now() - cool_since >= cfg.stable_for_s:
                return CooldownResult(elapsed, start_temp, temp)
        else:
            cool_since = None
        if elapsed >= cfg.timeout_s:
            return CooldownResult(elapsed, start_temp, temp, timed_out=True)
        if on_wait is not None and temp is not None:
            on_wait(temp, elapsed)
        sleep(cfg.poll_s)
        temp = read_temp()
