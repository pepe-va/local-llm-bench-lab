"""Background sampling of hardware state while a benchmark process runs."""

import threading
import time
from dataclasses import dataclass, field

from llmbench.config.experiment import Telemetry
from llmbench.telemetry import sources


@dataclass(frozen=True)
class Sample:
    t: float
    temp_c: float | None
    cpu_freq_mhz: float | None
    ram_used_mb: float
    proc_rss_mb: float | None


@dataclass(frozen=True)
class TelemetrySummary:
    temp_before_c: float | None
    temp_peak_c: float | None
    temp_after_c: float | None
    ram_before_mb: float
    ram_peak_mb: float
    rss_peak_mb: float | None
    cpu_freq_min_mhz: float | None
    cpu_freq_avg_mhz: float | None
    throttled_before: int | None
    throttled_after: int | None
    samples: list[Sample] = field(default_factory=list[Sample])

    def to_dict(self) -> dict[str, object]:
        d = self.__dict__.copy()
        d.pop("samples")
        return d


class Sampler:
    def __init__(self, cfg: Telemetry) -> None:
        self._period = 1 / cfg.hz
        self._rpi = "rpi" in cfg.sources
        self._pid: int | None = None
        self._samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._before: tuple[float | None, float, int | None] = (None, 0.0, None)

    def attach(self, pid: int) -> None:
        self._pid = pid

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._before = (
            sources.temperature_c(use_rpi=self._rpi),
            sources.ram_used_mb(),
            sources.throttled_raw() if self._rpi else None,
        )
        self._thread = threading.Thread(target=self._loop, name="llb-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> TelemetrySummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        temp_after = sources.temperature_c(use_rpi=self._rpi)
        throttled_after = sources.throttled_raw() if self._rpi else None
        return summarize(self._samples, self._before, temp_after, throttled_after)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._read())
            self._stop.wait(self._period)

    def _read(self) -> Sample:
        return Sample(
            t=time.monotonic() - self._t0,
            temp_c=sources.temperature_c(use_rpi=self._rpi),
            cpu_freq_mhz=sources.cpu_freq_mhz(),
            ram_used_mb=sources.ram_used_mb(),
            proc_rss_mb=sources.process_rss_mb(self._pid) if self._pid else None,
        )


def summarize(
    samples: list[Sample],
    before: tuple[float | None, float, int | None],
    temp_after: float | None,
    throttled_after: int | None,
) -> TelemetrySummary:
    temp_before, ram_before, throttled_before = before
    temps = [x for x in (temp_before, *(s.temp_c for s in samples), temp_after) if x is not None]
    rss = [s.proc_rss_mb for s in samples if s.proc_rss_mb is not None]
    freqs = [s.cpu_freq_mhz for s in samples if s.cpu_freq_mhz is not None]
    return TelemetrySummary(
        temp_before_c=temp_before,
        temp_peak_c=max(temps) if temps else None,
        temp_after_c=temp_after,
        ram_before_mb=ram_before,
        ram_peak_mb=max([ram_before, *(s.ram_used_mb for s in samples)]),
        rss_peak_mb=max(rss) if rss else None,
        cpu_freq_min_mhz=min(freqs) if freqs else None,
        cpu_freq_avg_mhz=sum(freqs) / len(freqs) if freqs else None,
        throttled_before=throttled_before,
        throttled_after=throttled_after,
        samples=samples,
    )
