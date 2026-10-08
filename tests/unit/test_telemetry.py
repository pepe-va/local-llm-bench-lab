from llmbench.config.experiment import Cooldown
from llmbench.telemetry import throttle
from llmbench.telemetry.cooldown import wait_for_cooldown
from llmbench.telemetry.sampler import Sample, summarize


class Clock:
    def __init__(self, temps: list[float | None]) -> None:
        self.t = 0.0
        self.temps = temps
        self.reads = 0

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s

    def read(self) -> float | None:
        value = self.temps[min(self.reads, len(self.temps) - 1)]
        self.reads += 1
        return value


def run(cfg: Cooldown, temps: list[float | None]):
    c = Clock(temps)
    return wait_for_cooldown(cfg, c.read, sleep=c.sleep, now=c.now)


def test_cool_machine_waits_only_for_stability() -> None:
    r = run(Cooldown(max_temp_c=60, stable_for_s=10, poll_s=5), [50.0])
    assert r.waited_s == 10
    assert not r.timed_out


def test_waits_until_below_threshold_and_stable() -> None:
    r = run(Cooldown(max_temp_c=60, stable_for_s=5, poll_s=5), [70, 65, 59, 61, 58, 57])
    # 70 -> 65 -> 59 (cool since t=10) -> 61 (reset) -> 58 (t=20) -> 57 (t=25, stable 5s)
    assert r.waited_s == 25
    assert r.end_temp_c == 57


def test_timeout() -> None:
    r = run(Cooldown(max_temp_c=60, poll_s=5, timeout_s=20), [80.0])
    assert r.timed_out
    assert r.waited_s == 20


def test_no_sensor_or_disabled_is_skipped() -> None:
    assert run(Cooldown(), [None]).skipped == "no temperature sensor"
    assert run(Cooldown(enabled=False), [90.0]).skipped == "disabled"


def test_throttle_decoding() -> None:
    s = throttle.decode(throttle.parse("throttled=0x50005"))
    assert s.now == ("under_voltage", "throttled")
    assert s.occurred == ("under_voltage", "throttled")
    assert not throttle.decode(0).has_occurred


def test_summary_peaks() -> None:
    samples = [
        Sample(0.0, 50.0, 2400, 1000, 300),
        Sample(0.5, 62.5, 1800, 1500, 900),
        Sample(1.0, None, None, 1200, None),
    ]
    s = summarize(samples, (48.0, 900.0, None), 55.0, None)
    assert s.temp_peak_c == 62.5
    assert s.ram_peak_mb == 1500
    assert s.rss_peak_mb == 900
    assert s.cpu_freq_min_mhz == 1800
    assert "samples" not in s.to_dict()
