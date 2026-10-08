"""Decoding of Raspberry Pi `vcgencmd get_throttled` bits."""

from dataclasses import dataclass

_NOW = {
    0: "under_voltage",
    1: "arm_freq_capped",
    2: "throttled",
    3: "soft_temp_limit",
}
_OCCURRED_SHIFT = 16


@dataclass(frozen=True)
class ThrottleState:
    raw: int
    now: tuple[str, ...]
    occurred: tuple[str, ...]

    @property
    def is_throttled_now(self) -> bool:
        return bool(self.now)

    @property
    def has_occurred(self) -> bool:
        return bool(self.occurred)


def decode(raw: int) -> ThrottleState:
    now = tuple(name for bit, name in _NOW.items() if raw & (1 << bit))
    occurred = tuple(name for bit, name in _NOW.items() if raw & (1 << (bit + _OCCURRED_SHIFT)))
    return ThrottleState(raw, now, occurred)


def parse(text: str) -> int:
    """`throttled=0x50005` -> 0x50005."""
    return int(text.strip().split("=", 1)[-1], 16)
