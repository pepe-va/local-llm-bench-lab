"""Pass/fail gates over configurations and selection of the winner."""

import operator
from collections.abc import Callable
from dataclasses import dataclass

from llmbench.analysis.view import ConfigRow
from llmbench.config.experiment import Gate, Select

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
}


@dataclass(frozen=True)
class GateCheck:
    gate: Gate
    value: float | None
    passed: bool

    @property
    def text(self) -> str:
        shown = "missing" if self.value is None else f"{self.value:.4g}"
        return f"{self.gate.metric} ({self.gate.agg}) {self.gate.op} {self.gate.value:g}: {shown}"


@dataclass(frozen=True)
class Verdict:
    row: ConfigRow
    checks: list[GateCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and self.row.failed_runs == 0

    @property
    def reasons(self) -> list[str]:
        out = [c.text for c in self.checks if not c.passed]
        if self.row.failed_runs:
            out.append(f"{self.row.failed_runs} failed run(s)")
        return out


def check(gates: list[Gate], rows: list[ConfigRow]) -> list[Verdict]:
    verdicts: list[Verdict] = []
    for row in rows:
        checks: list[GateCheck] = []
        for g in gates:
            value = row.metric(g.metric, g.agg)
            checks.append(GateCheck(g, value, value is not None and _OPS[g.op](value, g.value)))
        verdicts.append(Verdict(row, checks))
    return verdicts


def select(verdicts: list[Verdict], spec: Select | None) -> Verdict | None:
    """The best configuration by `spec` (among those that passed, by default)."""
    pool = [v for v in verdicts if v.passed] if spec is None or spec.among == "passed" else verdicts
    if spec is None:
        return pool[0] if len(pool) == 1 else None
    scored = [(v.row.metric(spec.metric, spec.agg), v) for v in pool]
    ranked = [(x, v) for x, v in scored if x is not None]
    if not ranked:
        return None
    best = max if spec.objective == "maximize" else min
    return best(ranked, key=lambda p: p[0])[1]
