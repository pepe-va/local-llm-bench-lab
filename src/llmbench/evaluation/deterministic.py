"""Objective evaluators: same output -> same score, no model involved."""

import json
import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Any

import jsonschema

from llmbench.config.evaluators import (
    All,
    AnyOf,
    Contains,
    EvaluatorSpec,
    Exact,
    JsonSchema,
    Length,
    LlmJudge,
    Normalization,
    NotContains,
    Regex,
    RegexExtract,
    Weighted,
)
from llmbench.data.requests import Request


@dataclass(frozen=True)
class Score:
    value: float
    """0..1."""
    passed: bool
    detail: str = ""
    pending_judge: bool = False
    """Contains an `llm_judge` part, scored later by the judge pass and stored apart."""
    has_objective: bool = True
    """False when the evaluator is only a judge: there is no objective score at all."""


def evaluate(spec: EvaluatorSpec, output: str, req: Request) -> Score:  # noqa: PLR0911, PLR0912 (dispatch table)
    match spec:
        case Exact():
            expected = spec.value if spec.value is not None else req.reference
            if expected is None:
                return Score(0, False, "exact: no value and no reference")
            ok = _norm(output, spec.normalize) == _norm(expected, spec.normalize)
            return _bool(ok, f"exact: expected {expected!r}")
        case Contains():
            hits = [v for v in spec.values if _has(output, v, spec.case_sensitive)]
            ok = len(hits) == len(spec.values) if spec.mode == "all" else bool(hits)
            missing = [v for v in spec.values if v not in hits]
            return Score(len(hits) / len(spec.values), ok, f"missing {missing}" if missing else "")
        case NotContains():
            found = [v for v in spec.values if _has(output, v, spec.case_sensitive)]
            return _bool(not found, f"found forbidden {found}" if found else "")
        case Regex():
            flags = re.IGNORECASE if spec.ignore_case else 0
            m = (re.fullmatch if spec.mode == "fullmatch" else re.search)(
                spec.pattern, output, flags
            )
            return _bool(m is not None, f"regex {spec.mode} {spec.pattern!r}")
        case RegexExtract():
            return _regex_extract(spec, output, req)
        case Length():
            return _length(spec, output, req)
        case JsonSchema():
            return _json_schema(spec, output)
        case LlmJudge():
            return Score(0, False, "", pending_judge=True, has_objective=False)
        case All():
            parts = [evaluate(p, output, req) for p in spec.of]
            obj = _objective(parts)
            if not obj:
                return _combine(parts, 0, False)
            return _combine(parts, min(p.value for p in obj), all(p.passed for p in obj))
        case AnyOf():
            parts = [evaluate(p, output, req) for p in spec.of]
            obj = _objective(parts)
            if not obj:
                return _combine(parts, 0, False)
            return _combine(parts, max(p.value for p in obj), any(p.passed for p in obj))
        case Weighted():
            scored = [(w.weight, evaluate(w.evaluator, output, req)) for w in spec.of]
            obj = [(w, s) for w, s in scored if not s.pending_judge]
            if not obj:
                return _combine([s for _, s in scored], 0, False)
            value = sum(w * s.value for w, s in obj) / sum(w for w, _ in obj)
            return _combine([s for _, s in scored], value, value >= spec.pass_threshold)


def _bool(ok: bool, detail: str) -> Score:
    return Score(1.0 if ok else 0.0, ok, "" if ok else detail)


def _objective(parts: list[Score]) -> list[Score]:
    """Judge parts are scored later and apart (PRD §18): they never move objective scores."""
    return [p for p in parts if not p.pending_judge]


def _combine(parts: list[Score], value: float, passed: bool) -> Score:
    detail = "; ".join(p.detail for p in parts if p.detail)
    return Score(
        value,
        passed,
        detail,
        pending_judge=any(p.pending_judge for p in parts),
        has_objective=any(p.has_objective for p in parts),
    )


def _norm(text: str, ops: list[Normalization]) -> str:
    for op in ops:
        if op == "strip":
            text = text.strip()
        elif op == "lower":
            text = text.lower()
        elif op == "collapse_ws":
            text = " ".join(text.split())
        elif op == "strip_punct":
            text = text.translate(str.maketrans("", "", string.punctuation + "¿¡«»"))
        else:  # strip_accents
            text = "".join(
                c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
            )
    return text


def _has(text: str, needle: str, case_sensitive: bool) -> bool:
    return needle in text if case_sensitive else needle.lower() in text.lower()


def _regex_extract(spec: RegexExtract, output: str, req: Request) -> Score:
    got = _extract(spec.pattern, output)
    expected = spec.value if spec.value is not None else req.reference
    if expected is not None and spec.reference_pattern is not None:
        expected = _extract(spec.reference_pattern, expected)
    if got is None:
        return Score(0, False, f"pattern {spec.pattern!r} not found in output")
    if expected is None:
        return Score(0, False, "no expected value")
    if spec.compare == "numeric":
        a, b = _number(got), _number(expected)
        ok = a is not None and b is not None and abs(a - b) <= spec.tolerance
    else:
        ok = got.strip() == expected.strip()
    return _bool(ok, f"got {got!r}, expected {expected!r}")


def _extract(pattern: str, text: str) -> str | None:
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    m = matches[-1]  # the final answer is usually the last match
    return m.group(1) if m.groups() else m.group(0)


def _number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _length(spec: Length, output: str, req: Request) -> Score:
    chars, words = len(output.strip()), len(output.split())
    problems: list[str] = []
    for value, lo, hi, unit in (
        (chars, spec.min_chars, spec.max_chars, "chars"),
        (words, spec.min_words, spec.max_words, "words"),
    ):
        if lo is not None and value < lo:
            problems.append(f"{value} {unit} < {lo}")
        if hi is not None and value > hi:
            problems.append(f"{value} {unit} > {hi}")
    if spec.ratio_to is not None:
        base_text = req.input if spec.ratio_to == "input" else req.reference
        base = len((base_text or "").split())
        if base == 0:
            problems.append(f"no {spec.ratio_to} text to compare with")
        else:
            ratio = words / base
            if spec.min_ratio is not None and ratio < spec.min_ratio:
                problems.append(f"ratio {ratio:.2f} < {spec.min_ratio}")
            if spec.max_ratio is not None and ratio > spec.max_ratio:
                problems.append(f"ratio {ratio:.2f} > {spec.max_ratio}")
    return _bool(not problems, "; ".join(problems))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _json_schema(spec: JsonSchema, output: str) -> Score:
    m = _FENCE.search(output)
    text = m.group(1) if m else output
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        return Score(0, False, f"not JSON: {e}")
    try:
        jsonschema.validate(data, spec.schema_)  # pyright: ignore[reportUnknownMemberType]
    except jsonschema.ValidationError as e:
        return Score(0, False, f"schema: {e.message}")
    return Score(1, True)
