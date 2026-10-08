"""Result domain objects. Independent of the backend that produced them."""

from dataclasses import dataclass, field
from typing import Any, Literal

type Phase = Literal["bench", "latency", "quality"]
type RunStatus = Literal["completed", "failed", "interrupted"]
type TestKind = Literal["pp", "tg", "pg"]


@dataclass(frozen=True)
class Measurement:
    """One llama-bench test (e.g. pp512 at depth 0) with its per-repetition samples."""

    test: TestKind
    n_prompt: int
    n_gen: int
    n_depth: int
    avg_ts: float
    stddev_ts: float
    avg_ns: int
    stddev_ns: int
    samples_ts: list[float]
    raw: dict[str, Any] = field(repr=False)
    median_ts: float = 0.0
    cv: float = 0.0
    unstable: bool = False

    @property
    def label(self) -> str:
        name = {"pp": f"pp{self.n_prompt}", "tg": f"tg{self.n_gen}"}.get(
            self.test, f"pp{self.n_prompt}+tg{self.n_gen}"
        )
        return f"{name}@d{self.n_depth}" if self.n_depth else name


@dataclass(frozen=True)
class RequestResult:
    """One timed llama-server request (client-side timing + server `timings`)."""

    id: str
    ok: bool
    error: str | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    prompt_n: int | None = None
    prompt_ms: float | None = None
    predicted_n: int | None = None
    predicted_ms: float | None = None
    prompt_tps: float | None = None
    gen_tps: float | None = None
    cache_n: int | None = None
    stop_type: str | None = None
    itl_ms: list[float] = field(default_factory=list[float], repr=False)
    """Gaps between consecutive streamed token events."""
    content: str = field(default="", repr=False)
    reasoning: str = field(default="", repr=False)
    """Thinking text, when the server separates it (`reasoning_format`)."""
    prompt_text: str | None = field(default=None, repr=False)
    category: str | None = None
    score: float | None = None
    passed: bool | None = None
    eval_detail: str | None = None
    judge_pending: bool = False
    label: float | None = None
    """Embeddings: the human similarity label of the pair."""
    similarity: float | None = None
    """Embeddings: cosine similarity of the pair."""


@dataclass(frozen=True)
class RunRecord:
    """Everything needed to reproduce and compare one run."""

    fingerprint: str
    experiment: str
    phase: str
    status: RunStatus
    build: dict[str, Any]
    model: dict[str, Any]
    variant: dict[str, Any]
    config: dict[str, Any]
    argv: list[str]
    ignored: list[str]
    env: dict[str, Any]
    started_at: str
    finished_at: str
    duration_s: float
    error: str | None = None
    telemetry: dict[str, Any] | None = None
    cooldown: dict[str, Any] | None = None
    measurements: list[Measurement] = field(default_factory=list[Measurement])
    suite: str | None = None
    suite_provenance: dict[str, Any] | None = None
    metrics: dict[str, float | None] = field(default_factory=dict[str, float | None])
    """Aggregates of a latency run (ttft_ms_p50, gen_tps_mean, ...)."""
    requests: list[RequestResult] = field(default_factory=list[RequestResult])
