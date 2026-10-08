"""Normalized requests, independent of where they came from (synthetic, items, files, datasets)."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from llmbench.config.evaluators import EvaluatorSpec

_EVAL = TypeAdapter[EvaluatorSpec](EvaluatorSpec)


@dataclass(frozen=True)
class Request:
    id: str
    prompt: str | list[int]
    """Rendered user text, or a token array sent verbatim (synthetic data)."""
    n_predict: int
    system: str | None = None
    """System prompt; only used by the chat endpoint."""
    input: str | None = None
    reference: str | None = None
    category: str | None = None
    evaluator: EvaluatorSpec | None = None

    @property
    def expected_prompt_tokens(self) -> int | None:
        return len(self.prompt) if isinstance(self.prompt, list) else None

    def messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        out.append({"role": "user", "content": self.prompt})
        return out

    def digest(self) -> str:
        payload = [
            self.prompt,
            self.n_predict,
            self.system,
            self.input,
            self.reference,
            _EVAL.dump_python(self.evaluator, mode="json") if self.evaluator else None,
        ]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def requests_digest(requests: list[Request]) -> str:
    """Identity of a request set (enters the run fingerprint)."""
    h = hashlib.sha256()
    for r in requests:
        h.update(r.digest().encode())
    return h.hexdigest()
