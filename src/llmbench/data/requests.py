"""Normalized requests, independent of where they came from (synthetic, files, datasets)."""

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    id: str
    prompt: str | list[int]
    """Text, or a token array sent verbatim (synthetic data)."""
    n_predict: int

    @property
    def expected_prompt_tokens(self) -> int | None:
        return len(self.prompt) if isinstance(self.prompt, list) else None

    def digest(self) -> str:
        payload = json.dumps([self.prompt, self.n_predict], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def requests_digest(requests: list[Request]) -> str:
    """Identity of a request set (enters the run fingerprint)."""
    h = hashlib.sha256()
    for r in requests:
        h.update(r.digest().encode())
    return h.hexdigest()
