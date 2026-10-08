"""HTTP client for llama-server: tokenize, and streamed completions with client-side timing.

TTFT and inter-token latency are measured on the client (what a user would experience);
prompt/generation throughput come from the server's own `timings` in the final event.
"""

import json
import time
from typing import Any

import httpx

from llmbench.data.requests import Request
from llmbench.results.models import RequestResult


class LlamaClient:
    def __init__(self, base_url: str, *, timeout_s: float = 600) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout_s, connect=10))

    def close(self) -> None:
        self._http.close()

    def tokenize(self, text: str, *, add_special: bool = False) -> list[int]:
        r = self._http.post("/tokenize", json={"content": text, "add_special": add_special})
        r.raise_for_status()
        return list(r.json()["tokens"])

    def bos(self) -> list[int]:
        """Tokens the model adds in front of a prompt (BOS), as the server would."""
        return self.tokenize("", add_special=True)

    def complete(self, req: Request, sampling: dict[str, Any]) -> RequestResult:
        body = {
            **sampling,
            "prompt": req.prompt,
            "n_predict": req.n_predict,
            "stream": True,
            "cache_prompt": False,  # every request processes its full prompt
        }
        t0 = time.perf_counter()
        first: float | None = None
        last: float | None = None
        itl: list[float] = []
        parts: list[str] = []
        final: dict[str, Any] = {}
        try:
            with self._http.stream("POST", "/completion", json=body) as r:
                if r.status_code != 200:  # noqa: PLR2004
                    r.read()
                    error = f"HTTP {r.status_code}: {r.text[:300]}"
                    return RequestResult(req.id, ok=False, error=error)
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event: dict[str, Any] = json.loads(line[6:])
                    now = time.perf_counter()
                    if event.get("content") or event.get("tokens"):
                        if first is None:
                            first = now
                        elif last is not None:
                            itl.append((now - last) * 1000)
                        last = now
                        parts.append(event.get("content", ""))
                    if event.get("stop"):
                        final = event
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return RequestResult(req.id, ok=False, error=f"{type(e).__name__}: {e}")
        end = time.perf_counter()

        t: dict[str, Any] = final.get("timings", {})
        return RequestResult(
            id=req.id,
            ok=bool(final),
            error=None if final else "stream ended without a final event",
            ttft_ms=(first - t0) * 1000 if first is not None else None,
            e2e_ms=(end - t0) * 1000,
            prompt_n=t.get("prompt_n"),
            prompt_ms=t.get("prompt_ms"),
            predicted_n=t.get("predicted_n"),
            predicted_ms=t.get("predicted_ms"),
            prompt_tps=t.get("prompt_per_second"),
            gen_tps=t.get("predicted_per_second"),
            cache_n=t.get("cache_n"),
            stop_type=final.get("stop_type"),
            itl_ms=itl,
            content="".join(parts),
        )
