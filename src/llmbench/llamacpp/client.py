"""HTTP client for llama-server: tokenize, and streamed completions with client-side timing.

TTFT and inter-token latency are measured on the client (what a user would experience);
prompt/generation throughput come from the server's own `timings` in the final event.
"""

import json
import time
from typing import Any

import httpx

from llmbench.data.requests import Request
from llmbench.evaluation.retrieval import cosine, mrr, ndcg_at_k, recall_at_k
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

    def complete(
        self, req: Request, sampling: dict[str, Any], *, endpoint: str = "completion"
    ) -> RequestResult:
        """Stream one request; `endpoint` is `completion` (raw) or `chat` (chat template)."""
        if endpoint == "chat":
            path = "/v1/chat/completions"
            body: dict[str, Any] = {
                **sampling,
                "messages": req.messages(),
                "max_tokens": req.n_predict,
                "stream_options": {"include_usage": True},
            }
        else:
            path = "/completion"
            body = {**sampling, "prompt": req.prompt, "n_predict": req.n_predict}
        body |= {"stream": True, "cache_prompt": False}  # full prompt processing every time
        return self._stream(req, path, body, chat=endpoint == "chat")

    def embed(self, req: Request) -> RequestResult:
        """Embed the pair (`a`, `b`), or the prompt, and report the cosine similarity."""
        texts = [req.a, req.b] if req.a is not None and req.b is not None else [str(req.prompt)]
        t0 = time.perf_counter()
        try:
            r = self._http.post("/v1/embeddings", json={"input": texts})
            r.raise_for_status()
            data: dict[str, Any] = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return RequestResult(req.id, ok=False, error=f"{type(e).__name__}: {e}")
        e2e = (time.perf_counter() - t0) * 1000
        vectors: list[list[float]] = [d["embedding"] for d in data["data"]]
        prompt_n = int(data.get("usage", {}).get("prompt_tokens", 0)) or None
        sim = cosine(vectors[0], vectors[1]) if len(vectors) == 2 else None  # noqa: PLR2004
        return RequestResult(
            req.id,
            ok=True,
            e2e_ms=e2e,
            prompt_n=prompt_n,
            prompt_tps=prompt_n / (e2e / 1000) if prompt_n and e2e else None,
            category=req.category,
            label=req.label,
            similarity=sim,
        )

    def rerank(self, req: Request, *, k: int) -> RequestResult:
        """Rank `documents` for `query`; score = nDCG@k against the `relevant` indices."""
        body = {"query": req.query, "documents": list(req.documents), "top_n": len(req.documents)}
        t0 = time.perf_counter()
        try:
            r = self._http.post("/v1/rerank", json=body)
            r.raise_for_status()
            data: dict[str, Any] = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return RequestResult(req.id, ok=False, error=f"{type(e).__name__}: {e}")
        e2e = (time.perf_counter() - t0) * 1000
        ranking = _ranked(data["results"])
        relevant = set(req.relevant)
        prompt_n = int(data.get("usage", {}).get("prompt_tokens", 0)) or None
        detail = {
            "ranking": ranking,
            "mrr": mrr(ranking, relevant),
            f"recall@{k}": recall_at_k(ranking, relevant, k),
        }
        return RequestResult(
            req.id,
            ok=True,
            e2e_ms=e2e,
            prompt_n=prompt_n,
            prompt_tps=prompt_n / (e2e / 1000) if prompt_n and e2e else None,
            category=req.category,
            score=ndcg_at_k(ranking, relevant, k),
            passed=bool(ranking) and ranking[0] in relevant,
            eval_detail=json.dumps(detail),
        )

    def _stream(
        self, req: Request, path: str, body: dict[str, Any], *, chat: bool
    ) -> RequestResult:
        t0 = time.perf_counter()
        first: float | None = None
        last: float | None = None
        itl: list[float] = []
        parts: list[str] = []
        thinking: list[str] = []
        timings: dict[str, Any] = {}
        stop_type: str | None = None
        done = False
        try:
            with self._http.stream("POST", path, json=body) as r:
                if r.status_code != 200:  # noqa: PLR2004
                    r.read()
                    error = f"HTTP {r.status_code}: {r.text[:300]}"
                    return RequestResult(req.id, ok=False, error=error)
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        done = True
                        break
                    event: dict[str, Any] = json.loads(line[6:])
                    now = time.perf_counter()
                    text, reasoning, finish = _delta(event, chat=chat)
                    if text or reasoning or (not chat and event.get("tokens")):
                        if first is None:
                            first = now
                        elif last is not None:
                            itl.append((now - last) * 1000)
                        last = now
                        parts.append(text)
                        thinking.append(reasoning)
                    if "timings" in event:
                        timings = event["timings"]
                    if finish is not None:
                        stop_type = finish
                    if not chat and event.get("stop"):
                        done = True
                        break
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            return RequestResult(req.id, ok=False, error=f"{type(e).__name__}: {e}")
        end = time.perf_counter()
        ok = done or bool(timings)
        return RequestResult(
            id=req.id,
            ok=ok,
            error=None if ok else "stream ended without a final event",
            ttft_ms=(first - t0) * 1000 if first is not None else None,
            e2e_ms=(end - t0) * 1000,
            prompt_n=timings.get("prompt_n"),
            prompt_ms=timings.get("prompt_ms"),
            predicted_n=timings.get("predicted_n"),
            predicted_ms=timings.get("predicted_ms"),
            prompt_tps=timings.get("prompt_per_second"),
            gen_tps=timings.get("predicted_per_second"),
            cache_n=timings.get("cache_n"),
            stop_type=stop_type,
            itl_ms=itl,
            content="".join(parts),
            reasoning="".join(thinking),
            prompt_text=req.prompt if isinstance(req.prompt, str) else None,
            category=req.category,
        )


def _ranked(results: list[dict[str, Any]]) -> list[int]:
    ordered = sorted(results, key=lambda r: float(r["relevance_score"]), reverse=True)
    return [int(r["index"]) for r in ordered]


def _delta(event: dict[str, Any], *, chat: bool) -> tuple[str, str, str | None]:
    """(content, reasoning, finish/stop type) of one SSE event."""
    if not chat:
        stop = event.get("stop_type") if event.get("stop") else None
        return event.get("content") or "", "", stop
    choices: list[dict[str, Any]] = event.get("choices") or []
    if not choices:
        return "", "", None
    delta: dict[str, Any] = choices[0].get("delta") or {}
    return (
        delta.get("content") or "",
        delta.get("reasoning_content") or "",
        choices[0].get("finish_reason"),
    )
