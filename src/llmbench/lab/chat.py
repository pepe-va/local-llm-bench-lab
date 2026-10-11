"""The lab's chat: one llama-server on 127.0.0.1, and a streaming proxy that adds metrics.

The browser never talks to llama-server directly (it only listens locally); every answer
goes through `ChatServer.stream`, which measures time to first token on this machine and
ends the stream with llama-server's own timings plus memory and temperature.
"""

import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from llmbench.config.llama import LlamaConfig
from llmbench.lab.models import LocalModel
from llmbench.llamacpp.argv import server_argv
from llmbench.llamacpp.server import LlamaServer, ServerError
from llmbench.telemetry.sources import process_rss_mb, temperature_c

ChatState = Literal["loading", "ready", "error", "stopped"]
Reasoning = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class ChatSettings:
    ctx_size: int = 4096
    """llama.cpp would otherwise take the model's full context, whose KV cache may not fit."""
    threads: int | None = None
    reasoning: Reasoning = "auto"

    def config(self) -> LlamaConfig:
        return LlamaConfig.model_validate(
            {
                "load": {"ctx_size": self.ctx_size, "threads": self.threads},
                # auto is llama.cpp's own default: pass no flag at all
                "server": {"reasoning": None if self.reasoning == "auto" else self.reasoning},
            }
        )


class ChatServer:
    """llama-server for one model, loaded in the background (big models take a while)."""

    def __init__(
        self, model: LocalModel, binary: Path, settings: ChatSettings, log_path: Path
    ) -> None:
        self.model = model
        self.binary = binary
        self.settings = settings
        self.state: ChatState = "loading"
        self.error: str | None = None
        self.started = time.time()
        self._stopped = threading.Event()
        cfg = settings.config()
        self._server = LlamaServer(
            make_argv=lambda port: server_argv(binary, model.path, cfg, port=port),
            log_path=log_path,
            startup_timeout_s=900,
        )
        threading.Thread(target=self._load, name="llb-lab-chat", daemon=True).start()

    def _load(self) -> None:
        try:
            self._server.start()
        except (ServerError, httpx.HTTPError, OSError) as e:
            if not self._stopped.is_set():
                self.state, self.error = "error", str(e)
            return
        if self._stopped.is_set():  # stopped while loading
            self._server.stop()
            return
        self.state = "ready"

    def stop(self) -> None:
        self._stopped.set()
        self.state = "stopped"
        self._server.stop()

    def rss_mb(self) -> float | None:
        try:
            return process_rss_mb(self._server.pid) if self.state == "ready" else None
        except AssertionError:
            return None

    def to_json(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "error": self.error,
            "model": self.model.key,
            "name": self.model.name,
            "quant": self.model.quant,
            "settings": {
                "ctx_size": self.settings.ctx_size,
                "threads": self.settings.threads,
                "reasoning": self.settings.reasoning,
            },
            "load_s": round(self._server.load_s, 1) if self.state == "ready" else None,
            "rss_mb": self.rss_mb(),
            "argv": self._server.argv,
        }

    def stream(self, body: dict[str, Any]) -> Iterator[bytes]:
        """Server-sent events: `{content, reasoning}` deltas, then `{metrics}` or `{error}`."""
        if self.state != "ready":
            yield _sse({"error": f"the chat is {self.state}"})
            return
        t0 = time.perf_counter()
        first: float | None = None
        timings: dict[str, Any] = {}
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(900, connect=5)) as client,
                client.stream(
                    "POST", f"{self._server.url}/v1/chat/completions", json=body | {"stream": True}
                ) as r,
            ):
                if r.status_code != 200:  # noqa: PLR2004
                    r.read()
                    yield _sse({"error": f"llama-server: HTTP {r.status_code}: {r.text[:300]}"})
                    return
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event: dict[str, Any] = json.loads(line[6:])
                    timings = event.get("timings") or timings
                    choices: list[dict[str, Any]] = event.get("choices") or []
                    delta: dict[str, Any] = (choices[0].get("delta") or {}) if choices else {}
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    if text or reasoning:
                        first = first or time.perf_counter()
                        yield _sse({"content": text, "reasoning": reasoning})
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            yield _sse({"error": f"{type(e).__name__}: {e}"})
            return
        end = time.perf_counter()
        yield _sse(
            {
                "metrics": {
                    "ttft_ms": round((first - t0) * 1000) if first else None,
                    "total_s": round(end - t0, 2),
                    "prompt_n": timings.get("prompt_n"),
                    "prompt_tps": timings.get("prompt_per_second"),
                    "predicted_n": timings.get("predicted_n"),
                    "gen_tps": timings.get("predicted_per_second"),
                    "rss_mb": self.rss_mb(),
                    "temp_c": temperature_c(),
                }
            }
        )


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
