"""Normalized requests, independent of where they came from (synthetic, items, files, datasets)."""

import base64
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    image: str | None = None
    """Absolute path or http(s)/data URL (vision)."""
    a: str | None = None
    b: str | None = None
    label: float | None = None
    query: str | None = None
    documents: tuple[str, ...] = ()
    relevant: tuple[int, ...] = ()

    @property
    def expected_prompt_tokens(self) -> int | None:
        return len(self.prompt) if isinstance(self.prompt, list) else None

    def messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if self.system:
            out.append({"role": "system", "content": self.system})
        if self.image is None:
            out.append({"role": "user", "content": self.prompt})
        else:
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": self.prompt},
                {"type": "image_url", "image_url": {"url": image_url(self.image)}},
            ]
            out.append({"role": "user", "content": parts})
        return out

    def digest(self) -> str:
        payload = [
            self.prompt,
            self.n_predict,
            self.system,
            self.input,
            self.reference,
            _EVAL.dump_python(self.evaluator, mode="json") if self.evaluator else None,
            _image_identity(self.image),
            self.a,
            self.b,
            self.label,
            self.query,
            list(self.documents),
            list(self.relevant),
        ]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def requests_digest(requests: list[Request]) -> str:
    """Identity of a request set (enters the run fingerprint)."""
    h = hashlib.sha256()
    for r in requests:
        h.update(r.digest().encode())
    return h.hexdigest()


_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
         ".gif": "image/gif", ".bmp": "image/bmp"}  # fmt: skip


def image_url(image: str) -> str:
    """Local files are inlined as data URLs (llama-server does not read local paths)."""
    if image.startswith(("http://", "https://", "data:")):
        return image
    path = Path(image)
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def _image_identity(image: str | None) -> str | None:
    """Content hash for local images, so editing an image changes the run identity."""
    if image is None or image.startswith(("http://", "https://", "data:")):
        return image
    return hashlib.sha256(Path(image).read_bytes()).hexdigest()
