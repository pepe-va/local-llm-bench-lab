"""Synthetic requests with exact input-token counts, built with the model's own tokenizer.

Approach (same idea as GuideLLM's synthetic data): tokenize a long deterministic corpus once
per model, then cut token windows of the requested length at seeded random offsets and send
them as token arrays. llama-server uses token arrays verbatim (no BOS added), so the model's
BOS (if any) is prepended and counted. Prompt caching is disabled per request by the client.
"""

import random
from collections.abc import Callable
from pathlib import Path

from llmbench.config.suite import Distribution, Fixed, Normal, Synthetic, Uniform
from llmbench.data.requests import Request

# A fixed vocabulary keeps the corpus identical across machines and Python versions.
_WORDS = (  # noqa: SIM905 (a text block reads better than 150 list items)
    "the of and to in a is that for it as was with be by on not he this are or his from at "
    "which but have an they you were her she there their one all we can has been if more when "
    "will would who so no time people year way day man thing woman life child world school "
    "state family student group country problem hand part place case week company system "
    "program question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind head house "
    "service friend father power hour game line end member law car city community name "
    "president team minute idea kid body information back parent face others level office "
    "door health person art war history party result change morning reason research girl guy "
    "moment air teacher force education river mountain light garden window letter music paper"
).split()

type Tokenize = Callable[[str], list[int]]


def corpus_text(corpus: str | Path, n_words: int, seed: int) -> str:
    if isinstance(corpus, Path):
        text = corpus.expanduser().read_text(encoding="utf-8")
        while len(text.split()) < n_words:
            text += "\n" + text
        return text
    rng = random.Random(seed)
    words = [rng.choice(_WORDS) for _ in range(n_words)]
    # sentences of 8-20 words, so the tokenizer sees punctuation and capitals too
    out: list[str] = []
    i = 0
    while i < len(words):
        n = rng.randint(8, 20)
        sentence = words[i : i + n]
        out.append(" ".join(sentence).capitalize() + ".")
        i += n
    return " ".join(out)


def sample(dist: Distribution, rng: random.Random) -> int:
    match dist:
        case Fixed():
            return dist.value
        case Uniform():
            return rng.randint(dist.min, dist.max)
        case Normal():
            value = round(rng.gauss(dist.mean, dist.std))
            value = max(dist.min, value)
            return min(dist.max, value) if dist.max is not None else value


def shapes(spec: Synthetic) -> list[tuple[int, int]]:
    """(prompt tokens, output tokens) of every request, BOS included, without a tokenizer."""
    return _shapes(spec, random.Random(spec.seed))


def _shapes(spec: Synthetic, rng: random.Random) -> list[tuple[int, int]]:
    return [
        (sample(spec.prompt_tokens, rng), sample(spec.output_tokens, rng))
        for _ in range(spec.count)
    ]


def build(
    spec: Synthetic,
    *,
    tokenize: Tokenize,
    bos: list[int],
    suite: str,
) -> list[Request]:
    rng = random.Random(spec.seed)
    sizes = _shapes(spec, rng)  # same draws as `shapes()`; offsets continue from this rng
    longest = max(n for n, _ in sizes)

    n_words = max(256, longest * 2)
    tokens = tokenize(corpus_text(spec.corpus, n_words, spec.seed))
    while len(tokens) < longest * 2:
        n_words *= 2
        tokens = tokenize(corpus_text(spec.corpus, n_words, spec.seed))

    requests: list[Request] = []
    for i, (n_in, n_out) in enumerate(sizes):
        body = n_in - len(bos)
        if body < 1:
            raise ValueError(f"prompt_tokens={n_in} leaves no room after the BOS token")
        offset = rng.randrange(0, len(tokens) - body)
        requests.append(
            Request(
                id=f"{suite}/{i:05d}",
                prompt=bos + tokens[offset : offset + body],
                n_predict=n_out,
            )
        )
    return requests
