"""Embeddings/rerank metrics, vision requests and modality suites."""

import base64
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from llmbench.config.experiment import Experiment
from llmbench.config.llama import LlamaConfig
from llmbench.config.suite import Suite
from llmbench.data.requests import Request, image_url
from llmbench.engine.planner import make_plan
from llmbench.evaluation.retrieval import cosine, mrr, ndcg_at_k, pearson, recall_at_k, spearman
from llmbench.llamacpp.argv import server_argv
from llmbench.llamacpp.introspect import parse_help, unknown_flags
from llmbench.results.models import RequestResult
from llmbench.results.stats import retrieval_metrics
from tests.conftest import LLAMACPP


def test_similarity_and_correlation() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0)
    assert cosine([0, 0], [1, 1]) == 0
    assert spearman([0.1, 0.5, 0.9], [1, 2, 3]) == pytest.approx(1)
    assert spearman([0.9, 0.5, 0.1], [1, 2, 3]) == pytest.approx(-1)
    assert spearman([0.1, 0.1, 0.9], [1, 2, 3]) == pytest.approx(0.866, abs=1e-3)  # ties averaged
    assert pearson([1, 1], [1, 2]) is None
    assert spearman([1.0], [2.0]) is None


def test_ranking_metrics() -> None:
    assert ndcg_at_k([1, 0, 2], {1}, 3) == pytest.approx(1)
    assert ndcg_at_k([0, 1, 2], {1}, 3) == pytest.approx(1 / 1.585, abs=1e-3)
    assert ndcg_at_k([0, 2, 1], {1}, 2) == 0
    assert mrr([2, 0, 1], {1}) == pytest.approx(1 / 3)
    assert recall_at_k([0, 3, 1], {0, 1}, 2) == 0.5


def test_retrieval_aggregates() -> None:
    emb = [
        RequestResult("a", True, similarity=0.9, label=4.5, prompt_tps=100),
        RequestResult("b", True, similarity=0.2, label=0.5, prompt_tps=300),
        RequestResult("c", True, similarity=0.5, label=2.0),
    ]
    m = retrieval_metrics(emb, "embeddings", 10)
    assert m["spearman"] == pytest.approx(1)
    assert m["embed_tps_mean"] == 200
    rr = [
        RequestResult(
            "q1",
            True,
            score=1.0,
            passed=True,
            eval_detail=json.dumps({"mrr": 1.0, "recall@3": 1.0}),
        ),
        RequestResult(
            "q2",
            True,
            score=0.5,
            passed=False,
            eval_detail=json.dumps({"mrr": 0.5, "recall@3": 0.5}),
        ),
    ]
    m = retrieval_metrics(rr, "rerank", 3)
    assert (m["ndcg@3"], m["mrr"], m["recall@3"], m["top1_accuracy"]) == (0.75, 0.75, 0.75, 0.5)


def test_vision_message_inlines_local_images(tmp_path: Path) -> None:
    img = tmp_path / "x.png"
    img.write_bytes(b"\x89PNG fake")
    msgs = Request("r", "What is it?", 8, system="brief", image=str(img)).messages()
    content = msgs[1]["content"]
    assert content[0] == {"type": "text", "text": "What is it?"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG fake"
    assert image_url("https://x/y.png") == "https://x/y.png"
    # editing the image changes the request identity
    before = Request("r", "p", 8, image=str(img)).digest()
    img.write_bytes(b"\x89PNG other")
    assert Request("r", "p", 8, image=str(img)).digest() != before


def test_server_modes_and_mmproj_flags() -> None:
    help_flags = parse_help((LLAMACPP / "llama-server.help.txt").read_text())
    for mode, expected in (
        ("embeddings:mean", ["--embeddings", "--pooling", "mean"]),
        ("embeddings:default", ["--embeddings"]),
        ("rerank", ["--reranking"]),
    ):
        argv = server_argv(Path("s"), Path("m"), LlamaConfig(), port=1, mode=mode)
        assert argv.args[-len(expected) :] == expected or all(e in argv.args for e in expected)
        assert unknown_flags(argv.args, help_flags) == []
    argv = server_argv(Path("s"), Path("m"), LlamaConfig(), port=1, mmproj=Path("p.gguf"))
    assert argv.args[9:11] == ["-mm", "p.gguf"]
    assert (
        "--pooling"
        not in server_argv(
            Path("s"), Path("m"), LlamaConfig(), port=1, mode="embeddings:default"
        ).args
    )


def test_suite_validation_per_endpoint() -> None:
    with pytest.raises(ValidationError, match="embeddings items need a, b"):
        Suite.model_validate(
            {"name": "e", "endpoint": "embeddings", "source": {"items": [{"prompt": "x"}]}}
        )
    synthetic = {"synthetic": {"prompt_tokens": 8, "output_tokens": 8}}
    with pytest.raises(ValidationError, match="synthetic sources only feed"):
        Suite.model_validate({"name": "e", "endpoint": "rerank", "source": synthetic})
    s = Suite.model_validate({"name": "e", "endpoint": "embeddings", "pooling": "cls",
                              "source": {"items": [{"a": "x", "b": "y", "label": 1}]}})  # fmt: skip
    assert s.server_mode == "embeddings:cls"


def test_models_can_be_limited_to_some_suites(tmp_path: Path) -> None:
    names = ("emb", "gen")
    suites = [str(tmp_path / f"{n}.yaml") for n in names]
    for name, endpoint, item in (
        ("emb", "embeddings", "{a: x, b: y, label: 1}"),
        ("gen", "chat", "{prompt: hola}"),
    ):
        (tmp_path / f"{name}.yaml").write_text(
            f"name: {name}\nendpoint: {endpoint}\nsource: {{items: [{item}]}}\n"
        )
    exp = Experiment.model_validate(
        {
            "experiment": {"name": "t"},
            "llama_cpp": {"builds": [{"id": "b", "bin_dir": "/x"}]},
            "models": [
                {"id": "embedder", "source": "e.gguf", "suites": ["emb"]},
                {"id": "chatter", "source": "c.gguf"},
            ],
            "phases": {
                "bench": {"enabled": False},
                "quality": {"enabled": True, "suites": suites},
            },
        }
    )  # fmt: skip
    runs = [(r.model.id, r.suite.name if r.suite else None) for r in make_plan(exp).runs]
    assert runs == [("embedder", "emb"), ("chatter", "emb"), ("chatter", "gen")]
    servers = {r.server_key() for r in make_plan(exp).runs}
    assert len(servers) == 3  # embeddings mode never shares a process with generation
