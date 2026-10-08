"""Serving plan and metrics, lm-eval adapter, LLM judge."""

import json
from pathlib import Path

import pytest

from llmbench.config.experiment import Experiment, JudgePhase, LmEvalPhase
from llmbench.engine.judge import _parse, prompt, schema  # pyright: ignore[reportPrivateUsage]
from llmbench.engine.phases.lm_eval import flatten_results, lm_eval_argv
from llmbench.engine.planner import make_plan
from llmbench.results.models import RequestResult
from llmbench.results.stats import serving_metrics


def experiment(tmp_path: Path, **phases: object) -> Experiment:
    (tmp_path / "s.yaml").write_text(
        "name: s\nsource: {synthetic: {prompt_tokens: 8, output_tokens: 4, count: 5}}\n"
    )
    return Experiment.model_validate(
        {
            "experiment": {"name": "t"},
            "llama_cpp": {"builds": [{"id": "b", "bin_dir": "/x"}]},
            "models": [{"id": "m", "source": "m.gguf"}],
            "defaults": {"server": {"parallel": 2}},
            "phases": {"bench": {"enabled": False}} | phases,
        }
    )


def test_serving_plan_runs_one_level_each_with_enough_slots(tmp_path: Path) -> None:
    exp = experiment(
        tmp_path,
        serving={
            "enabled": True,
            "suites": [str(tmp_path / "s.yaml")],
            "concurrency": [1, 4, 8],
            "requests": 40,
        },
    )
    p = make_plan(exp)
    assert [r.concurrency for r in p.serving_runs] == [1, 4, 8]
    assert {r.config.server.parallel for r in p.serving_runs} == {8}  # raised from 2
    assert p.servers == 1
    assert p.requests("serving") == 120


def test_lm_eval_runs_skip_specialised_models(tmp_path: Path) -> None:
    exp = experiment(tmp_path, lm_eval={"enabled": True, "tasks": ["gsm8k"]})
    assert [r.phase for r in make_plan(exp).runs] == ["lm_eval"]


def test_serving_metrics() -> None:
    results = [
        RequestResult("a", True, prompt_n=10, predicted_n=20),
        RequestResult("b", True, prompt_n=10, predicted_n=20),
        RequestResult("c", False, error="x"),
    ]
    m = serving_metrics(results, 2.0, 4)
    assert (m["requests_per_s"], m["gen_tps_aggregate"], m["prompt_tps_aggregate"]) == (1, 20, 10)
    assert m["error_rate"] == pytest.approx(1 / 3)
    assert serving_metrics([], 0, 1)["requests_per_s"] is None


def test_lm_eval_argv_and_results() -> None:
    cfg = LmEvalPhase(
        enabled=True, tasks=["gsm8k", "ifeval"], limit=10, num_fewshot=5, extra_args=["--seed", "1"]
    )
    argv = lm_eval_argv(cfg, "http://127.0.0.1:9", "qwen", Path("/out"))
    joined = " ".join(argv)
    assert "--model local-chat-completions" in joined
    assert "base_url=http://127.0.0.1:9/v1/chat/completions,model=qwen" in joined
    assert "--tasks gsm8k,ifeval" in joined
    assert argv[-3:] == ["--apply_chat_template", "--seed", "1"]
    raw = LmEvalPhase(enabled=True, model="local-completions")
    assert "/v1/completions" in " ".join(lm_eval_argv(raw, "http://h", "m", Path("o")))
    assert "--apply_chat_template" not in lm_eval_argv(raw, "http://h", "m", Path("o"))
    gsm8k = {"alias": "gsm8k", "exact_match,strict-match": 0.4, "acc,none": 0.7}
    data = {"results": {"gsm8k": gsm8k | {"exact_match_stderr,strict-match": 0.1}}}
    assert flatten_results(data) == {
        "lm_eval:gsm8k.exact_match.strict-match": 0.4,
        "lm_eval:gsm8k.acc": 0.7,
    }


def test_judge_prompt_schema_and_parsing() -> None:
    s = schema(["grammar", "clarity"])
    assert s["required"] == ["grammar", "clarity", "explanation"]
    assert s["properties"]["grammar"] == {"type": "integer", "minimum": 1, "maximum": 5}
    text = prompt(
        {"prompt_text": "Corrige", "content": "Hola", "reference": "Hola."}, ["grammar"], "be harsh"
    )
    assert "TASK:\nCorrige" in text
    assert "REFERENCE" in text
    assert "Additional rubric: be harsh" in text
    assert _parse(
        json.dumps({"grammar": 4, "clarity": 9, "explanation": "ok"}), ["grammar", "clarity"]
    ) == {
        "grammar": (4.0, "ok"),
        "clarity": (None, "ok"),  # out of range is not a score
    }
    assert _parse("not json", ["grammar"]) is None


def test_judge_needs_a_model() -> None:
    with pytest.raises(ValueError, match="has no `model`"):
        JudgePhase(enabled=True)
