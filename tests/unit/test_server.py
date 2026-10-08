"""llama-server argv, help parsing, synthetic data, latency aggregates and planning."""

import itertools
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from llmbench.config.experiment import Experiment, LatencyPhase
from llmbench.config.llama import LlamaConfig
from llmbench.config.suite import Suite, Synthetic
from llmbench.data import synthetic
from llmbench.engine.phases.latency import effective_sampling
from llmbench.engine.planner import latency_fingerprint, make_plan
from llmbench.llamacpp.argv import server_argv
from llmbench.llamacpp.introspect import parse_help, unknown_flags
from llmbench.llamacpp.server import parse_prometheus
from llmbench.results.models import RequestResult
from llmbench.results.stats import latency_metrics, percentile
from llmbench.telemetry.peak import PeakMemory
from tests.conftest import LLAMACPP

SERVER_HELP = (LLAMACPP / "llama-server.help.txt").read_text()

FULL = LlamaConfig.model_validate(
    {
        "load": {
            "threads": 4, "threads_batch": 4, "ctx_size": 4096, "batch_size": 512,
            "ubatch_size": 256, "cpu_mask": "0xf", "cpu_range": "0-3", "cpu_strict": True,
            "poll": 0, "prio": 2, "cpu_mask_batch": "0xf", "cpu_range_batch": "0-3",
            "cpu_strict_batch": True, "poll_batch": False, "prio_batch": 1, "numa": "distribute",
            "load_mode": "mmap", "lazy_mode": "off", "repack": False, "no_host": True,
            "flash_attn": "on", "fit_target_mib": 512, "fit_min_ctx": 2048,
        },
        "kv": {"cache_type_k": "q8_0", "cache_type_v": "q8_0", "offload": False,
               "swa_full": True, "unified": True},
        "gpu": {"n_gpu_layers": "all", "split_mode": "layer", "main_gpu": 0,
                "tensor_split": [1, 1], "override_tensor": ["exps=CPU"], "cpu_moe": True,
                "n_cpu_moe": 2, "n_cpu_ffn": 1, "devices": ["CPU"], "op_offload": False},
        "rope": {"scaling": "yarn", "scale": 4, "freq_base": 10000, "freq_scale": 0.25,
                 "yarn_orig_ctx": 4096, "yarn_ext_factor": 1, "yarn_attn_factor": 1,
                 "yarn_beta_slow": 1, "yarn_beta_fast": 32},
        "speculative": {
            "types": ["draft-simple", "ngram-mod"], "draft_model": "d.gguf", "draft_n_max": 16,
            "draft_n_min": 2, "draft_p_min": 0.8, "draft_p_split": 0.1,
            "draft_sampling": "greedy", "draft_n_gpu_layers": 0, "draft_threads": 2,
            "draft_threads_batch": 2, "draft_cache_type_k": "q8_0", "draft_cache_type_v": "q8_0",
            "draft_devices": ["CPU"], "ngram_mod_n_min": 1, "ngram_mod_n_max": 8,
            "ngram_mod_n_match": 24, "ngram_simple_size_n": 3, "ngram_simple_size_m": 4,
            "ngram_simple_min_hits": 1,
        },
        "server": {
            "parallel": 2, "cont_batching": True, "cache_reuse": 256, "cache_ram_mib": 1024,
            "ctx_checkpoints": 2, "context_shift": False, "keep": -1, "warmup": False,
            "special": True, "jinja": True, "chat_template": "chatml",
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning": "off",
            "reasoning_format": "deepseek", "reasoning_budget": 0,
            "lora": [{"path": "a.gguf", "scale": 0.5}], "threads_http": 2, "timeout_s": 600,
        },
        "extra_args": {"server": ["--log-jsonl"]},
    }
)  # fmt: skip


# -- argv & help ---------------------------------------------------------------------


def test_every_server_flag_exists_in_the_real_binary() -> None:
    argv = server_argv(Path("s"), Path("m.gguf"), FULL, port=8080, draft_model=Path("d.gguf"))
    assert unknown_flags(argv.args, parse_help(SERVER_HELP)) == []
    assert argv.ignored == []


def test_server_argv_snapshot(snapshot: SnapshotAssertion) -> None:
    argv = server_argv(Path("s"), Path("m.gguf"), FULL, port=8080, draft_model=Path("d.gguf"))
    assert argv.args == snapshot


def test_server_is_local_only_and_defaults_are_minimal() -> None:
    argv = server_argv(Path("s"), Path("m.gguf"), LlamaConfig(), port=9000)
    assert argv.args == [
        "s", "-m", "m.gguf", "--host", "127.0.0.1", "--port", "9000",
        "--no-webui", "--metrics", "-np", "1",
    ]  # fmt: skip


def test_unresolved_draft_model_is_reported() -> None:
    cfg = LlamaConfig.model_validate({"speculative": {"draft_model": "hf://o/r:Q4_0"}})
    assert server_argv(Path("s"), Path("m"), cfg, port=1).ignored == ["speculative.draft_model"]


def test_help_parser_drops_removed_flags_and_continuations() -> None:
    flags = parse_help(SERVER_HELP)
    assert "--spec-draft-n-max" in flags
    assert "--spec-ngram-mod-n-max" in flags  # only mentioned in a removed flag's description
    assert "--draft-max" not in flags  # "the argument has been removed"
    assert "--spec-ngram-size-n" not in flags


def test_parse_prometheus() -> None:
    text = "# HELP x\nllamacpp:prompt_tokens_total 12\nllamacpp:n_busy_slots_per_decode 0.5\nbad"
    assert parse_prometheus(text) == {"prompt_tokens_total": 12, "n_busy_slots_per_decode": 0.5}


# -- synthetic data ----------------------------------------------------------------------


def fake_tokenize(text: str) -> list[int]:
    return [hash(w) % 1000 + 2 for w in text.split()]


def syn(**kw: object) -> Synthetic:
    return Synthetic.model_validate({"prompt_tokens": 128, "output_tokens": 32} | kw)


def test_exact_prompt_lengths_with_bos() -> None:
    spec = syn(prompt_tokens={"dist": "uniform", "min": 50, "max": 300}, count=25)
    reqs = synthetic.build(spec, tokenize=fake_tokenize, bos=[1], suite="s")
    rng = random.Random(spec.seed)
    shapes = [
        (synthetic.sample(spec.prompt_tokens, rng), synthetic.sample(spec.output_tokens, rng))
        for _ in range(25)
    ]
    assert [len(r.prompt) for r in reqs] == [n for n, _ in shapes]  # BOS counted in the total
    assert all(50 <= len(r.prompt) <= 300 for r in reqs)
    assert all(isinstance(r.prompt, list) and r.prompt[0] == 1 for r in reqs)
    assert {r.n_predict for r in reqs} == {32}


def test_synthetic_is_deterministic_and_seed_sensitive() -> None:
    a = synthetic.build(syn(), tokenize=fake_tokenize, bos=[], suite="s")
    b = synthetic.build(syn(), tokenize=fake_tokenize, bos=[], suite="s")
    c = synthetic.build(syn(seed=1), tokenize=fake_tokenize, bos=[], suite="s")
    assert [r.prompt for r in a] == [r.prompt for r in b]
    assert [r.prompt for r in a] != [r.prompt for r in c]


def test_distributions() -> None:
    rng = random.Random(0)
    normal = syn(output_tokens={"dist": "normal", "mean": 100, "std": 80, "min": 10, "max": 150})
    values = [synthetic.sample(normal.output_tokens, rng) for _ in range(500)]
    assert min(values) >= 10
    assert max(values) <= 150
    assert syn(output_tokens={"value": 7}).output_tokens.dist == "fixed"
    with pytest.raises(ValueError, match="min must be <= max"):
        syn(prompt_tokens={"dist": "uniform", "min": 10, "max": 5})


def test_corpus_file(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text("alpha beta gamma delta")
    reqs = synthetic.build(syn(corpus=corpus, count=3), tokenize=fake_tokenize, bos=[], suite="s")
    assert all(len(r.prompt) == 128 for r in reqs)


def test_prompt_shorter_than_bos_fails() -> None:
    with pytest.raises(ValueError, match="no room after the BOS"):
        synthetic.build(syn(prompt_tokens=1), tokenize=fake_tokenize, bos=[1], suite="s")


# -- aggregates ------------------------------------------------------------------------


def test_percentile() -> None:
    assert percentile([], 50) is None
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([5.0], 95) == 5.0


def test_latency_metrics() -> None:
    results = [
        RequestResult("a", True, ttft_ms=10, e2e_ms=100, prompt_n=5, predicted_n=10,
                      prompt_tps=500, gen_tps=100, itl_ms=[1, 2, 3]),
        RequestResult("b", True, ttft_ms=30, e2e_ms=300, prompt_n=5, predicted_n=10,
                      prompt_tps=700, gen_tps=80, itl_ms=[3, 4]),
        RequestResult("c", False, error="boom"),
    ]  # fmt: skip
    m = latency_metrics(results)
    assert (m["requests"], m["requests_failed"]) == (3, 1)
    assert m["ttft_ms_p50"] == 20
    assert m["gen_tps_mean"] == 90
    assert m["itl_ms_max"] == 4
    assert m["gen_tokens_total"] == 20


# -- planning & identity -------------------------------------------------------------------


def experiment(tmp_path: Path, **phases: object) -> Experiment:
    suite = tmp_path / "s.yaml"
    suite.write_text(
        "name: s\nsource: {synthetic: {prompt_tokens: 64, output_tokens: 8, count: 3}}\n"
    )
    return Experiment.model_validate(
        {
            "experiment": {"name": "t"},
            "llama_cpp": {"builds": [{"id": "b", "bin_dir": "/x"}]},
            "models": [{"id": "m1", "source": "a.gguf"}, {"id": "m2", "source": "b.gguf"}],
            "matrix": {"load.threads": [2, 4], "sampling.temperature": [0, 0.8]},
            "phases": {"latency": {"enabled": True, "suites": [str(suite)], "passes": 2}} | phases,
        }
    )


def test_bench_dedupes_sampling_only_variants(tmp_path: Path) -> None:
    p = make_plan(experiment(tmp_path))
    assert len(p.bench_runs) == 2 * 2  # models x threads (temperature does not matter)
    assert len(p.latency_runs) == 2 * 4
    assert p.servers == 2 * 2  # temperature variants share a server
    assert p.total_requests == 8 * 3 * 2


def test_latency_runs_are_grouped_by_server(tmp_path: Path) -> None:
    keys = [r.server_key() for r in make_plan(experiment(tmp_path)).latency_runs]
    changes = sum(1 for a, b in itertools.pairwise(keys) if a != b)
    assert changes == len(set(keys)) - 1


def test_latency_fingerprint(tmp_path: Path) -> None:
    run = make_plan(experiment(tmp_path)).latency_runs[0]
    assert run.suite is not None

    def fp(**kw: object) -> str:
        args: dict[str, object] = {
            "build": {"commit": "c"}, "model_sha256": "m", "draft_sha256": None,
            "config": run.config, "sampling": {"temperature": 0}, "suite": run.suite,
            "latency": LatencyPhase(enabled=True, suites=[Path("x")]),
        } | kw  # fmt: skip
        return latency_fingerprint(**args)  # type: ignore[arg-type]

    assert fp() == fp(latency=LatencyPhase(enabled=True, suites=[Path("y")], request_timeout_s=5))
    assert fp() != fp(sampling={"temperature": 0.8})
    assert fp() != fp(draft_sha256="d")
    assert fp() != fp(latency=LatencyPhase(enabled=True, suites=[Path("x")], passes=3))


def test_effective_sampling(tmp_path: Path) -> None:
    run = make_plan(experiment(tmp_path)).latency_runs[1]
    assert effective_sampling(run) == {
        "temperature": run.config.sampling.temperature,
        "ignore_eos": True,
    }
    suite = Suite.model_validate(
        {"name": "s", "source": {"synthetic": {"prompt_tokens": 8, "output_tokens": 8}},
         "sampling": {"ignore_eos": False, "top_k": 1}}
    )  # fmt: skip
    assert effective_sampling(replace(run, suite=suite))["ignore_eos"] is False


def test_latency_needs_suites() -> None:
    with pytest.raises(ValueError, match="lists no suites"):
        LatencyPhase(enabled=True)


# -- peak memory -------------------------------------------------------------------------


def test_peak_memory_ignores_parent_size() -> None:
    ballast = bytearray(200 * 1024 * 1024)
    for i in range(0, len(ballast), 4096):
        ballast[i] = 1
    code = (
        "import time; x = bytearray(40 * 1024 * 1024); "
        "x[::4096] = b'1' * len(x[::4096]); time.sleep(0.5)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code])
    peak = PeakMemory(proc.pid, interval_s=0.01)
    proc.wait()
    value = peak.stop()
    assert value is not None
    assert 40 <= value < 150  # the child's own peak, not the 200 MB parent
