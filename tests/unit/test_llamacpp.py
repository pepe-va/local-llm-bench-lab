"""argv generation and introspection, checked against the real c811cb8 llama-bench."""

from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion

from llmbench.config.experiment import BenchPhase
from llmbench.config.llama import LlamaConfig
from llmbench.llamacpp.argv import bench_argv
from llmbench.llamacpp.bench import BenchError, parse_jsonl
from llmbench.llamacpp.introspect import (
    IntrospectionError,
    parse_help,
    parse_version,
    unknown_flags,
)
from tests.conftest import LLAMACPP

FULL = LlamaConfig.model_validate(
    {
        "load": {
            "threads": 4,
            "threads_batch": 4,
            "ctx_size": 4096,
            "batch_size": 512,
            "ubatch_size": 256,
            "cpu_mask": "0xf",
            "cpu_strict": True,
            "poll": 0,
            "prio": 2,
            "numa": "distribute",
            "load_mode": "mmap+mlock",
            "lazy_mode": "off",
            "repack": False,
            "no_host": True,
            "flash_attn": "on",
            "fit_target_mib": 512,
            "fit_min_ctx": 2048,
        },
        "kv": {"cache_type_k": "q8_0", "cache_type_v": "q4_0", "offload": False},
        "gpu": {
            "n_gpu_layers": "all",
            "split_mode": "tensor",
            "main_gpu": 0,
            "tensor_split": [3, 1],
            "override_tensor": ["exps=CPU", "blk\\.1\\.=CPU"],
            "n_cpu_moe": 4,
            "devices": ["CUDA0", "CUDA1"],
            "op_offload": False,
        },
        "extra_args": {"bench": ["--progress"]},
    }
)
BENCH = BenchPhase(n_prompt=[128, 512], n_gen=[32], n_depth=[0, 4096], pg=[(512, 128)])


def test_help_parsing(bench_help: str) -> None:
    flags = parse_help(bench_help)
    for f in (
        "-m",
        "-p",
        "-ctk",
        "--cache-type-k",
        "-lm",
        "--load-mode",
        "-fa",
        "-ot",
        "-nkvo",
        "--repack",
        "--no-host",
        "-pg",
        "--no-warmup",
        "-fitt",
        "--fit-ctx",
    ):
        assert f in flags, f
    assert "-mmp" not in flags  # removed upstream in favour of --load-mode
    assert "-1" not in flags  # placeholders such as <-1|0|1|2|3> are not flags


def test_server_help_parsing() -> None:
    flags = parse_help((LLAMACPP / "llama-server.help.txt").read_text())
    for f in ("-tb", "--threads-batch", "-Cr", "-kvo", "-nkvo", "--swa-full", "-cmoe", "-ncffn"):
        assert f in flags, f


def test_every_generated_flag_exists_in_the_real_binary(bench_help: str) -> None:
    argv = bench_argv(Path("llama-bench"), Path("m.gguf"), FULL, BENCH)
    assert unknown_flags(argv.args, parse_help(bench_help)) == []


def test_full_argv(snapshot: SnapshotAssertion) -> None:
    argv = bench_argv(Path("llama-bench"), Path("m.gguf"), FULL, BENCH)
    assert argv.args == snapshot
    assert argv.ignored == ["load.threads_batch", "load.ctx_size"]


def test_defaults_pass_no_tuning_flags() -> None:
    argv = bench_argv(Path("llama-bench"), Path("m.gguf"), LlamaConfig(), BenchPhase())
    assert argv.args == [
        "llama-bench", "-m", "m.gguf", "-o", "jsonl", "-r", "5",
        "-p", "512", "-n", "128", "-d", "0",
    ]  # fmt: skip
    assert argv.ignored == []


def test_unknown_flags_spots_typos(bench_help: str) -> None:
    supported = parse_help(bench_help)
    assert unknown_flags(["llama-bench", "-t", "4", "-mmp", "0", "--prio", "-1"], supported) == [
        "-mmp"
    ]


def test_version_formats() -> None:
    v = parse_version((LLAMACPP / "llama-bench.version.txt").read_text())
    assert (v.version, v.build_number, v.commit) == ("0.6.0-dev", 1, "c811cb8")
    old = parse_version("version: 4589 (a1b2c3d)\nbuilt with cc for aarch64")
    assert (old.build_number, old.commit) == (4589, "a1b2c3d")
    with pytest.raises(IntrospectionError):
        parse_version("nonsense")


def test_parse_real_jsonl() -> None:
    ms = parse_jsonl((LLAMACPP / "llama-bench.jsonl").read_text())
    assert [m.label for m in ms] == ["pp64", "tg16", "pp64@d32", "tg16@d32"]
    assert ms[0].test == "pp"
    assert len(ms[0].samples_ts) == 3
    assert ms[0].raw["build_commit"] == "c811cb8"


def test_parse_ignores_noise_and_rejects_garbage() -> None:
    assert parse_jsonl("loading model...\n\n") == []
    with pytest.raises(BenchError, match="invalid JSONL"):
        parse_jsonl("{not json")
