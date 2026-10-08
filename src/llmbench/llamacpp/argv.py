"""Config objects -> argv. The only module that knows llama.cpp flag spellings.

Every field set in the config either becomes a flag or is reported in `ignored`,
so the user sees exactly which knobs a tool did not receive.
"""

from dataclasses import dataclass, field
from pathlib import Path

from llmbench.config.experiment import BenchPhase
from llmbench.config.llama import LlamaConfig

_ALL_LAYERS = 999  # llama-bench takes a number; "all" means more layers than any model has


@dataclass(frozen=True)
class Argv:
    args: list[str]
    ignored: list[str] = field(default_factory=list)
    """Dotted config paths that were set but have no equivalent in this tool."""


def _b(v: bool) -> str:
    return "1" if v else "0"


def bench_argv(binary: Path, model: Path, cfg: LlamaConfig, bench: BenchPhase) -> Argv:
    a: list[str] = [str(binary), "-m", str(model), "-o", "jsonl", "-r", str(bench.repetitions)]
    ignored: list[str] = []

    a += ["-p", ",".join(map(str, bench.n_prompt)), "-n", ",".join(map(str, bench.n_gen))]
    a += ["-d", ",".join(map(str, bench.n_depth))]
    for pp, tg in bench.pg:
        a += ["-pg", f"{pp},{tg}"]
    if not bench.warmup:
        a.append("--no-warmup")
    if bench.delay_s is not None:
        a += ["--delay", str(bench.delay_s)]

    ld = cfg.load
    for flag, value in (
        ("-t", ld.threads),
        ("-b", ld.batch_size),
        ("-ub", ld.ubatch_size),
        ("-C", ld.cpu_mask),
        ("--poll", ld.poll),
        ("--prio", ld.prio),
        ("--numa", ld.numa),
        ("-lm", ld.load_mode),
        ("-lzm", ld.lazy_mode),
        ("-fa", ld.flash_attn),
        ("-fitt", ld.fit_target_mib),
        ("-fitc", ld.fit_min_ctx),
    ):
        if value is not None:
            a += [flag, str(value)]
    for flag, value in (
        ("--cpu-strict", ld.cpu_strict),
        ("--repack", ld.repack),
        ("--no-host", ld.no_host),
    ):
        if value is not None:
            a += [flag, _b(value)]
    for name in (
        "threads_batch",
        "ctx_size",
        "cpu_range",
        "cpu_mask_batch",
        "cpu_range_batch",
        "cpu_strict_batch",
        "poll_batch",
        "prio_batch",
    ):
        if getattr(ld, name) is not None:
            ignored.append(f"load.{name}")

    kv = cfg.kv
    if kv.cache_type_k is not None:
        a += ["-ctk", kv.cache_type_k]
    if kv.cache_type_v is not None:
        a += ["-ctv", kv.cache_type_v]
    if kv.offload is not None:
        a += ["-nkvo", _b(not kv.offload)]
    ignored += [f"kv.{n}" for n in ("swa_full", "unified") if getattr(kv, n) is not None]

    g = cfg.gpu
    if g.n_gpu_layers == "all":
        a += ["-ngl", str(_ALL_LAYERS)]
    elif g.n_gpu_layers == "auto":
        ignored.append("gpu.n_gpu_layers")  # llama-bench has no "auto"; its default applies
    elif g.n_gpu_layers is not None:
        a += ["-ngl", str(g.n_gpu_layers)]
    if g.split_mode is not None:
        a += ["-sm", g.split_mode]
    if g.main_gpu is not None:
        a += ["-mg", str(g.main_gpu)]
    if g.tensor_split is not None:
        a += ["-ts", "/".join(f"{x:g}" for x in g.tensor_split)]
    if g.override_tensor:
        a += ["-ot", ";".join(g.override_tensor)]
    if g.n_cpu_moe is not None:
        a += ["-ncmoe", str(g.n_cpu_moe)]
    if g.devices is not None:
        a += ["-dev", "/".join(g.devices)]
    if g.op_offload is not None:
        a += ["-nopo", _b(not g.op_offload)]
    ignored += [f"gpu.{n}" for n in ("cpu_moe", "n_cpu_ffn") if getattr(g, n) is not None]

    a += cfg.extra_args.bench
    return Argv(a, ignored)
