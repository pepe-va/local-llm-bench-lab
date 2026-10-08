"""Config objects -> argv. The only module that knows llama.cpp flag spellings.

Every field set in the config either becomes a flag or is reported in `ignored`,
so the user sees exactly which knobs a tool did not receive.
"""

import json
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


def _flag(a: list[str], flag: str, value: object) -> None:
    if value is not None:
        a += [flag, str(value)]


def _toggle(a: list[str], value: bool | None, on: str, off: str | None) -> None:
    """Boolean with explicit on/off spellings; `None` passes nothing."""
    if value is True:
        a.append(on)
    elif value is False and off is not None:
        a.append(off)


def server_argv(
    binary: Path,
    model: Path,
    cfg: LlamaConfig,
    *,
    port: int,
    draft_model: Path | None = None,
) -> Argv:
    """llama-server bound to 127.0.0.1 (never exposed; see PRD §37)."""
    a: list[str] = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port)]
    a += ["--no-webui", "--metrics"]
    ignored: list[str] = []

    ld = cfg.load
    for flag, value in (
        ("-t", ld.threads),
        ("-tb", ld.threads_batch),
        ("-c", ld.ctx_size),
        ("-b", ld.batch_size),
        ("-ub", ld.ubatch_size),
        ("-C", ld.cpu_mask),
        ("-Cr", ld.cpu_range),
        ("--poll", ld.poll),
        ("--prio", ld.prio),
        ("-Cb", ld.cpu_mask_batch),
        ("-Crb", ld.cpu_range_batch),
        ("--prio-batch", ld.prio_batch),
        ("--numa", ld.numa),
        ("-lm", ld.load_mode),
        ("-lzm", ld.lazy_mode),
        ("-fa", ld.flash_attn),
        ("-fitt", ld.fit_target_mib),
        ("-fitc", ld.fit_min_ctx),
    ):
        _flag(a, flag, value)
    for flag, value in (
        ("--cpu-strict", ld.cpu_strict),
        ("--cpu-strict-batch", ld.cpu_strict_batch),
        ("--poll-batch", ld.poll_batch),
    ):
        if value is not None:
            a += [flag, _b(value)]
    _toggle(a, ld.repack, "--repack", "--no-repack")
    _toggle(a, ld.no_host, "--no-host", None)

    kv = cfg.kv
    _flag(a, "-ctk", kv.cache_type_k)
    _flag(a, "-ctv", kv.cache_type_v)
    _toggle(a, kv.offload, "-kvo", "-nkvo")
    _toggle(a, kv.swa_full, "--swa-full", None)
    _toggle(a, kv.unified, "-kvu", "-no-kvu")

    g = cfg.gpu
    _flag(a, "-ngl", g.n_gpu_layers)
    _flag(a, "-sm", g.split_mode)
    _flag(a, "-mg", g.main_gpu)
    if g.tensor_split is not None:
        a += ["-ts", ",".join(f"{x:g}" for x in g.tensor_split)]
    if g.override_tensor:
        a += ["-ot", ",".join(g.override_tensor)]
    _toggle(a, g.cpu_moe, "-cmoe", None)
    _flag(a, "-ncmoe", g.n_cpu_moe)
    _flag(a, "-ncffn", g.n_cpu_ffn)
    if g.devices is not None:
        a += ["-dev", ",".join(g.devices)]
    _toggle(a, g.op_offload, "--op-offload", "--no-op-offload")

    r = cfg.rope
    for flag, value in (
        ("--rope-scaling", r.scaling),
        ("--rope-scale", r.scale),
        ("--rope-freq-base", r.freq_base),
        ("--rope-freq-scale", r.freq_scale),
        ("--yarn-orig-ctx", r.yarn_orig_ctx),
        ("--yarn-ext-factor", r.yarn_ext_factor),
        ("--yarn-attn-factor", r.yarn_attn_factor),
        ("--yarn-beta-slow", r.yarn_beta_slow),
        ("--yarn-beta-fast", r.yarn_beta_fast),
    ):
        _flag(a, flag, value)

    sp = cfg.speculative
    if sp.types is not None:
        a += ["--spec-type", ",".join(sp.types)]
    if draft_model is not None:
        a += ["-md", str(draft_model)]
    elif sp.draft_model is not None:
        ignored.append("speculative.draft_model")  # unresolved: caller must pass the path
    for flag, value in (
        ("--spec-draft-n-max", sp.draft_n_max),
        ("--spec-draft-n-min", sp.draft_n_min),
        ("--spec-draft-p-min", sp.draft_p_min),
        ("--spec-draft-p-split", sp.draft_p_split),
        ("--spec-draft-sampling", sp.draft_sampling),
        ("-ngld", sp.draft_n_gpu_layers),
        ("-td", sp.draft_threads),
        ("-tbd", sp.draft_threads_batch),
        ("-ctkd", sp.draft_cache_type_k),
        ("-ctvd", sp.draft_cache_type_v),
        ("--spec-ngram-mod-n-min", sp.ngram_mod_n_min),
        ("--spec-ngram-mod-n-max", sp.ngram_mod_n_max),
        ("--spec-ngram-mod-n-match", sp.ngram_mod_n_match),
        ("--spec-ngram-simple-size-n", sp.ngram_simple_size_n),
        ("--spec-ngram-simple-size-m", sp.ngram_simple_size_m),
        ("--spec-ngram-simple-min-hits", sp.ngram_simple_min_hits),
    ):
        _flag(a, flag, value)
    if sp.draft_devices is not None:
        a += ["-devd", ",".join(sp.draft_devices)]

    s = cfg.server
    for flag, value in (
        ("-np", s.parallel),
        ("--cache-reuse", s.cache_reuse),
        ("-cram", s.cache_ram_mib),
        ("-ctxcp", s.ctx_checkpoints),
        ("--keep", s.keep),
        ("--chat-template", s.chat_template),
        ("--chat-template-file", s.chat_template_file),
        ("-rea", s.reasoning),
        ("--reasoning-format", s.reasoning_format),
        ("--reasoning-budget", s.reasoning_budget),
        ("--threads-http", s.threads_http),
        ("-to", s.timeout_s),
    ):
        _flag(a, flag, value)
    _toggle(a, s.cont_batching, "-cb", "-nocb")
    _toggle(a, s.context_shift, "--context-shift", "--no-context-shift")
    _toggle(a, s.warmup, "--warmup", "--no-warmup")
    _toggle(a, s.special, "-sp", None)
    _toggle(a, s.jinja, "--jinja", "--no-jinja")
    if s.chat_template_kwargs is not None:
        a += ["--chat-template-kwargs", json.dumps(s.chat_template_kwargs, sort_keys=True)]
    if s.lora:
        a += ["--lora-scaled", ",".join(f"{lo.path}:{lo.scale:g}" for lo in s.lora)]

    a += cfg.extra_args.server
    return Argv(a, ignored)
