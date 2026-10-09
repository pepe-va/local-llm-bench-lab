from collections.abc import Callable
from pathlib import Path

from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.config.llama import LlamaConfig
from llmbench.config.loader import load_suite
from llmbench.config.matrix import apply_variant
from llmbench.engine.planner import context_warnings, make_plan
from llmbench.hub.store import ModelNotFoundError, ModelStore
from llmbench.llamacpp import builds
from llmbench.llamacpp.argv import Argv, bench_argv, server_argv
from llmbench.llamacpp.introspect import IntrospectionError, binary_flags, unknown_flags


def validate(experiment: ExperimentArg) -> None:
    """Validate an experiment: schema, matrix, models on disk and flags against each build."""
    exp = load(experiment)
    console.print(f"[green]✓[/] schema and matrix ({len(exp.matrix.expand())} variants)")
    problems = 0

    models = ModelStore(allow_download=False)
    for m in exp.enabled_models:
        try:
            r = models.resolve(m)
            where = f" ({r.repo}@{r.commit[:12]}, {r.origin})" if r.repo and r.commit else ""
            console.print(f"[green]✓[/] model {m.id}: {r.path}{where}")
        except ModelNotFoundError as e:
            problems += 1
            console.print(f"[red]✗[/] {e}")

    variants = [apply_variant(exp.defaults, v) for v in exp.matrix.expand()]
    tools: list[tuple[str, Callable[[Path, LlamaConfig], Argv]]] = []
    if exp.phases.bench.enabled:
        tools.append(("llama-bench", lambda b, c: bench_argv(b, b, c, exp.phases.bench)))
    phases = (exp.phases.latency, exp.phases.quality, exp.phases.serving)
    server_suites = list(dict.fromkeys(s for ph in phases if ph.enabled for s in ph.suites))
    if server_suites or exp.phases.lm_eval.enabled:
        tools.append(
            (
                "llama-server",
                lambda b, c: server_argv(
                    b, b, c, port=0, draft_model=b if c.speculative.draft_model else None
                ),
            )
        )
    for suite in server_suites:
        console.print(f"[green]✓[/] suite {load_suite(suite).name}: {suite}")
    for spec in exp.llama_cpp.builds:
        try:
            rb = builds.resolve(spec)
        except builds.BuildError as e:
            console.print(f"[yellow]•[/] build {spec.id}: flags not checked ({e})")
            continue
        for tool, make in tools:
            problems += _check_tool(rb, tool, make, variants)

    for w in context_warnings(make_plan(exp)):
        console.print(f"[yellow]![/] {w}", highlight=False)
    if problems:
        fail(f"{problems} problem(s) found", code=1)
    console.print("[green]valid[/]")


def _check_tool(
    rb: builds.ResolvedBuild,
    tool: str,
    make: Callable[[Path, LlamaConfig], Argv],
    variants: list[LlamaConfig],
) -> int:
    try:
        supported = binary_flags(rb.tool(tool))
    except IntrospectionError as e:
        console.print(f"[red]✗[/] build {rb.id}: {e}")
        return 1
    bad: set[str] = set()
    ignored: set[str] = set()
    for cfg in variants:
        argv = make(rb.tool(tool), cfg)
        bad |= set(unknown_flags(argv.args, supported))
        ignored |= set(argv.ignored)
    if bad:
        console.print(
            f"[red]✗[/] build {rb.id} ({rb.commit[:12]}) {tool} lacks: {', '.join(sorted(bad))}"
        )
    else:
        console.print(f"[green]✓[/] build {rb.id} ({rb.commit[:12]}): all {tool} flags supported")
    if ignored:
        console.print(f"  [dim]not applicable to {tool}: {', '.join(sorted(ignored))}[/]")
    return 1 if bad else 0
