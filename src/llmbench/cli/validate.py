from llmbench.cli.common import ExperimentArg, console, fail, load
from llmbench.config.matrix import apply_variant
from llmbench.engine.planner import make_plan
from llmbench.hub.store import ModelNotFoundError, ModelStore
from llmbench.llamacpp import builds
from llmbench.llamacpp.argv import bench_argv
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

    bench = exp.phases.bench
    for spec in exp.llama_cpp.builds:
        try:
            rb = builds.resolve(spec)
            supported = binary_flags(rb.tool("llama-bench"))
        except (builds.BuildError, IntrospectionError) as e:
            console.print(f"[yellow]•[/] build {spec.id}: flags not checked ({e})")
            continue
        bad: set[str] = set()
        ignored: set[str] = set()
        for variant in exp.matrix.expand():
            argv = bench_argv(
                rb.tool("llama-bench"),
                rb.bin_dir / "model.gguf",
                apply_variant(exp.defaults, variant),
                bench,
            )
            bad |= set(unknown_flags(argv.args, supported))
            ignored |= set(argv.ignored)
        if bad:
            problems += 1
            console.print(
                f"[red]✗[/] build {spec.id} ({rb.commit[:12]}) lacks: {', '.join(sorted(bad))}"
            )
        else:
            console.print(
                f"[green]✓[/] build {spec.id} ({rb.commit[:12]}): all llama-bench flags supported"
            )
        if ignored:
            console.print(f"  [dim]not applicable to llama-bench: {', '.join(sorted(ignored))}[/]")

    make_plan(exp)
    if problems:
        fail(f"{problems} problem(s) found", code=1)
    console.print("[green]valid[/]")
