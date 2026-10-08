"""Self-contained HTML report (Jinja2 + Plotly, embedded so it opens offline) and PNG charts."""

from importlib import resources
from pathlib import Path
from typing import Any

from llmbench.report.data import ReportData
from llmbench.report.markdown import _fmt, _pct  # pyright: ignore[reportPrivateUsage]


class MissingExtraError(Exception):
    pass


def _need(module: str) -> Any:
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingExtraError(
            f"{module} is needed for this report format: `uv sync --extra report`"
        ) from None


def charts(d: ReportData) -> list[tuple[str, Any]]:
    """(title, plotly figure) pairs; empty sections are skipped."""
    go = _need("plotly.graph_objects")
    figs: list[tuple[str, Any]] = []
    if d.bench:
        fig = go.Figure()
        for test in dict.fromkeys(r["test"] for r in d.bench):
            rows = [r for r in d.bench if r["test"] == test]
            fig.add_bar(
                name=test,
                x=[r["config"] for r in rows],
                y=[r["tps"] for r in rows],
                error_y={"type": "data", "array": [r["stddev"] for r in rows]},
            )
        fig.update_layout(barmode="group", yaxis_title="tokens/s", xaxis_tickangle=-20)
        figs.append(("llama-bench throughput", fig))
    if d.latency:
        fig = go.Figure()
        for suite in dict.fromkeys(r["suite"] for r in d.latency):
            rows = [r for r in d.latency if r["suite"] == suite]
            p50 = [r.get("ttft_ms_p50") or 0 for r in rows]
            p95 = [r.get("ttft_ms_p95") or 0 for r in rows]
            fig.add_bar(
                name=f"{suite} TTFT p50",
                x=[r["config"] for r in rows],
                y=p50,
                error_y={"type": "data", "symmetric": False,
                         "array": [b - a for a, b in zip(p50, p95, strict=True)],
                         "arrayminus": [0] * len(rows)},
            )  # fmt: skip
        fig.update_layout(barmode="group", yaxis_title="ms (bar p50, whisker p95)")
        figs.append(("Time to first token", fig))
    if d.serving:
        fig = go.Figure()
        for cfg in dict.fromkeys(r["config"] + " · " + str(r["suite"]) for r in d.serving):
            rows = sorted(
                (r for r in d.serving if r["config"] + " · " + str(r["suite"]) == cfg),
                key=lambda r: r.get("c") or 0,
            )
            fig.add_scatter(
                name=cfg,
                mode="lines+markers",
                x=[r.get("c") for r in rows],
                y=[r.get("gen_tps_aggregate") for r in rows],
            )
        fig.update_layout(xaxis_title="concurrent clients", yaxis_title="aggregate gen tokens/s")
        figs.append(("Throughput vs concurrency", fig))
    scored = [r for r in d.quality if r.get("score_mean") is not None]
    if scored:
        fig = go.Figure()
        for suite in dict.fromkeys(r["suite"] for r in scored):
            rows = [r for r in scored if r["suite"] == suite]
            fig.add_scatter(
                name=suite,
                mode="markers+text",
                x=[r.get("gen_tps_mean") for r in rows],
                y=[r.get("score_mean") for r in rows],
                text=[r["model"] for r in rows],
                textposition="top center",
                hovertext=[r["config"] for r in rows],
            )
        fig.update_layout(xaxis_title="generation tokens/s", yaxis_title="score (0-1)")
        figs.append(("Quality vs speed", fig))
    for _, fig in figs:
        fig.update_layout(
            template="plotly_white", height=420, margin={"t": 30, "l": 60, "r": 20, "b": 120}
        )
    return figs


def render_html(d: ReportData) -> str:
    jinja2 = _need("jinja2")
    pio = _need("plotly.io")
    template_text = (resources.files("llmbench.report") / "templates" / "report.html.j2").read_text(
        encoding="utf-8"
    )
    env = jinja2.Environment(autoescape=True)
    undefined = jinja2.Undefined

    def defined(v: Any) -> Any:
        return None if isinstance(v, undefined) else v  # metrics a run did not record

    def fmt(v: Any, digits: int = 1) -> str:
        return _fmt(defined(v), digits)

    def pct(v: Any) -> str:
        return _pct(defined(v))

    env.filters["fmt"] = fmt
    env.filters["pct"] = pct
    figs = charts(d)
    blocks = [
        (title, pio.to_html(fig, include_plotlyjs=False, full_html=False)) for title, fig in figs
    ]
    # plotly.js is embedded once, so the report opens offline (e.g. copied off the Pi)
    library = _need("plotly.offline").get_plotlyjs() if figs else ""
    return env.from_string(template_text).render(d=d, charts=blocks, plotly_library=library)


def render_png(d: ReportData, out_dir: Path) -> list[Path]:
    """One PNG per chart, drawn with matplotlib (no browser needed)."""
    mpl = _need("matplotlib")
    mpl.use("Agg")
    plt = _need("matplotlib.pyplot")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    if d.bench:
        tests = list(dict.fromkeys(r["test"] for r in d.bench))
        configs = list(dict.fromkeys(r["config"] for r in d.bench))
        fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.6), 4.5))
        width = 0.8 / len(tests)
        for i, test in enumerate(tests):
            vals = {r["config"]: r["tps"] for r in d.bench if r["test"] == test}
            ax.bar([j + i * width for j in range(len(configs))],
                   [vals.get(c, 0) for c in configs], width, label=test)  # fmt: skip
        ax.set_xticks([j + width * (len(tests) - 1) / 2 for j in range(len(configs))])
        ax.set_xticklabels([c.split(" · ")[-1] for c in configs], rotation=20, ha="right")
        ax.set_ylabel("tokens/s")
        ax.set_title(f"{d.experiment}: llama-bench")
        ax.legend()
        save(fig, "bench.png")
    if d.latency:
        fig, ax = plt.subplots(figsize=(max(6, len(d.latency) * 1.2), 4.5))
        labels = [f"{r['variant']}\n{r['suite']}" for r in d.latency]
        ax.bar(labels, [r.get("ttft_ms_p50") or 0 for r in d.latency], label="TTFT p50")
        ax.scatter(labels, [r.get("ttft_ms_p95") or 0 for r in d.latency], color="black",
                   marker="_", s=400, label="TTFT p95", zorder=3)  # fmt: skip
        ax.set_ylabel("ms")
        ax.set_title(f"{d.experiment}: time to first token")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend()
        save(fig, "latency.png")
    scored = [r for r in d.quality if r.get("score_mean") is not None]
    if scored:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for r in scored:
            ax.scatter(r.get("gen_tps_mean") or 0, r["score_mean"])
            point = (r.get("gen_tps_mean") or 0, r["score_mean"])
            label = f"{r['model']} ({r['suite']})"
            ax.annotate(label, point, fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("generation tokens/s")
        ax.set_ylabel("score (0-1)")
        ax.set_title(f"{d.experiment}: quality vs speed")
        save(fig, "quality.png")
    return written
