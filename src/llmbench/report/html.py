"""Self-contained HTML report (Jinja2 + Plotly, embedded so it opens offline) and PNG charts."""

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

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


def _gap(low: float | None, high: float | None) -> float:
    """Whisker length; none when the CI is unknown (a single repetition)."""
    return high - low if low is not None and high is not None else 0.0


type Theme = Literal["light", "terminal"]

TERMINAL_COLORS = ["#4ade80", "#fbbf24", "#22d3ee", "#f472b6", "#a78bfa", "#fb923c", "#94a3b8"]


@dataclass(frozen=True)
class Chart:
    key: str
    title: str
    help: str
    fig: Any


def short_labels(labels: list[str], sep: str = " · ") -> dict[str, str]:
    """Drop the `sep`-separated parts every label shares (e.g. the build id when there is one)."""
    parts = [label.split(sep) for label in labels]
    if len(parts) < 2 or len({len(p) for p in parts}) != 1:  # noqa: PLR2004
        return {label: label for label in labels}
    keep = [i for i in range(len(parts[0])) if len({p[i] for p in parts}) > 1]
    return {
        label: sep.join(p[i] for i in keep) or label for label, p in zip(labels, parts, strict=True)
    }


def _bench_chart(go: Any, d: ReportData) -> Chart:
    subplots = _need("plotly.subplots")
    tests = list(dict.fromkeys(r["test"] for r in d.bench))
    short = short_labels(list(dict.fromkeys(r["config"] for r in d.bench)))
    # one panel per test: prompt processing is often 5-10x faster than generation
    fig = subplots.make_subplots(rows=1, cols=len(tests), subplot_titles=tests)
    for col, test in enumerate(tests, start=1):
        rows = [r for r in d.bench if r["test"] == test]
        fig.add_trace(
            go.Bar(
                name=test,
                x=[short[r["config"]] for r in rows],
                y=[r["tps"] for r in rows],
                customdata=[[r["config"], r["ci_low"], r["ci_high"], r["cv"]] for r in rows],
                hovertemplate="%{customdata[0]}<br>%{y:.2f} t/s<br>95% CI "
                "[%{customdata[1]:.2f}, %{customdata[2]:.2f}]<br>cv %{customdata[3]:.1%}"
                "<extra>" + test + "</extra>",
                error_y={"type": "data", "symmetric": False,
                         "array": [_gap(r["tps"], r["ci_high"]) for r in rows],
                         "arrayminus": [_gap(r["ci_low"], r["tps"]) for r in rows]},
                marker_color=_color(col - 1),
                showlegend=False,
            ),
            row=1, col=col,
        )  # fmt: skip
        fig.update_yaxes(title_text="tokens/s", row=1, col=col)
    fig.update_xaxes(tickangle=-20)
    return Chart(
        "bench",
        "llama-bench throughput",
        "Tokens per second measured by llama-bench, one panel per test. "
        "ppN = prompt processing: reading an N-token prompt in batches (what you wait for before "
        "the first word). tgN = text generation: writing N tokens one by one (the reading speed "
        "of the answer). Bar = mean of the repetitions; whisker = 95% bootstrap confidence "
        "interval of that mean (no whisker: a single repetition). Higher is better.",
        fig,
    )


def _memory_chart(go: Any, d: ReportData) -> Chart | None:
    seen: dict[str, float] = {}
    for r in d.bench:
        if r.get("rss_mb") is not None:
            seen[r["config"]] = max(seen.get(r["config"], 0.0), float(r["rss_mb"]))
    if not seen:
        return None
    short = short_labels(list(seen))
    fig = go.Figure(
        go.Bar(
            x=[short[c] for c in seen],
            y=list(seen.values()),
            customdata=list(seen),
            hovertemplate="%{customdata}<br>%{y:.0f} MB<extra></extra>",
            marker_color=_color(2),
        )
    )
    fig.update_layout(yaxis_title="peak RSS (MB)", xaxis_tickangle=-20)
    return Chart(
        "memory",
        "Peak memory",
        "Highest resident memory (VmHWM) of the llama-bench process for each configuration: "
        "model weights + KV cache + compute buffers. It is what must fit in RAM. Lower is better.",
        fig,
    )


def charts(d: ReportData, theme: Theme = "light") -> list[Chart]:
    """Every chart that has data, with a title and a plain-language explanation."""
    go = _need("plotly.graph_objects")
    out: list[Chart] = []
    if d.bench:
        out.append(_bench_chart(go, d))
        if memory := _memory_chart(go, d):
            out.append(memory)
    if d.latency:
        fig = go.Figure()
        short = short_labels(list(dict.fromkeys(r["config"] for r in d.latency)))
        for i, suite in enumerate(dict.fromkeys(r["suite"] for r in d.latency)):
            rows = [r for r in d.latency if r["suite"] == suite]
            p50 = [r.get("ttft_ms_p50") or 0 for r in rows]
            p95 = [r.get("ttft_ms_p95") or 0 for r in rows]
            fig.add_bar(
                name=f"{suite} TTFT p50",
                x=[short[r["config"]] for r in rows],
                y=p50,
                customdata=[[r["config"], b] for r, b in zip(rows, p95, strict=True)],
                hovertemplate="%{customdata[0]}<br>p50 %{y:.0f} ms<br>p95 %{customdata[1]:.0f} ms"
                "<extra>" + str(suite) + "</extra>",
                error_y={"type": "data", "symmetric": False,
                         "array": [b - a for a, b in zip(p50, p95, strict=True)],
                         "arrayminus": [0] * len(rows)},
                marker_color=_color(i),
            )  # fmt: skip
        fig.update_layout(barmode="group", yaxis_title="ms", xaxis_tickangle=-20)
        out.append(
            Chart(
                "latency",
                "Time to first token",
                "TTFT: milliseconds from sending a request to llama-server until the first token "
                "arrives (prompt processing + queueing). Bar = median (p50, a typical request); "
                "whisker up to p95 (1 request in 20 is slower than this). Lower is better.",
                fig,
            )
        )
    if d.serving:
        fig = go.Figure()
        series = dict.fromkeys(r["config"] + " · " + str(r["suite"]) for r in d.serving)
        for i, cfg in enumerate(series):
            rows = sorted(
                (r for r in d.serving if r["config"] + " · " + str(r["suite"]) == cfg),
                key=lambda r: r.get("c") or 0,
            )
            fig.add_scatter(
                name=cfg,
                mode="lines+markers",
                x=[r.get("c") for r in rows],
                y=[r.get("gen_tps_aggregate") for r in rows],
                line_color=_color(i),
            )
        fig.update_layout(xaxis_title="concurrent clients", yaxis_title="aggregate gen tokens/s")
        out.append(
            Chart(
                "serving",
                "Throughput vs concurrency",
                "Total tokens per second generated for all clients together, as more clients send "
                "requests at the same time. A flattening line means the hardware is saturated: "
                "more users only make each one wait longer.",
                fig,
            )
        )
    scored = [r for r in d.quality if r.get("score_mean") is not None]
    if scored:
        fig = go.Figure()
        for i, suite in enumerate(dict.fromkeys(r["suite"] for r in scored)):
            rows = [r for r in scored if r["suite"] == suite]
            fig.add_scatter(
                name=suite,
                mode="markers+text",
                x=[r.get("gen_tps_mean") for r in rows],
                y=[r.get("score_mean") for r in rows],
                text=[r["model"] for r in rows],
                textposition="top center",
                hovertext=[r["config"] for r in rows],
                marker={"size": 11, "color": _color(i)},
            )
        fig.update_layout(xaxis_title="generation tokens/s", yaxis_title="score (0-1)")
        out.append(
            Chart(
                "quality",
                "Quality vs speed",
                "Each point is a configuration: horizontal = generation speed, vertical = mean "
                "score of the deterministic evaluators (0-1). The best trade-offs sit at the top "
                "right.",
                fig,
            )
        )
    for c in out:
        _style(c.fig, theme)
    return out


def _color(i: int) -> str:
    return TERMINAL_COLORS[i % len(TERMINAL_COLORS)]


def _style(fig: Any, theme: Theme) -> None:
    fig.update_layout(height=420, margin={"t": 40, "l": 60, "r": 20, "b": 110})
    if theme == "light":
        fig.update_layout(template="plotly_white")
        return
    mono = 'ui-monospace, "JetBrains Mono", "Cascadia Code", Menlo, Consolas, monospace'
    grid = "#1e2a22"
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": mono, "color": "#b8c4bb", "size": 12},
        colorway=TERMINAL_COLORS,
        legend={"bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"font_family": mono, "bgcolor": "#0d1310", "bordercolor": "#4ade80"},
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor="#2b3a30")
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor="#2b3a30")
    fig.update_annotations(font={"family": mono, "color": "#4ade80"})


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
        (c.title, c.help, pio.to_html(c.fig, include_plotlyjs=False, full_html=False)) for c in figs
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
