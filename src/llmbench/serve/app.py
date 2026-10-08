"""`llb serve`: read-only dashboard over results stores (FastAPI + HTMX, extra `serve`).

A thin layer on the same analysis/report code as `llb report` (ADR-0005). It never writes,
and is meant to run on another machine than the one being measured, or on the Pi between
runs. Live progress works by polling SQLite (WAL lets `llb run` write meanwhile).
"""

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import jinja2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from llmbench import __version__
from llmbench.analysis.view import build_view
from llmbench.config.experiment import Experiment
from llmbench.config.loader import ConfigError, load_experiment
from llmbench.paths import runs_dir
from llmbench.report.data import collect
from llmbench.report.html import charts
from llmbench.report.markdown import _fmt, _pct  # pyright: ignore[reportPrivateUsage]
from llmbench.results.store import Store


@dataclass(frozen=True)
class StoreRef:
    id: str
    path: Path

    def open(self) -> Store:
        return Store(self.path)


def discover(roots: list[Path]) -> dict[str, StoreRef]:
    """Every `benchmark.sqlite` under the given roots (a file, a run dir, or a parent dir)."""
    found: list[Path] = []
    for given in roots:
        root = given.expanduser()
        if root.is_file():
            found.append(root)
        elif (root / "benchmark.sqlite").is_file():
            found.append(root / "benchmark.sqlite")
        elif root.is_dir():
            found += sorted(root.glob("*/benchmark.sqlite"))
    refs: dict[str, StoreRef] = {}
    for p in found:
        sid = hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:10]
        refs[sid] = StoreRef(sid, p.resolve())
    return refs


def _env() -> jinja2.Environment:
    env = jinja2.Environment(
        loader=jinja2.PackageLoader("llmbench.serve", "templates"),
        autoescape=True,
    )

    def defined(v: Any) -> Any:
        return None if isinstance(v, jinja2.Undefined) else v

    def fmt(v: Any, digits: int = 1) -> str:
        return _fmt(defined(v), digits)

    def pct(v: Any) -> str:
        return _pct(defined(v))

    env.filters["fmt"] = fmt
    env.filters["pct"] = pct
    env.filters["tojson_pretty"] = lambda v: json.dumps(v, indent=2, default=str)  # pyright: ignore[reportUnknownLambdaType]
    return env


def create_app(  # noqa: PLR0915 (route table: one function per endpoint)
    roots: list[Path] | None = None, experiments: list[Path] | None = None
) -> FastAPI:
    """`roots`: where to look for results (default `$LLB_HOME/runs`). `experiments`: YAMLs
    whose gates and selection should be shown for their results."""
    app = FastAPI(title="llb dashboard", version=__version__, docs_url="/api/docs")
    env = _env()
    search = roots or [runs_dir()]
    gates: dict[str, Experiment] = {}
    for path in experiments or []:
        try:
            exp = load_experiment(path)
        except ConfigError:
            continue
        gates[exp.experiment.name] = exp

    def stores() -> dict[str, StoreRef]:
        return discover(search)  # rescanned on every request: new experiments just appear

    def get_store(sid: str) -> StoreRef:
        ref = stores().get(sid)
        if ref is None:
            raise HTTPException(404, "unknown results store")
        return ref

    def page(name: str, **ctx: Any) -> HTMLResponse:
        return HTMLResponse(env.get_template(name).render(version=__version__, **ctx))

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return page("index.html.j2", experiments=_experiments(stores()), roots=search)

    @app.get("/static/htmx.min.js")
    def htmx() -> Response:
        js = (resources.files("llmbench.serve") / "static" / "htmx.min.js").read_bytes()
        return Response(
            js, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"}
        )

    @app.get("/static/plotly.min.js")
    def plotly() -> Response:
        from plotly.offline import get_plotlyjs  # noqa: PLC0415 (heavy import, only when needed)

        return Response(
            get_plotlyjs(), media_type="text/javascript", headers={"Cache-Control": "max-age=86400"}
        )

    @app.get("/e/{sid}/{experiment}", response_class=HTMLResponse)
    def experiment_page(sid: str, experiment: str) -> HTMLResponse:
        ref = get_store(sid)
        exp = gates.get(experiment)
        store = ref.open()
        try:
            d = collect(store, experiment, exp.gates if exp else None, exp.select if exp else None)
        finally:
            store.close()
        if not d.rows:
            raise HTTPException(404, "no runs for this experiment")
        import plotly.io as pio  # noqa: PLC0415

        to_html: Any = pio.to_html  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        blocks = [
            (t, str(to_html(f, include_plotlyjs=False, full_html=False))) for t, f in charts(d)
        ]
        return page("experiment.html.j2", d=d, sid=sid, charts=blocks, store=ref)

    @app.get("/e/{sid}/{experiment}/runs", response_class=HTMLResponse)
    def runs_partial(sid: str, experiment: str) -> HTMLResponse:
        """HTMX fragment: the latest runs, polled for live progress."""
        return page("runs.html.j2", runs=_runs(get_store(sid), experiment, 30), sid=sid)

    @app.get("/run/{sid}/{run_id}", response_class=HTMLResponse)
    def run_page(sid: str, run_id: int) -> HTMLResponse:
        ref = get_store(sid)
        store = ref.open()
        try:
            run = _run(store, run_id)
            requests = store.requests_of(run_id)
            samples = store.telemetry(run_id)
        finally:
            store.close()
        return page("run.html.j2", run=run, requests=requests, samples=samples, sid=sid)

    # -- JSON API ---------------------------------------------------------------------

    @app.get("/api/experiments")
    def api_experiments() -> JSONResponse:
        return JSONResponse(_experiments(stores()))

    @app.get("/api/e/{sid}/{experiment}/configs")
    def api_configs(sid: str, experiment: str) -> JSONResponse:
        store = get_store(sid).open()
        try:
            rows = build_view(store, experiment)
        finally:
            store.close()
        return JSONResponse(
            [
                {
                    "build": r.build_id,
                    "model": r.model_id,
                    "variant": r.variant,
                    "metrics": {k: r.metric(k) for k in sorted(r.values)},
                    "failed_runs": r.failed_runs,
                }
                for r in rows
            ]
        )

    @app.get("/api/e/{sid}/{experiment}/runs")
    def api_runs(sid: str, experiment: str, limit: int = 100) -> JSONResponse:
        return JSONResponse(_runs(get_store(sid), experiment, limit))

    @app.get("/api/run/{sid}/{run_id}")
    def api_run(sid: str, run_id: int) -> JSONResponse:
        store = get_store(sid).open()
        try:
            return JSONResponse(_run(store, run_id) | {"requests": store.requests_of(run_id)})
        finally:
            store.close()

    return app


def _experiments(refs: dict[str, StoreRef]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ref in refs.values():
        store = ref.open()
        try:
            for row in store.db.execute(
                """SELECT experiment, COUNT(*) AS runs,
                          SUM(status != 'completed') AS failed,
                          MAX(finished_at) AS last,
                          GROUP_CONCAT(DISTINCT phase) AS phases
                   FROM runs GROUP BY experiment ORDER BY last DESC"""
            ):
                out.append(dict(row) | {"sid": ref.id, "path": str(ref.path)})
        finally:
            store.close()
    return sorted(out, key=lambda e: e["last"] or "", reverse=True)


def _runs(ref: StoreRef, experiment: str, limit: int) -> list[dict[str, Any]]:
    store = ref.open()
    try:
        rows = store.db.execute(
            """SELECT id, phase, status, model_id, build_id, suite, concurrency, variant,
                      started_at, duration_s, error
               FROM runs WHERE experiment = ? ORDER BY id DESC LIMIT ?""",
            (experiment, limit),
        ).fetchall()
        result = [dict(r) | {"variant": json.loads(r["variant"])} for r in rows]
        for r in result:
            r["metrics"] = dict(
                store.db.execute("SELECT name, value FROM run_metrics WHERE run_id = ?", (r["id"],))
            )
            r["measurements"] = [
                dict(m)
                for m in store.db.execute(
                    "SELECT label, avg_ts FROM measurements WHERE run_id = ?", (r["id"],)
                )
            ]
        return result
    finally:
        store.close()


def _run(store: Store, run_id: int) -> dict[str, Any]:
    row = store.db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "unknown run")
    run = dict(row)
    for key in ("build", "model", "variant", "config", "argv", "ignored", "env", "telemetry",
                "cooldown", "suite_provenance"):  # fmt: skip
        if run.get(key):
            run[key] = json.loads(run[key])
    run["metrics"] = dict(
        store.db.execute("SELECT name, value FROM run_metrics WHERE run_id = ?", (run_id,))
    )
    run["measurements"] = [
        dict(m) for m in store.db.execute("SELECT * FROM measurements WHERE run_id = ?", (run_id,))
    ]
    run["judge"] = [
        dict(j) for j in store.db.execute("SELECT * FROM judge_scores WHERE run_id = ?", (run_id,))
    ]
    return run
