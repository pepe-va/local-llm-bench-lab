"""`/lab/system`: the environment, llama.cpp builds and their flags; and the models tab's
extras (inspect a GGUF, prune the Hugging Face cache, pull by reference)."""

from collections.abc import Callable
from typing import Annotated, Any

import jinja2
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from llmbench import __version__
from llmbench.lab import system
from llmbench.lab.jobs import LabBusyError
from llmbench.lab.state import Lab, LabError


class BuildReq(BaseModel):
    ref: Annotated[str, Field(max_length=100)]
    backend: str = "cpu"
    cmake: Annotated[list[str], Field(max_length=50)] = []
    jobs: int | None = None
    force: bool = False


class PruneReq(BaseModel):
    older_than_days: Annotated[float, Field(ge=0, le=100_000)] | None = None
    repo: Annotated[str, Field(max_length=200)] | None = None
    dry_run: bool = True
    expect: list[str] | None = None
    """The revisions (`repo@commit`) the user saw in the preview and confirmed."""


class PullReq(BaseModel):
    ref: Annotated[str, Field(min_length=3, max_length=500)]
    favorite: bool = False


def _call[T](fn: Callable[[], T]) -> T:
    try:
        return fn()
    except LabBusyError as e:
        raise HTTPException(409, str(e)) from None
    except LabError as e:
        raise HTTPException(400, str(e)) from None


def register(app: FastAPI, lab: Lab, env: jinja2.Environment) -> None:
    """Add the system tab's page and API (and the models tab's extra API) to the lab app."""

    @app.get("/lab/system", response_class=HTMLResponse)
    def system_tab() -> HTMLResponse:
        return HTMLResponse(
            env.get_template("system.html.j2").render(version=__version__, tab="system")
        )

    @app.get("/lab/api/system")
    def environment() -> JSONResponse:
        return JSONResponse(system.environment(lab))

    @app.get("/lab/api/system/builds")
    def build_list() -> JSONResponse:
        return JSONResponse(
            {"builds": system.build_list(lab), "options": system.build_options(lab)}
        )

    @app.post("/lab/api/system/build")
    def build(req: BuildReq) -> JSONResponse:
        def go() -> dict[str, Any]:
            return system.start_build(
                lab, req.ref, backend=req.backend, cmake=req.cmake, jobs=req.jobs,
                force=req.force,
            )  # fmt: skip

        return JSONResponse(_call(go))

    @app.get("/lab/api/system/flags")
    def flags(
        build: Annotated[str, Query(max_length=200)],
        tool: str = "llama-bench",
        raw: bool = False,
    ) -> JSONResponse:
        return JSONResponse(_call(lambda: system.flags(lab, build, tool, raw=raw)))

    @app.get("/lab/api/models/{key}/inspect")
    def inspect(key: str) -> JSONResponse:
        return JSONResponse(_call(lambda: system.inspect(lab, key)))

    @app.post("/lab/api/models/prune")
    def prune(req: PruneReq) -> JSONResponse:
        def go() -> dict[str, Any]:
            return system.prune(
                lab, older_than_days=req.older_than_days, repo=req.repo,
                dry_run=req.dry_run, expect=req.expect,
            )  # fmt: skip

        return JSONResponse(_call(go))

    @app.post("/lab/api/models/pull")
    def pull(req: PullReq) -> JSONResponse:
        d = _call(lambda: system.pull(lab, req.ref, favorite=req.favorite))
        return JSONResponse(lab.download_progress(d))
