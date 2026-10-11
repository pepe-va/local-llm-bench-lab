"""`llb lab`: the dashboard plus tabs that act: models, chat and test (an experiment editor).

Every action goes through `Lab` (state.py). Requests that change something must carry the
`X-LLB-Lab` header: a browser only sends custom headers cross-site after a CORS preflight,
which this app never grants, so another website cannot trigger downloads or runs (CSRF).
"""

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Literal

import jinja2
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from llmbench import __version__
from llmbench.lab import routes_suites, routes_system
from llmbench.lab.chat import ChatSettings
from llmbench.lab.editor import EditorError, parse, patch
from llmbench.lab.jobs import LabBusyError
from llmbench.lab.state import Lab, LabError
from llmbench.paths import runs_dir
from llmbench.serve.app import create_app, discover

ACTION_HEADER = "X-LLB-Lab"
STATIC = {
    "common.js", "form.js", "models.js", "chat.js", "editor.js", "suites.js", "system.js",
}  # fmt: skip
NO_STORE = {"Cache-Control": "no-store"}


class ChatStart(BaseModel):
    model: str
    build: str | None = None
    ctx_size: Annotated[int, Field(ge=256, le=1_048_576)] = 4096
    threads: Annotated[int, Field(ge=1, le=512)] | None = None
    reasoning: Literal["auto", "on", "off"] = "auto"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatTurn(BaseModel):
    messages: list[Message]
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.7
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = 1024


class DownloadReq(BaseModel):
    repo: str
    file: str


class Text(BaseModel):
    text: str


class PatchReq(BaseModel):
    text: str
    path: list[str | int]
    value: Any = None
    """`null` removes the key (and mappings it leaves empty)."""


class NewReq(BaseModel):
    models: list[str] = []
    profile: str | None = None
    presets: list[str] = []


class SaveReq(BaseModel):
    name: str
    text: str
    overwrite: bool = False


class CheckReq(BaseModel):
    text: str
    command: Literal["validate", "plan"]
    models: list[str] = []


class CommandReq(BaseModel):
    name: str
    command: Literal["run", "guide", "judge", "report"]
    models: list[str] = []
    resume: bool = False
    no_build: bool = False
    force: bool = False
    formats: list[Literal["html", "md", "png"]] = []


def create_lab_app(  # noqa: PLR0915 (route table: one function per endpoint)
    lab: Lab, roots: list[Path] | None = None, experiments: list[Path] | None = None
) -> FastAPI:
    app = create_app(roots, experiments, lab=True)
    env: jinja2.Environment = app.state.env
    search = roots or [runs_dir()]

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
        yield
        lab.close()

    app.router.lifespan_context = lifespan

    @app.middleware("http")
    async def require_action_header(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method not in ("GET", "HEAD") and ACTION_HEADER not in request.headers:
            return JSONResponse({"detail": f"missing {ACTION_HEADER} header"}, status_code=403)
        return await call_next(request)

    def call(fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except LabBusyError as e:
            raise HTTPException(409, str(e)) from None
        except LabError as e:
            raise HTTPException(400, str(e)) from None
        except EditorError as e:
            raise HTTPException(400, {"message": str(e), "line": e.line}) from None

    @app.get("/lab")
    def lab_home() -> RedirectResponse:
        return RedirectResponse("/lab/models")

    def tab(name: str) -> HTMLResponse:
        return HTMLResponse(
            env.get_template(f"{name}.html.j2").render(version=__version__, tab=name)
        )

    @app.get("/lab/models", response_class=HTMLResponse)
    def models_tab() -> HTMLResponse:
        return tab("models")

    @app.get("/lab/chat", response_class=HTMLResponse)
    def chat_tab() -> HTMLResponse:
        return tab("chat")

    @app.get("/lab/test", response_class=HTMLResponse)
    def test_tab() -> HTMLResponse:
        return tab("test")

    @app.get("/lab/static/{name}")
    def lab_static(name: str) -> Response:
        if name not in STATIC:
            raise HTTPException(404)
        js = (resources.files("llmbench.lab") / "static" / name).read_bytes()
        return Response(js, media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    @app.get("/lab/api/status")
    def status() -> JSONResponse:
        data = lab.status()
        job = data["job"]
        if job and job["experiment"]:
            job["results"] = _results_url(search, job["experiment"])
        return JSONResponse(data)

    @app.get("/lab/api/search")
    def search_models(q: Annotated[str, Query(min_length=2)]) -> JSONResponse:
        return JSONResponse(call(lambda: lab.search(q)))

    @app.get("/lab/api/repo")
    def repo(id: Annotated[str, Query(min_length=3)]) -> JSONResponse:
        return JSONResponse(call(lambda: lab.repo(id)))

    @app.post("/lab/api/downloads")
    def download(req: DownloadReq) -> JSONResponse:
        d = call(lambda: lab.download(req.repo, req.file))
        return JSONResponse(lab.download_progress(d))

    @app.get("/lab/api/models")
    def models() -> JSONResponse:
        return JSONResponse({"models": lab.models(), "ram_total": lab.ram_total()})

    @app.post("/lab/api/models/{key}/favorite")
    def favorite(key: str) -> JSONResponse:
        call(lambda: lab.favorite(key))
        return JSONResponse({"ok": True})

    @app.delete("/lab/api/models/{key}")
    def delete(key: str) -> JSONResponse:
        call(lambda: lab.delete(key))
        return JSONResponse({"ok": True})

    @app.get("/lab/api/builds")
    def builds() -> JSONResponse:
        return JSONResponse(lab.builds())

    @app.post("/lab/api/chat/start")
    def chat_start(req: ChatStart) -> JSONResponse:
        settings = ChatSettings(req.ctx_size, req.threads, req.reasoning)
        chat = call(lambda: lab.start_chat(req.model, settings, req.build))
        return JSONResponse(chat.to_json())

    @app.post("/lab/api/chat/stop")
    def chat_stop() -> JSONResponse:
        lab.stop_chat()
        return JSONResponse({"ok": True})

    @app.post("/lab/api/chat")
    def chat(turn: ChatTurn) -> StreamingResponse:
        if lab.chat is None:
            raise HTTPException(409, "start the chat first")
        body = {
            "messages": [m.model_dump() for m in turn.messages],
            "temperature": turn.temperature,
            "max_tokens": turn.max_tokens,
        }
        return StreamingResponse(
            lab.chat.stream(body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- experiment editor -------------------------------------------------------------

    @app.get("/lab/api/editor/meta")
    def editor_meta() -> JSONResponse:
        return JSONResponse(lab.editor_meta())

    @app.get("/lab/api/editor/file")
    def editor_file(name: str) -> JSONResponse:
        text = call(lambda: lab.experiments.read(name))
        return JSONResponse({"name": name, "text": text})

    @app.post("/lab/api/editor/parse")
    def editor_parse(req: Text) -> JSONResponse:
        return JSONResponse({"doc": call(lambda: parse(req.text))}, headers=NO_STORE)

    @app.post("/lab/api/editor/patch")
    def editor_patch(req: PatchReq) -> JSONResponse:
        text = call(lambda: patch(req.text, req.path, req.value))
        return JSONResponse({"text": text, "doc": parse(text)})

    @app.post("/lab/api/editor/new")
    def editor_new(req: NewReq) -> JSONResponse:
        text = call(lambda: lab.new_experiment(req.models, req.profile, req.presets))
        return JSONResponse({"text": text, "doc": parse(text)})

    @app.post("/lab/api/editor/save")
    def editor_save(req: SaveReq) -> JSONResponse:
        path = call(lambda: lab.experiments.save(req.name, req.text, overwrite=req.overwrite))
        return JSONResponse({"name": req.name, "path": str(path), "files": lab.experiments.files()})

    @app.post("/lab/api/editor/check")
    def editor_check(req: CheckReq) -> JSONResponse:
        return JSONResponse(call(lambda: lab.experiments.check(req.text, req.command, req.models)))

    @app.post("/lab/api/editor/command")
    def editor_command(req: CommandReq) -> JSONResponse:
        def go() -> dict[str, Any]:
            return lab.command(
                req.name,
                req.command,
                models=req.models,
                resume=req.resume,
                no_build=req.no_build,
                force=req.force,
                formats=list(req.formats),
            )

        return JSONResponse(call(go))

    @app.post("/lab/api/job/cancel")
    def cancel() -> JSONResponse:
        lab.cancel_job()
        return JSONResponse({"ok": True})

    routes_suites.register(app, lab, env)
    routes_system.register(app, lab, env)
    return app


def _results_url(roots: list[Path], experiment: str) -> str | None:
    """The dashboard page of a lab test, once its results store exists."""
    for ref in discover(roots).values():
        if ref.path.parent.name == experiment:
            return f"/e/{ref.id}/{experiment}"
    return None
