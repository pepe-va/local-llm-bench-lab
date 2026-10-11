"""`/lab/suites`: create, edit, check and try request suites (`suites/*.yaml`).

The YAML text is the source of truth, as in the experiment editor: the form sends one
change at a time and the server applies it with a comment-preserving patch. Checks run on
the unsaved text through the same loader as `llb`; "try it" streams the answers of the
chat model (NDJSON, one event per line) and never runs beside a job.
"""

import json
import threading
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Annotated, Any

import jinja2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from llmbench import __version__
from llmbench.config.suite import Suite
from llmbench.lab import suites as logic
from llmbench.lab.chat import ChatServer
from llmbench.lab.editor import EditorError, parse, patch
from llmbench.lab.state import Lab

NO_STORE = {"Cache-Control": "no-store"}


class PatchReq(BaseModel):
    text: str
    path: list[str | int]
    value: Any = None
    """`null` removes the key (and mappings it leaves empty)."""


class NewReq(BaseModel):
    kind: str
    name: Annotated[str, Field(min_length=1, max_length=80)]


class SaveReq(BaseModel):
    name: str
    text: str
    overwrite: bool = False


class CheckReq(BaseModel):
    text: str
    fetch: bool = False
    """Download a Hugging Face dataset to preview its rows."""


class TryReq(BaseModel):
    text: str
    limit: Annotated[int, Field(ge=1, le=logic.TRY_MAX)] = 5
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.0
    max_tokens: Annotated[int, Field(ge=1, le=32768)] | None = None


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return Suite.model_json_schema(by_alias=True)


def register(app: FastAPI, lab: Lab, env: jinja2.Environment) -> None:
    """Add the suites tab's page and API to the lab app."""
    files = logic.Suites(lab.root / "suites", lab.experiments.folder)
    trying = threading.Lock()

    def call[T](fn: Callable[[], T]) -> T:
        try:
            return fn()
        except EditorError as e:
            raise HTTPException(400, {"message": str(e), "line": e.line}) from None

    @app.get("/lab/suites", response_class=HTMLResponse)
    def suites_tab() -> HTMLResponse:
        return HTMLResponse(
            env.get_template("suites.html.j2").render(version=__version__, tab="suites")
        )

    @app.get("/lab/api/suites/meta")
    def suites_meta() -> JSONResponse:
        return JSONResponse(
            {
                "schema": _schema(),
                "folder": str(files.folder),
                "files": files.files(),
                "used_by": files.used_by(),
                "kinds": [{"key": k.key, "title": k.title, "help": k.help} for k in logic.KINDS],
                "try_max": logic.TRY_MAX,
            },
            headers=NO_STORE,
        )

    @app.get("/lab/api/suites/file")
    def suites_file(name: str) -> JSONResponse:
        text = call(lambda: files.read(name))
        return JSONResponse({"name": name, "text": text}, headers=NO_STORE)

    @app.post("/lab/api/suites/patch")
    def suites_patch(req: PatchReq) -> JSONResponse:
        text = call(lambda: patch(req.text, req.path, logic.literal_blocks(req.value)))
        return JSONResponse({"text": text, "doc": parse(text)})

    @app.post("/lab/api/suites/new")
    def suites_new(req: NewReq) -> JSONResponse:
        text = call(lambda: logic.new_suite(req.kind, req.name))
        return JSONResponse({"text": text, "doc": parse(text)})

    @app.post("/lab/api/suites/save")
    def suites_save(req: SaveReq) -> JSONResponse:
        path = call(lambda: files.save(req.name, req.text, overwrite=req.overwrite))
        return JSONResponse({"name": req.name, "path": str(path), "files": files.files()})

    @app.post("/lab/api/suites/check")
    def suites_check(req: CheckReq) -> JSONResponse:
        return JSONResponse(
            call(lambda: logic.preview(files, req.text, fetch=req.fetch)), headers=NO_STORE
        )

    @app.post("/lab/api/suites/try")
    def suites_try(req: TryReq) -> StreamingResponse:
        chat = lab.chat
        if lab.jobs.busy:
            raise HTTPException(409, "a job is running: try the suite when it is done")
        if chat is None or chat.state != "ready":
            raise HTTPException(409, "load a model in the chat tab first")
        suite = call(lambda: files.load(req.text))
        opts = logic.TryOptions(req.limit, req.temperature, req.max_tokens)

        def alive() -> str | None:
            if lab.jobs.busy:
                return "a job started"
            if lab.chat is not chat or chat.state != "ready":
                return "the chat model was unloaded"
            return None

        def stream() -> Iterator[bytes]:
            # taken here, not before: a stream that never starts must not keep the lock
            if not trying.acquire(blocking=False):
                yield _line({"error": "another try is still running"})
                return
            try:
                header = {"name": chat.model.name, "quant": chat.model.quant}
                yield _line({"model": header})
                for event in logic.try_events(suite, _url(chat), opts, alive=alive):
                    yield _line(event)
            finally:
                trying.release()

        return StreamingResponse(
            stream(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


def _url(chat: ChatServer) -> str:
    """The chat's llama-server (on 127.0.0.1); `ChatServer` keeps it private."""
    server: Any = vars(chat)["_server"]
    return str(server.url)


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode()
