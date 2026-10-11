"""`llb lab` system tab (environment, llama.cpp builds and flags) and the models tab's extras
(inspect, prune, pull by reference)."""

import json
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from llmbench.hub.store import ModelStore
from llmbench.lab.app import ACTION_HEADER, create_lab_app
from llmbench.lab.state import Download, Lab
from llmbench.lab.system import build_args
from llmbench.paths import builds_dir
from tests.helpers import FakeHub, write_gguf

ACT = {ACTION_HEADER: "1"}
QWEN = "unsloth/Qwen3-1.7B-GGUF"


def wait(cond: Callable[[], bool], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


@pytest.fixture
def lab(tmp_path: Path, hub: FakeHub) -> Iterator[Lab]:
    root = tmp_path / "repo"
    (root / "experiments").mkdir(parents=True)
    the_lab = Lab(root=root, store=ModelStore(client=hub))
    yield the_lab
    the_lab.close()


@pytest.fixture
def client(lab: Lab) -> TestClient:
    return TestClient(create_lab_app(lab))


@pytest.fixture
def managed_build(fake_bin_dir: Path) -> Path:
    """A managed build directory as `llb llama build` leaves it, with the fake binaries."""
    out = builds_dir() / "c811cb8aaaaa-cpu"
    bin_dir = out / "cmake-build" / "bin"
    bin_dir.mkdir(parents=True)
    for tool in ("llama-bench", "llama-server"):
        shutil.copy2(fake_bin_dir / tool, bin_dir / tool)
    manifest = {
        "ref": "b6500",
        "commit": "c811cb8" + "a" * 33,
        "backend": "cpu",
        "cmake": {"CMAKE_BUILD_TYPE": "Release", "GGML_NATIVE": "ON", "GGML_LTO": "ON"},
        "version": "6500",
        "built_at": "2026-10-01T10:00:00+00:00",
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return bin_dir


def model_key(client: TestClient, **match: Any) -> str:
    models: list[dict[str, Any]] = client.get("/lab/api/models").json()["models"]
    [m] = [m for m in models if all(m[k] == v for k, v in match.items())]
    return m["key"]


# -- environment and builds -------------------------------------------------------------


def test_system_page_and_environment(client: TestClient, tmp_path: Path) -> None:
    page = client.get("/lab/system")
    assert page.status_code == 200
    assert "/lab/static/system.js" in page.text
    assert client.get("/lab/static/system.js").status_code == 200

    (tmp_path / "models").mkdir()
    write_gguf(tmp_path / "models" / "loose-Q4_0.gguf")
    env = client.get("/lab/api/system").json()
    assert all(env[k] for k in ("llb", "python", "machine", "system"))
    locations = {loc["label"]: loc for loc in env["locations"]}
    assert locations["LLB_HOME"]["path"] == str(tmp_path / "llbhome")
    assert locations["models/ (favorites)"]["used"] > 0
    assert locations["runs"]["used"] == 0
    assert all(loc["free"] is None or loc["free"] > 0 for loc in env["locations"])
    extras = {x["name"]: x["installed"] for x in env["extras"]}
    assert extras["serve"] is True  # the lab itself runs on it
    assert set(extras) == {"report", "serve", "hf-datasets", "lm-eval"}


def test_builds_list_and_flags(client: TestClient, managed_build: Path) -> None:
    data = client.get("/lab/api/system/builds").json()
    [b] = data["builds"]
    assert (b["id"], b["ref"], b["version"], b["backend"]) == (
        "c811cb8aaaaa-cpu", "b6500", "6500", "cpu",
    )  # fmt: skip
    assert b["bin_dir"] == str(managed_build)
    assert b["tools"] == {"llama-bench": True, "llama-server": True, "llama-cli": False}
    assert b["built_at"].startswith("2026-10-01")
    assert "cpu" in data["options"]["backends"]
    assert data["options"]["default_ref"] == "master"  # no profile in this repo

    r = client.get("/lab/api/system/flags", params={"build": b["id"], "tool": "llama-bench"})
    assert r.status_code == 200
    flags = r.json()
    assert {"-m", "--model", "-r", "--repetitions"} <= set(flags["flags"])
    assert "help" not in flags
    raw = client.get(
        "/lab/api/system/flags", params={"build": b["id"], "tool": "llama-server", "raw": True}
    ).json()
    assert "--ctx-size" in raw["flags"]
    assert "usage" in raw["help"]

    # only builds the lab knows of, and only the tools llb drives
    unknown = client.get("/lab/api/system/flags", params={"build": "../../bin", "tool": "ls"})
    assert unknown.status_code == 400
    tool = client.get("/lab/api/system/flags", params={"build": b["id"], "tool": "sh"})
    assert tool.status_code == 400


def test_build_request_becomes_llb_llama_build(
    lab: Lab, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[tuple[str, str, list[str], bool]] = []

    def fake_start(
        kind: str, title: str, args: list[str], *, heavy: bool, experiment: str | None = None
    ) -> dict[str, Any]:
        started.append((kind, title, args, heavy))
        return {"job": {"title": title}, "stopped_chat": False}

    monkeypatch.setattr(lab, "start_job", fake_start)
    body = {
        "ref": "b6500",
        "backend": "vulkan",
        "cmake": ["GGML_LTO=ON", " CMAKE_CXX_FLAGS=-O3 -march=native "],
        "jobs": 4,
        "force": True,
    }
    r = client.post("/lab/api/system/build", json=body, headers=ACT)
    assert r.status_code == 200, r.text
    [(kind, title, args, heavy)] = started
    assert (kind, heavy) == ("build", True)
    assert "b6500" in title
    assert args == [
        "llama", "build", "--ref", "b6500", "--backend", "vulkan",
        "--cmake", "GGML_LTO=ON", "--cmake", "CMAKE_CXX_FLAGS=-O3 -march=native",
        "--jobs", "4", "--force",
    ]  # fmt: skip
    minimal = build_args("master", backend="cpu", cmake=[], jobs=None, force=False)
    assert minimal == ["llama", "build", "--ref", "master", "--backend", "cpu"]

    bad: list[dict[str, Any]] = [
        {"ref": ""},
        {"ref": "-x"},
        {"ref": "b1;rm -rf /"},
        {"ref": "a..b"},
        {"ref": "b1", "backend": "rocm"},
        {"ref": "b1", "cmake": ["NOEQUALS"]},
        {"ref": "b1", "cmake": ["1BAD=x"]},
        {"ref": "b1", "cmake": ["K=a\nb"]},
        {"ref": "b1", "jobs": 0},
    ]
    for b in bad:
        assert client.post("/lab/api/system/build", json=b, headers=ACT).status_code == 400, b
    assert len(started) == 1
    # every action needs the lab header
    assert client.post("/lab/api/system/build", json=body).status_code == 403


# -- models: inspect -------------------------------------------------------------------


def test_inspect_summarises_big_metadata(client: TestClient, tmp_path: Path) -> None:
    tokens = [f"tok{i}" for i in range(5000)]
    template = "{% for m in messages %}" + "x" * 10_000
    write_gguf(
        tmp_path / "models" / "tiny-Q4_K_M.gguf",
        {
            "general.name": "tiny",
            "general.file_type": 15,
            "tokenizer.ggml.tokens": tokens,
            "general.tags": ["a", "b"],
            "tokenizer.chat_template": template,
        },
    )
    key = model_key(client, where="local")
    r = client.get(f"/lab/api/models/{key}/inspect")
    assert r.status_code == 200
    data = r.json()
    assert len(r.content) < 30_000
    assert data["summary"]["architecture"] == "llama"
    assert data["summary"]["name"] == "tiny"
    assert data["summary"]["file_type"] == "Q4_K_M"
    assert data["tensors"] == 2
    shares = {t["type"]: t["share"] for t in data["tensor_types"]}
    assert sum(shares.values()) == pytest.approx(1)
    meta = {e["key"]: e for e in data["metadata"]}
    assert meta["tokenizer.ggml.tokens"] == {
        "key": "tokenizer.ggml.tokens", "type": "array", "value": "[5,000 items]",
    }  # fmt: skip
    assert meta["general.tags"] == {"key": "general.tags", "type": "array", "value": "['a', 'b']"}
    assert meta["tokenizer.chat_template"]["length"] == len(template)
    assert len(meta["tokenizer.chat_template"]["value"]) < 5000
    assert meta["llama.context_length"]["value"] == "2048"

    assert client.get("/lab/api/models/nope/inspect").status_code == 400


# -- models: prune ---------------------------------------------------------------------


def test_prune_previews_then_deletes_through_the_revision_api(
    lab: Lab, client: TestClient, hub: FakeHub
) -> None:
    commit = hub.repos[QWEN].commit
    hub.download(QWEN, "Qwen3-1.7B-Q4_K_M.gguf", commit)
    hub.download(
        "ggml-org/models", "phi-2/ggml-model-q4_0.gguf", hub.repos["ggml-org/models"].commit
    )
    fav = write_gguf(Path(lab.store.favorites.root) / "kept-Q4_0.gguf")

    def prune(**body: Any) -> Any:
        return client.post("/lab/api/models/prune", json=body, headers=ACT)

    assert prune(older_than_days=30).json()["files"] == []  # used just now
    hub.age(40)
    preview = prune(older_than_days=30, repo=QWEN).json()
    assert [f["name"] for f in preview["files"]] == ["Qwen3-1.7B-Q4_K_M.gguf"]
    assert preview["revisions"] == [f"{QWEN}@{commit}"]
    assert preview["files"][0]["idle_days"] >= 40
    assert preview["size"] > 0
    assert len(hub.cached_files()) == 2  # a preview deletes nothing

    # the cache changed since the preview the user confirmed
    stale = prune(older_than_days=30, dry_run=False, expect=preview["revisions"])
    assert stale.status_code == 409
    # not while a download writes to the cache
    lab.downloads["x"] = Download("x", "org/m", "m.gguf", "c" * 40, [("m.gguf", 1)], time.time())
    busy = prune(older_than_days=30, repo=QWEN, dry_run=False, expect=preview["revisions"])
    assert busy.status_code == 409
    lab.downloads.clear()

    done = prune(older_than_days=30, repo=QWEN, dry_run=False, expect=preview["revisions"])
    assert done.status_code == 200
    assert done.json()["freed"] > 0
    assert [c.repo for c in hub.cached_files()] == ["ggml-org/models"]
    assert fav.is_file()  # models/ is never pruned


def test_prune_never_deletes_the_model_in_the_chat(
    lab: Lab, client: TestClient, hub: FakeHub
) -> None:
    hub.download(QWEN, "Qwen3-1.7B-Q4_K_M.gguf", hub.repos[QWEN].commit)
    key = model_key(client, where="cache")
    lab.chat = cast(Any, SimpleNamespace(model=lab.model(key), state="ready", stop=lambda: None))
    preview = client.post("/lab/api/models/prune", json={}, headers=ACT).json()
    assert preview["files"][0]["in_chat"] is True
    body = {"dry_run": False, "expect": preview["revisions"]}
    assert client.post("/lab/api/models/prune", json=body, headers=ACT).status_code == 409
    assert len(hub.cached_files()) == 1


# -- models: pull by reference ---------------------------------------------------------


def test_pull_by_reference(lab: Lab, client: TestClient, hub: FakeHub, tmp_path: Path) -> None:
    def pull(ref: str, favorite: bool = False) -> Any:
        return client.post(
            "/lab/api/models/pull", json={"ref": ref, "favorite": favorite}, headers=ACT
        )

    def finished() -> bool:
        return all(
            d["state"] != "running" for d in client.get("/lab/api/status").json()["downloads"]
        )

    r = pull(f"hf://{QWEN}:Q8_0")
    assert r.status_code == 200, r.text
    assert (r.json()["file"], r.json()["favorite"]) == ("Qwen3-1.7B-Q8_0.gguf", False)
    wait(finished)
    assert hub.downloads == [f"{QWEN}/Qwen3-1.7B-Q8_0.gguf"]
    assert model_key(client, where="cache", quant="Q8_0")

    # an explicit file, kept in models/; the default quant is Q4_K_M, as in llama.cpp
    r = pull("hf://ggml-org/models/phi-2/ggml-model-q4_0.gguf", favorite=True)
    assert r.status_code == 200, r.text
    wait(finished)
    assert (tmp_path / "models" / "ggml-org--models" / "ggml-model-q4_0.gguf").is_file()
    assert model_key(client, where="favorite")
    assert pull(QWEN).json()["file"] == "Qwen3-1.7B-Q4_K_M.gguf"  # without hf://
    wait(finished)

    assert pull("hf://nobody/nothing:Q4_K_M").status_code == 400
    assert pull(f"hf://{QWEN}:Q9_9").status_code == 400
    assert pull("not a reference").status_code == 400
    assert pull("hf://ggml-org/models:F16").status_code == 400  # ambiguous: several models
    # the lab downloads the current revision; an older one needs `llb models pull`
    old = pull(f"hf://{QWEN}@0123456789abcdef:Q4_K_M")
    assert old.status_code == 400
