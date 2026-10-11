"""`llb lab` suites tab: files, comment-keeping patches, validation/preview and "try it"."""

import json
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmbench.config.evaluators import All, Contains, Length, Weighted, WeightedPart
from llmbench.config.loader import load_suite
from llmbench.config.suite import HfDataset
from llmbench.data.hf_datasets import DatasetError, LoadedDataset
from llmbench.hub.store import ModelStore
from llmbench.lab import suites as logic
from llmbench.lab.app import ACTION_HEADER, create_lab_app
from llmbench.lab.jobs import Jobs
from llmbench.lab.state import Lab
from tests.helpers import FakeHub, write_gguf

REPO = Path(__file__).parents[2]
ACT = {ACTION_HEADER: "1"}

CHECKS = """\
# my comment
version: 1
name: checks
endpoint: chat
max_tokens: 64
source:
  items:
    - id: paris
      prompt: The capital of France is Paris
      evaluator: { type: contains, values: [Paris] }
    - id: berlin
      prompt: The capital of France is Paris
      evaluator: { type: contains, values: [Berlin] }
    - id: exact
      prompt: hello
      reference: hello
      evaluator: { type: exact }
    - id: judged
      prompt: write
      evaluator: { type: llm_judge }
    - id: plain
      prompt: no evaluator here
"""


def wait(cond: Callable[[], bool], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A repository root with copies of some real suites and one experiment using one."""
    repo = tmp_path / "repo"
    (repo / "suites").mkdir(parents=True)
    for name in ("es-writing.yaml", "synthetic-512x128.yaml", "hf-gsm8k.yaml", "rerank-es.yaml"):
        shutil.copy(REPO / "suites" / name, repo / "suites")
    (repo / "experiments").mkdir()
    (repo / "experiments" / "uses.yaml").write_text(
        "version: 1\nphases:\n  quality: { enabled: true, suites: [../suites/es-writing.yaml] }\n"
        "  latency: { suites: [../suites/es-writing.yaml, ../suites/missing.yaml] }\n"
    )
    (repo / "experiments" / "broken.yaml").write_text("phases: [unclosed\n")
    return repo


@pytest.fixture
def lab(root: Path, hub: FakeHub, fake_bin_dir: Path) -> Iterator[Lab]:
    the_lab = Lab(
        root=root, store=ModelStore(client=hub), llama_server=fake_bin_dir / "llama-server"
    )
    yield the_lab
    the_lab.close()


@pytest.fixture
def client(lab: Lab) -> TestClient:
    return TestClient(create_lab_app(lab))


@pytest.fixture
def chat_ready(client: TestClient, tmp_path: Path) -> TestClient:
    """The lab chat running the fake llama-server (it echoes the last message)."""
    write_gguf(tmp_path / "models" / "tiny-Q4_K_M.gguf", {"general.name": "tiny"})
    [m] = client.get("/lab/api/models").json()["models"]
    r = client.post("/lab/api/chat/start", json={"model": m["key"]}, headers=ACT)
    assert r.status_code == 200, r.text
    wait(lambda: client.get("/lab/api/status").json()["chat"]["state"] != "loading")
    assert client.get("/lab/api/status").json()["chat"]["state"] == "ready"
    return client


def events(client: TestClient, body: dict[str, Any]) -> list[dict[str, Any]]:
    r = client.post("/lab/api/suites/try", json=body, headers=ACT)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")
    return [json.loads(line) for line in r.text.splitlines() if line]


# -- pure parts -------------------------------------------------------------------


def test_describe_evaluator_and_literal_blocks() -> None:
    spec = All(
        type="all",
        of=[
            Contains(type="contains", values=["a"]),
            Weighted(
                type="weighted",
                of=[WeightedPart(weight=2, evaluator=Length(type="length", max_words=3))],
            ),
        ],
    )
    assert logic.describe_evaluator(spec) == "all(contains, weighted(2*length))"
    assert logic.describe_evaluator(None) is None
    out = logic.literal_blocks({"a": "one line", "b": ["x\ny"], "c": 3})
    assert type(out["a"]) is str
    assert type(out["b"][0]).__name__ == "LiteralScalarString"


def test_every_new_suite_kind_is_valid(tmp_path: Path) -> None:
    folder = tmp_path / "suites"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "k-file.jsonl").write_text('{"question": "q", "answer": "a"}\n')
    files = logic.Suites(folder, tmp_path / "experiments")
    for kind in logic.KINDS:
        text = logic.new_suite(kind.key, f"k-{kind.key}")
        suite = files.load(text)
        assert suite.name == f"k-{kind.key}"
        assert logic.source_kind(suite) == kind.key.replace("chat", "items").replace(
            "embeddings", "items"
        ).replace("rerank", "items")
    with pytest.raises(logic.EditorError):
        logic.new_suite("nope", "x")
    with pytest.raises(logic.EditorError):
        logic.new_suite("chat", "../evil")
    assert list(folder.glob(".llb-lab-*")) == []  # drafts are removed


# -- files ------------------------------------------------------------------------


def test_page_meta_and_files(client: TestClient, root: Path) -> None:
    page = client.get("/lab/suites")
    assert page.status_code == 200
    assert "/lab/static/suites.js" in page.text
    assert client.get("/lab/static/suites.js").status_code == 200
    meta = client.get("/lab/api/suites/meta").json()
    assert meta["files"] == [
        "es-writing.yaml", "hf-gsm8k.yaml", "rerank-es.yaml", "synthetic-512x128.yaml",
    ]  # fmt: skip
    assert meta["used_by"] == {"es-writing.yaml": ["uses.yaml"], "missing.yaml": ["uses.yaml"]}
    assert meta["folder"] == str(root / "suites")
    assert {k["key"] for k in meta["kinds"]} >= {"chat", "synthetic", "file", "hf_dataset"}
    assert "EvaluatorSpec" in meta["schema"]["$defs"]
    r = client.get("/lab/api/suites/file", params={"name": "es-writing.yaml"})
    assert r.json()["text"] == (root / "suites" / "es-writing.yaml").read_text()


@pytest.mark.parametrize("name", ["../experiments/uses.yaml", "x.txt", ".hidden.yaml", "a/b.yaml"])
def test_bad_names_are_refused(client: TestClient, name: str) -> None:
    assert client.get("/lab/api/suites/file", params={"name": name}).status_code == 400
    r = client.post("/lab/api/suites/save", json={"name": name, "text": "a: 1"}, headers=ACT)
    assert r.status_code == 400


def test_save_asks_before_overwriting(client: TestClient, root: Path) -> None:
    body = {"name": "new.yaml", "text": "# hi\nname: new\n"}
    assert client.post("/lab/api/suites/save", json=body).status_code == 403  # no header
    r = client.post("/lab/api/suites/save", json=body, headers=ACT)
    assert r.status_code == 200, r.text
    assert "new.yaml" in r.json()["files"]
    assert (root / "suites" / "new.yaml").read_text() == "# hi\nname: new\n"
    r = client.post("/lab/api/suites/save", json=body, headers=ACT)
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]["message"]
    again = body | {"text": "name: newer\n", "overwrite": True}
    assert client.post("/lab/api/suites/save", json=again, headers=ACT).status_code == 200
    bad = {"name": "bad.yaml", "text": "a: [1"}
    assert client.post("/lab/api/suites/save", json=bad, headers=ACT).status_code == 400
    assert not (root / "suites" / "bad.yaml").exists()


def test_patch_keeps_comments_and_writes_blocks(client: TestClient) -> None:
    body = {"text": CHECKS, "path": ["source", "items", 0, "prompt"], "value": "Line one\nTwo\n"}
    r = client.post("/lab/api/suites/patch", json=body, headers=ACT)
    assert r.status_code == 200, r.text
    text = r.json()["text"]
    assert text.startswith("# my comment\n")
    assert "prompt: |\n        Line one\n        Two\n" in text
    assert r.json()["doc"]["source"]["items"][0]["prompt"] == "Line one\nTwo\n"
    evaluator = {"type": "all", "of": [{"type": "regex", "pattern": "x"}]}
    body = {"text": text, "path": ["source", "items", 1, "evaluator"], "value": evaluator}
    r = client.post("/lab/api/suites/patch", json=body, headers=ACT)
    assert r.json()["doc"]["source"]["items"][1]["evaluator"] == evaluator
    body = {"text": r.json()["text"], "path": ["source", "items", 4], "value": None}
    doc = client.post("/lab/api/suites/patch", json=body, headers=ACT).json()["doc"]
    assert [i["id"] for i in doc["source"]["items"]] == ["paris", "berlin", "exact", "judged"]
    bad = {"text": "a: [1", "path": ["a"], "value": 1}
    assert client.post("/lab/api/suites/patch", json=bad, headers=ACT).status_code == 400


def test_new_suite_endpoint(client: TestClient) -> None:
    r = client.post("/lab/api/suites/new", json={"kind": "synthetic", "name": "fast"}, headers=ACT)
    assert r.status_code == 200
    assert r.json()["doc"]["name"] == "fast"
    assert r.json()["doc"]["source"]["synthetic"]["count"] == 20
    bad = {"kind": "synthetic", "name": "a b"}
    assert client.post("/lab/api/suites/new", json=bad, headers=ACT).status_code == 400


# -- validate & preview ---------------------------------------------------------------


def check(client: TestClient, text: str, **extra: Any) -> dict[str, Any]:
    r = client.post("/lab/api/suites/check", json={"text": text, **extra}, headers=ACT)
    assert r.status_code == 200, r.text
    return r.json()


def test_preview_items_like_a_run(client: TestClient, root: Path) -> None:
    text = (root / "suites" / "es-writing.yaml").read_text()
    p = check(client, text)
    suite = load_suite(root / "suites" / "es-writing.yaml")
    assert (p["name"], p["endpoint"], p["source"]) == ("es-writing", "chat", "items")
    assert p["count"] == len(suite.source.items or [])
    first = p["requests"][0]
    assert first["id"] == "es-writing/correccion-1"
    assert first["evaluator"] == "all(contains, length, llm_judge)"
    assert first["evaluator_spec"]["type"] == "all"
    assert first["max_tokens"] == 256
    assert first["system"].startswith("Eres un asistente")
    assert list((root / "suites").glob(".llb-lab-*")) == []


def test_preview_synthetic_without_a_server(client: TestClient, root: Path) -> None:
    p = check(client, (root / "suites" / "synthetic-512x128.yaml").read_text())
    assert p["source"] == "synthetic"
    assert p["shapes"]["prompt_tokens"]["min"] == 512
    first = {"id": "synthetic-512x128/00000", "prompt_tokens": 512, "max_tokens": 128}
    assert p["requests"][0] == first
    assert len(p["requests"]) == min(p["count"], logic.PREVIEW)
    assert any("tokenizer" in n for n in p["notes"])


def test_preview_dataset_only_when_asked(
    client: TestClient, root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (root / "suites" / "hf-gsm8k.yaml").read_text()
    p = check(client, text)
    assert p["dataset"]["repo"] == "openai/gsm8k"
    assert p["requests"] == []

    def fake_load(spec: HfDataset) -> LoadedDataset:
        rows = [{"question": "1+1?", "answer": "so #### 2"}]
        return LoadedDataset(spec.repo, spec.config, spec.split, "c" * 40, rows)

    monkeypatch.setattr("llmbench.data.sources.load_hf_dataset", fake_load)
    p = check(client, text, fetch=True)
    assert p["count"] == 1
    assert p["requests"][0]["prompt"].startswith("1+1?\n\nEnd your answer")
    assert p["requests"][0]["evaluator"] == "regex_extract"

    def offline(spec: HfDataset) -> LoadedDataset:
        raise DatasetError(f"{spec.repo}: cannot list the parquet conversion (offline)")

    monkeypatch.setattr("llmbench.data.sources.load_hf_dataset", offline)
    r = client.post("/lab/api/suites/check", json={"text": text, "fetch": True}, headers=ACT)
    assert r.status_code == 400
    assert "openai/gsm8k" in r.json()["detail"]["message"]


def test_preview_file_source_relative_to_suites(client: TestClient, root: Path) -> None:
    (root / "suites" / "data").mkdir()
    rows = [{"q": f"question {i}", "a": str(i)} for i in range(25)]
    (root / "suites" / "data" / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    text = (
        "name: rows\nendpoint: chat\nsource: { file: { path: data/rows.jsonl } }\n"
        "columns: { prompt: q, reference: a }\nevaluator: { type: exact }\n"
    )
    p = check(client, text)
    assert p["count"] == 25
    assert len(p["requests"]) == logic.PREVIEW
    third = p["requests"][3]
    assert (third["prompt"], third["reference"], third["evaluator"]) == ("question 3", "3", "exact")
    wrong = {"text": text.replace("q,", "nope,")}
    r = client.post("/lab/api/suites/check", json=wrong, headers=ACT)
    assert r.status_code == 400
    assert "column 'nope' not found" in r.json()["detail"]["message"]


def test_invalid_suites_report_the_line(client: TestClient) -> None:
    text = "name: x\nsource:\n  items:\n    - prompt: hi\n      evaluator: { type: nope }\n"
    r = client.post("/lab/api/suites/check", json={"text": text}, headers=ACT)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["line"] == 5
    assert "<draft>" in detail["message"]
    r = client.post("/lab/api/suites/check", json={"text": "name: x\n"}, headers=ACT)
    assert r.status_code == 400
    assert "source" in r.json()["detail"]["message"]
    r = client.post("/lab/api/suites/check", json={"text": ""}, headers=ACT)
    assert r.json()["detail"]["message"] == "the suite is empty"


# -- try it -----------------------------------------------------------------------------


def test_try_needs_the_chat_and_no_job(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    r = client.post("/lab/api/suites/try", json={"text": CHECKS}, headers=ACT)
    assert r.status_code == 409
    assert "chat" in r.json()["detail"]
    monkeypatch.setattr(Jobs, "busy", property(lambda _: True))
    r = client.post("/lab/api/suites/try", json={"text": CHECKS}, headers=ACT)
    assert r.status_code == 409
    assert "job" in r.json()["detail"]
    r = client.post("/lab/api/suites/try", json={"text": CHECKS, "limit": 0}, headers=ACT)
    assert r.status_code == 422


def test_try_scores_the_chat_answers(chat_ready: TestClient) -> None:
    out = events(chat_ready, {"text": CHECKS, "limit": 10, "max_tokens": 8})
    assert out[0]["model"]["quant"] == "Q4_K_M"
    assert out[1]["start"]["count"] == 5
    assert out[1]["start"]["sampling"] == {"temperature": 0.0}
    results = {e["result"]["id"]: e["result"] for e in out if "result" in e}
    assert results["checks/paris"]["passed"] is True
    assert results["checks/paris"]["answer"] == "The capital of France is Paris"  # fake echo
    assert results["checks/paris"]["max_tokens"] == 8
    assert results["checks/berlin"]["passed"] is False
    assert "Berlin" in results["checks/berlin"]["detail"]
    assert results["checks/exact"]["score"] == 1
    assert results["checks/judged"]["judge"] is True
    assert results["checks/judged"]["score"] is None
    assert "score" not in results["checks/plain"]
    assert all(r["ok"] and r["e2e_ms"] is not None for r in results.values())
    done = out[-1]["done"]
    assert (done["sent"], done["ok"], done["scored"], done["passed"]) == (5, 5, 3, 2)
    assert done["judge_pending"] == 1


def test_try_limits_and_synthetic(chat_ready: TestClient, root: Path) -> None:
    out = events(chat_ready, {"text": CHECKS, "limit": 2})
    assert [e["result"]["id"] for e in out if "result" in e] == ["checks/paris", "checks/berlin"]
    assert out[1]["start"]["of"] == 5
    synthetic = (root / "suites" / "synthetic-512x128.yaml").read_text()
    out = events(chat_ready, {"text": synthetic, "limit": 2, "max_tokens": 4})
    assert out[1]["start"]["sampling"] == {"temperature": 0.0, "ignore_eos": True}
    rows = [e["result"] for e in out if "result" in e]
    assert [r["prompt_tokens"] for r in rows] == [512, 512]
    assert all(r["ok"] and r["predicted_n"] == 4 for r in rows)
    rerank = (root / "suites" / "rerank-es.yaml").read_text()
    out = events(chat_ready, {"text": rerank})
    assert "experiment" in out[-1]["error"]


def test_try_stops_when_the_chat_goes(chat_ready: TestClient, lab: Lab) -> None:
    suite = lab.chat
    assert suite is not None
    files = logic.Suites(lab.root / "suites", lab.experiments.folder)
    parsed = files.load(CHECKS)
    calls = iter([None, "the chat model was unloaded"])
    url = str(vars(suite)["_server"].url)
    out = list(logic.try_events(parsed, url, logic.TryOptions(limit=5), alive=lambda: next(calls)))
    assert "result" in out[1]
    assert out[-1]["error"] == "stopped: the chat model was unloaded"
    assert out[-1]["totals"]["sent"] == 1
