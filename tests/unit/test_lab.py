"""`llb lab`: Hub search and downloads, local models, chat proxy, background tests."""

import json
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from llmbench.config.loader import load_experiment
from llmbench.hub.client import RemoteFile, RepoListing
from llmbench.hub.store import ModelStore
from llmbench.lab import editor
from llmbench.lab.app import ACTION_HEADER, create_lab_app
from llmbench.lab.jobs import Jobs
from llmbench.lab.models import fit, local_models, quant_label, repo_choices
from llmbench.lab.state import Download, Lab
from llmbench.llamacpp.server import free_port
from llmbench.paths import runs_dir
from tests.helpers import FakeHub, write_gguf

REPO = Path(__file__).parents[2]
ACT = {ACTION_HEADER: "1"}
GB = 1_000_000_000


def wait(cond: Callable[[], bool], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        time.sleep(0.05)


@pytest.fixture
def root(tmp_path: Path, fake_bin_dir: Path) -> Path:
    """A repository root whose only profile uses the fake llama.cpp binaries."""
    profiles = tmp_path / "repo" / "experiments" / "_profiles"
    profiles.mkdir(parents=True)
    shutil.copy(REPO / "experiments" / "_profiles" / "deterministic.yaml", profiles)
    (profiles / "fake.yaml").write_text(
        f"llama_cpp: {{builds: [{{id: fake, bin_dir: {fake_bin_dir}}}]}}\n"
        "hardware: {cooldown: {enabled: false}, telemetry: {hz: 20, sources: [linux]}}\n"
    )
    return tmp_path / "repo"


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
def chat_model(tmp_path: Path) -> Path:
    """A loose GGUF in models/ with a reasoning chat template."""
    return write_gguf(
        tmp_path / "models" / "tiny-Q4_K_M.gguf",
        {"general.name": "tiny", "tokenizer.chat_template": "{% if enable_thinking %}<think>"},
    )


def model_key(client: TestClient, **match: Any) -> str:
    models: list[dict[str, Any]] = client.get("/lab/api/models").json()["models"]
    [m] = [m for m in models if all(m[k] == v for k, v in match.items())]
    return m["key"]


# -- pure parts -------------------------------------------------------------------


def test_quant_labels_and_fit() -> None:
    assert quant_label("Qwen3-1.7B-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_label("Qwen3-4B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert quant_label("Qwen3-1.7B-IQ4_XS.gguf") == "IQ4_XS"
    assert quant_label("phi-2/ggml-model-q4_0.gguf") == "q4_0"
    assert quant_label("Qwen3-1.7B-BF16.gguf") == "BF16"
    assert quant_label("my-model.gguf") == "my-model"
    assert fit(2 * GB, 8 * GB) == "ok"
    assert fit(6 * GB, 8 * GB) == "tight"
    assert fit(8 * GB, 8 * GB) == "no"


def test_repo_choices_group_shards_and_skip_projectors() -> None:
    listing = RepoListing(
        "org/m",
        "c" * 40,
        [
            RemoteFile("M-Q4_K_M.gguf", 2 * GB, None),
            RemoteFile("Q8_0/M-Q8_0-00002-of-00002.gguf", 3 * GB, None),
            RemoteFile("Q8_0/M-Q8_0-00001-of-00002.gguf", 4 * GB, None),
            RemoteFile("mmproj-F16.gguf", GB, None),
            RemoteFile("README.md", 10, None),
        ],
    )
    q4, q8 = repo_choices(listing, 8 * GB)
    assert (q4.quant, q4.size, q4.shards, q4.fit) == ("Q4_K_M", 2 * GB, 1, "ok")
    assert (q8.file, q8.quant, q8.size, q8.shards, q8.fit) == (
        "Q8_0/M-Q8_0-00001-of-00002.gguf", "Q8_0", 7 * GB, 2, "no",
    )  # fmt: skip


def test_new_experiment_from_presets_is_valid(
    tmp_path: Path, chat_model: Path, hub: FakeHub
) -> None:
    """With the real profiles and suites, in a folder laid out like the repository."""
    repo = tmp_path / "copy"
    (repo / "experiments").mkdir(parents=True)
    (repo / "experiments" / "_profiles").symlink_to(REPO / "experiments" / "_profiles")
    (repo / "suites").symlink_to(REPO / "suites")
    [model] = local_models(ModelStore(client=hub))
    doc = editor.new_experiment(
        [model], folder=repo / "experiments", root=repo, profile="_profiles/pi5.yaml",
        presets=["speed", "latency", "quality"],
    )  # fmt: skip
    path = repo / "experiments" / "new.yaml"
    path.write_text(editor.dump(doc))
    exp = load_experiment(path)
    assert exp.experiment.name == "lab-tiny-q4_k_m"
    [spec] = exp.models
    assert (spec.source, spec.retention) == (str(chat_model), "keep")
    assert exp.phases.bench.enabled
    assert exp.phases.latency.enabled
    assert [p.name for p in exp.phases.quality.suites] == ["es-writing.yaml"]
    assert "../suites/es-writing.yaml" in path.read_text()  # relative, like the examples
    assert exp.hardware.preflight.require_governor == "performance"  # the pi5 profile
    assert exp.defaults.sampling.temperature == 0  # deterministic.yaml


def test_patch_keeps_comments_and_layout() -> None:
    text = "# top comment\nversion: 1\ndefaults:\n  load: { threads: 4 }  # four\n"
    out = editor.patch(text, ["defaults", "load", "ctx_size"], 8192)
    assert out.startswith("# top comment\n")
    assert "# four" in out
    assert editor.parse(out)["defaults"]["load"] == {"threads": 4, "ctx_size": 8192}
    out = editor.patch(out, ["defaults", "server", "reasoning"], "off")
    assert 'reasoning: "off"' in out  # quoted: YAML 1.1 would read a bare off as false
    out = editor.patch(out, ["matrix", "load.threads"], [2, 4])
    assert "load.threads: [2, 4]" in out
    out = editor.patch(out, ["models", 0], {"id": "a", "source": "hf://o/r:Q4_K_M"})
    assert editor.parse(out)["models"] == [{"id": "a", "source": "hf://o/r:Q4_K_M"}]
    out = editor.patch(out, ["defaults", "server", "reasoning"], None)
    assert "server" not in editor.parse(out)["defaults"]  # emptied mappings go too
    out = editor.patch(out, ["models", 0], None)
    assert editor.parse(out)["models"] == []
    with pytest.raises(editor.EditorError):
        editor.patch("a: [", ["a"], 1)
    with pytest.raises(editor.EditorError) as e:
        editor.parse("a: 1\nb: [\n")
    assert e.value.line is not None


def test_field_help_names_the_flags_from_argv() -> None:
    help_ = editor.field_help()
    assert "flag -t" in help_["load.threads"]
    assert "-np" in help_["server.parallel"]
    assert "--temp" in help_["sampling.temperature"]
    assert "CPU threads" in help_["load.threads"]  # the dashboard's own wording first


# -- web API ----------------------------------------------------------------------


def test_actions_need_the_lab_header(client: TestClient) -> None:
    r = client.post("/lab/api/chat/stop")
    assert r.status_code == 403
    assert client.post("/lab/api/chat/stop", headers=ACT).status_code == 200
    assert client.get("/lab", follow_redirects=False).headers["location"] == "/lab/models"
    for tab in ("models", "chat", "test"):
        assert client.get(f"/lab/{tab}").status_code == 200
    assert client.get("/lab/static/form.js").status_code == 200
    assert client.get("/lab/static/../app.py").status_code == 404
    assert 'href="/lab/models"' in client.get("/").text  # the dashboard links to the lab


def test_search_list_download_keep_and_delete(client: TestClient, hub: FakeHub) -> None:
    hits = client.get("/lab/api/search", params={"q": "qwen3 1.7b"}).json()
    assert [h["repo"] for h in hits] == ["unsloth/Qwen3-1.7B-GGUF"]
    direct = client.get("/lab/api/search", params={"q": "hf://unsloth/Qwen3-1.7B-GGUF:Q8_0"})
    assert direct.json() == [
        {"repo": "unsloth/Qwen3-1.7B-GGUF", "downloads": None, "likes": None, "direct": True}
    ]

    url = "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/tree/main"
    repo = client.get("/lab/api/repo", params={"id": url}).json()
    assert repo["repo"] == "unsloth/Qwen3-1.7B-GGUF"
    quants = {c["quant"]: c for c in repo["choices"]}
    assert {"Q4_K_M", "Q8_0", "BF16"} <= set(quants)
    assert all(c["fit"] in ("ok", "tight", "no") for c in repo["choices"])
    missing = client.get("/lab/api/repo", params={"id": "nobody/nothing"})
    assert missing.status_code == 400

    file = quants["Q4_K_M"]["file"]
    body = {"repo": "unsloth/Qwen3-1.7B-GGUF", "file": file}
    assert client.post("/lab/api/downloads", json=body, headers=ACT).status_code == 200
    wait(lambda: client.get("/lab/api/status").json()["downloads"][0]["state"] != "running")
    [d] = client.get("/lab/api/status").json()["downloads"]
    assert (d["state"], d["done"]) == ("done", d["size"])
    assert hub.downloads == [f"unsloth/Qwen3-1.7B-GGUF/{file}"]

    key = model_key(client, where="cache")
    [m] = client.get("/lab/api/models").json()["models"]
    assert (m["quant"], m["info"]["architecture"]) == ("Q4_K_M", "llama")
    assert client.post(f"/lab/api/models/{key}/favorite", headers=ACT).status_code == 200
    fav = model_key(client, where="favorite")
    assert client.delete(f"/lab/api/models/{fav}", headers=ACT).status_code == 200
    assert client.get("/lab/api/models").json()["models"] == []

    bad = client.post("/lab/api/downloads", json={**body, "file": "nope.gguf"}, headers=ACT)
    assert bad.status_code == 400


def chat_events(client: TestClient, text: str) -> list[dict[str, Any]]:
    body = {"messages": [{"role": "user", "content": text}], "temperature": 0}
    r = client.post("/lab/api/chat", json=body, headers=ACT)
    assert r.status_code == 200
    return [json.loads(line[6:]) for line in r.text.splitlines() if line.startswith("data: ")]


def test_chat_streams_answers_with_metrics(client: TestClient, chat_model: Path) -> None:
    key = model_key(client, where="local")
    [m] = client.get("/lab/api/models").json()["models"]
    assert m["info"]["reasoning"]
    assert m["source"] == str(chat_model)
    builds = client.get("/lab/api/builds").json()
    assert builds[0]["id"] == "given"

    start = {"model": key, "ctx_size": 2048, "threads": 2, "reasoning": "off"}
    r = client.post("/lab/api/chat/start", json=start, headers=ACT)
    assert r.status_code == 200, r.text
    wait(lambda: client.get("/lab/api/status").json()["chat"]["state"] != "loading")
    chat = client.get("/lab/api/status").json()["chat"]
    assert chat["state"] == "ready", chat["error"]
    assert chat["argv"][chat["argv"].index("-c") :][:2] == ["-c", "2048"]
    assert "-rea" in chat["argv"]

    events = chat_events(client, "hola que tal")
    assert "".join(e.get("content", "") for e in events) == "hola que tal"
    metrics = events[-1]["metrics"]
    assert metrics["ttft_ms"] is not None
    assert metrics["predicted_n"] == 3
    assert metrics["gen_tps"] > 0

    # the model in use cannot be deleted
    assert client.delete(f"/lab/api/models/{key}", headers=ACT).status_code == 409
    assert client.post("/lab/api/chat/stop", headers=ACT).status_code == 200
    assert chat_events(client, "x") == [{"error": "the chat is stopped"}]


def test_editor_saves_checks_and_runs_like_the_terminal(
    client: TestClient, lab: Lab, chat_model: Path
) -> None:
    key = model_key(client, where="local")
    client.post("/lab/api/chat/start", json={"model": key}, headers=ACT)
    wait(lambda: client.get("/lab/api/status").json()["chat"]["state"] == "ready")
    meta = client.get("/lab/api/editor/meta").json()
    assert meta["profiles"] == ["_profiles/deterministic.yaml", "_profiles/fake.yaml"]
    assert meta["default_profile"] == "_profiles/fake.yaml"
    assert "-t" in meta["help"]["load.threads"]
    assert "ModelSpec" in meta["schema"]["$defs"]

    body = {"models": [key], "profile": "_profiles/fake.yaml", "presets": ["speed"]}
    text = client.post("/lab/api/editor/new", json=body, headers=ACT).json()["text"]
    text = client.post(
        "/lab/api/editor/patch", json={"text": text, "path": ["phases", "bench", "repetitions"],
                                       "value": 2}, headers=ACT,
    ).json()["text"]  # fmt: skip
    check = {"text": text, "command": "validate"}
    r = client.post("/lab/api/editor/check", json=check, headers=ACT).json()
    assert r["ok"], r["output"]
    assert "<draft>" in r["output"] or "schema" in r["output"]
    assert not list(lab.experiments.folder.glob(".llb-lab-*"))  # the draft is gone
    plan = client.post("/lab/api/editor/check", json={**check, "command": "plan"}, headers=ACT)
    assert "x 2" in plan.json()["output"]  # 2 tests x 2 repetitions

    save = {"name": "mine.yaml", "text": text}
    saved = client.post("/lab/api/editor/save", json=save, headers=ACT).json()
    assert saved["files"] == ["mine.yaml"]
    [card] = saved["experiments"]  # the picker's card
    assert card["file"] == "mine.yaml"
    assert card["phases"] == ["bench"]
    assert len(card["models"]) == 1
    assert card["results"] is None
    assert client.post("/lab/api/editor/save", json=save, headers=ACT).status_code == 400
    bad = client.post("/lab/api/editor/save", json={**save, "name": "../x.yaml"}, headers=ACT)
    assert bad.status_code == 400
    assert client.get("/lab/api/editor/file", params={"name": "mine.yaml"}).json()["text"] == text

    run = {"name": "mine.yaml", "command": "run"}
    r = client.post("/lab/api/editor/command", json=run, headers=ACT)
    assert r.status_code == 200, r.text
    assert r.json()["stopped_chat"]
    assert client.get("/lab/api/status").json()["chat"]["state"] == "stopped"
    # nothing else heavy may start while it measures
    assert client.post("/lab/api/editor/command", json=run, headers=ACT).status_code == 409
    assert client.post("/lab/api/chat/start", json={"model": key}, headers=ACT).status_code == 409

    wait(lambda: not client.get("/lab/api/status").json()["job"]["running"], timeout=120)
    job = client.get("/lab/api/status").json()["job"]
    assert job["state"] == "ok", job["tail"]
    assert (runs_dir() / job["experiment"] / "benchmark.sqlite").is_file()
    [card] = client.get("/lab/api/editor/experiments").json()["experiments"]
    assert card["results"] == str(runs_dir() / job["experiment"])
    assert client.get(job["results"]).status_code == 200

    guide = client.post("/lab/api/editor/command", json={**run, "command": "guide"}, headers=ACT)
    assert "guidance.md" in guide.json()["sync"]["output"]


def test_new_experiments_get_the_chosen_name_or_a_free_one(
    client: TestClient, lab: Lab, chat_model: Path
) -> None:
    key = model_key(client, where="local")
    body: dict[str, Any] = {"models": [key], "presets": ["speed"]}
    named = client.post("/lab/api/editor/new", json={**body, "name": "my-exp"}, headers=ACT)
    assert editor.parse(named.json()["text"])["experiment"]["name"] == "my-exp"
    bad = client.post("/lab/api/editor/new", json={**body, "name": "two words"}, headers=ACT)
    assert bad.status_code == 400
    first = editor.parse(client.post("/lab/api/editor/new", json=body, headers=ACT).json()["text"])
    name = first["experiment"]["name"]
    (lab.experiments.folder / f"{name}.yaml").write_text("version: 1\n")
    again = editor.parse(client.post("/lab/api/editor/new", json=body, headers=ACT).json()["text"])
    assert again["experiment"]["name"] == f"{name}-2"
    cards = client.get("/lab/api/editor/experiments").json()["experiments"]
    assert [c["file"] for c in cards] == [f"{name}.yaml"]
    assert cards[0]["phases"] == [
        "bench"
    ]  # bench runs unless disabledlab: Lab, client: TestClient) -> None:
    bad = client.post("/lab/api/system/build", json={"ref": "b1; rm -rf /"}, headers=ACT)
    assert bad.status_code == 400
    # a download still writing to disk (no thread behind it: it never finishes)
    lab.downloads["x"] = Download("x", "org/m", "m.gguf", "c" * 40, [("m.gguf", 1)], time.time())
    build = client.post("/lab/api/system/build", json={"ref": "b6500"}, headers=ACT)
    assert build.status_code == 409
    assert lab.jobs.current is None


def test_a_restarted_lab_adopts_a_running_job(tmp_path: Path) -> None:
    jobs = Jobs(tmp_path)
    job = jobs.start("build", "help", ["--help"], cwd=tmp_path)
    wait(lambda: not job.running)
    assert job.to_json()["state"] == "ok"
    assert "Usage" in job.tail()
    assert Jobs(tmp_path).current is None  # finished: nothing to adopt

    # a long llb command outlives the lab that started it, and the next lab finds it
    long = jobs.start("run", "serve", ["serve", "--port", str(free_port())], cwd=tmp_path)
    try:
        adopted = Jobs(tmp_path).current
        assert adopted is not None
        assert (adopted.pid, adopted.running, adopted.returncode) == (long.pid, True, None)
        adopted.cancel()  # Ctrl-C to the job's process group
        wait(lambda: not long.running)
        assert adopted.to_json()["state"] == "finished"  # no exit code for an adopted job
    finally:
        long.cancel()

    sleeper = subprocess.Popen(["sleep", "30"])  # not an llb process: never adopted
    try:
        state: dict[str, Any] = {
            "kind": "test", "title": "t", "argv": [], "log": "x", "pid": sleeper.pid,
            "started_at": "now",
        }  # fmt: skip
        (tmp_path / "job.json").write_text(json.dumps(state))
        assert Jobs(tmp_path).current is None
    finally:
        sleeper.kill()
        sleeper.wait()


def test_new_experiment_puts_each_suite_in_its_phase(
    client: TestClient, root: Path, chat_model: Path
) -> None:
    (root / "suites").mkdir()
    (root / "suites" / "syn.yaml").write_text(
        "name: syn\nsource:\n  synthetic: {prompt_tokens: 64, output_tokens: 8, count: 2}\n"
    )
    (root / "suites" / "qa.yaml").write_text(
        "name: qa\nendpoint: chat\nsource:\n  items:\n    - {prompt: hi, reference: hi}\n"
    )
    meta = client.get("/lab/api/editor/meta").json()
    assert meta["suites"] == ["../suites/qa.yaml", "../suites/syn.yaml"]
    key = model_key(client, where="local")
    body = {"models": [key], "profile": "_profiles/fake.yaml", "suites": meta["suites"]}
    doc = client.post("/lab/api/editor/new", json=body, headers=ACT).json()["doc"]
    assert doc["phases"]["quality"] == {"enabled": True, "suites": ["../suites/qa.yaml"]}
    assert doc["phases"]["latency"] == {"enabled": True, "suites": ["../suites/syn.yaml"]}
    assert doc["phases"]["bench"] == {"enabled": False}  # no preset asked for speed

    bad = client.post(
        "/lab/api/editor/new", json={**body, "suites": ["../../etc/passwd"]}, headers=ACT
    )
    assert bad.status_code == 400


def test_the_manuals_experiment_example_is_valid(client: TestClient, tmp_path: Path) -> None:
    import html  # noqa: PLC0415
    import re  # noqa: PLC0415

    page = client.get("/lab/info").text
    [block] = re.findall(r'<pre data-example="experiment">(.*?)</pre>', page, re.S)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "_profiles").symlink_to(REPO / "experiments" / "_profiles")
    (tmp_path / "suites").symlink_to(REPO / "suites")
    exp = tmp_path / "experiments" / "example.yaml"
    exp.write_text(html.unescape(block))
    loaded = load_experiment(exp)
    assert loaded.matrix.zip
    assert len(loaded.gates) == 2


def test_the_manual_covers_every_tab_and_its_links_work(client: TestClient) -> None:
    import re  # noqa: PLC0415

    from llmbench.lab.suites import EVALUATOR_GUIDE  # noqa: PLC0415

    page = client.get("/lab/info").text
    manual = page[page.index('<div class="manual">') :]
    for anchor in re.findall(r'<a href="#([\w-]+)"', manual):
        assert f'id="{anchor}"' in manual, anchor
    for href in set(re.findall(r'href="(/[^"#]*)"', manual)):
        assert client.get(href).status_code == 200, href
    tabs = re.findall(r'<a href="/lab/(\w+)"', page[: page.index("<main>")])
    assert set(tabs) == {"models", "chat", "test", "suites", "system", "info"}
    for tab in set(tabs) - {"info"}:
        assert f'href="/lab/{tab}">{tab}</a></td>' in manual, tab  # a row in the tabs table
    for tip in EVALUATOR_GUIDE:
        assert f"<code>{tip.type}</code>" in manual
    # every metric the manual names is one llb really produces
    src = Path(__file__).parents[2] / "src" / "llmbench"
    code = "\n".join(p.read_text() for p in src.rglob("*.py"))
    suffixes = r"ms_p50|mean|rate|failed|per_s|aggregate|mb|c"
    metrics = re.findall(rf"<code>([a-z_]+_(?:{suffixes}))", manual)
    assert len(metrics) >= 10
    for metric in metrics:
        assert metric in code, metric
