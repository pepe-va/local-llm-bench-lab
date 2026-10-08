"""`llb serve` dashboard (FastAPI TestClient over a temporary store)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llmbench.results.models import Measurement, RequestResult
from llmbench.serve.app import create_app, discover
from tests.unit.test_gates import record


@pytest.fixture
def results(tmp_path: Path) -> Path:
    from llmbench.results.store import Store  # noqa: PLC0415

    out = tmp_path / "runs" / "exp"
    store = Store(out / "benchmark.sqlite")
    m = Measurement("tg", 0, 128, 0, 9.5, 0.2, 1, 1, [9.4, 9.6], {}, 9.5, 0.02, False)
    store.save_run(record("b", {"load.threads": 4}, "bench", measurements=[m]))
    store.save_run(
        record(
            "q", {"load.threads": 4}, "quality", suite="es",
            metrics={"score_mean": 0.5, "ttft_ms_p50": 120.0},
            requests=[RequestResult("es/a#0", True, ttft_ms=120, content="Hola", score=0.5)],
        )
    )  # fmt: skip
    store.close()
    return tmp_path / "runs"


@pytest.fixture
def client(results: Path) -> TestClient:
    return TestClient(create_app([results]))


def test_discovery(results: Path) -> None:
    refs = discover([results])
    assert len(refs) == 1
    assert discover([results / "exp"]) == refs
    assert discover([results / "exp" / "benchmark.sqlite"]) == refs
    assert discover([results / "missing"]) == {}


def test_index_and_api(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "/e/" in r.text
    [exp] = client.get("/api/experiments").json()
    assert (exp["experiment"], exp["runs"], exp["failed"]) == ("exp", 2, 0)
    configs = client.get(f"/api/e/{exp['sid']}/exp/configs").json()
    assert configs[0]["metrics"]["tps:tg128"] == 9.5


def test_experiment_runs_and_run_pages(client: TestClient) -> None:
    sid = client.get("/api/experiments").json()[0]["sid"]
    page = client.get(f"/e/{sid}/exp")
    assert page.status_code == 200
    assert 'hx-trigger="load, every 3s"' in page.text  # live progress
    assert "/static/plotly.min.js" in page.text
    partial = client.get(f"/e/{sid}/exp/runs")
    assert "tg128 9.5 t/s" in partial.text
    assert "score 0.500" in partial.text
    runs = client.get(f"/api/e/{sid}/exp/runs").json()
    detail = client.get(f"/run/{sid}/{runs[0]['id']}")
    assert detail.status_code == 200
    assert "Requests (1)" in detail.text
    assert client.get(f"/api/run/{sid}/{runs[0]['id']}").json()["requests"][0]["content"] == "Hola"


def test_static_assets_are_served_offline(client: TestClient) -> None:
    assert "htmx" in client.get("/static/htmx.min.js").text[:200]
    assert len(client.get("/static/plotly.min.js").content) > 1_000_000


def test_unknown_things_are_404(client: TestClient) -> None:
    assert client.get("/e/nope/exp").status_code == 404
    sid = client.get("/api/experiments").json()[0]["sid"]
    assert client.get(f"/e/{sid}/other").status_code == 404
    assert client.get(f"/run/{sid}/9999").status_code == 404


def test_gates_shown_when_experiment_yaml_given(results: Path, tmp_path: Path) -> None:
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "experiment: {name: exp}\nllama_cpp: {builds: [{id: b, bin_dir: /x}]}\n"
        "models: [{id: m, source: m.gguf}]\n"
        "gates: [{metric: 'tps:tg128', op: '>=', value: 5}]\n"
    )
    client = TestClient(create_app([results], [exp]))
    sid = client.get("/api/experiments").json()[0]["sid"]
    assert "Gates" in client.get(f"/e/{sid}/exp").text
