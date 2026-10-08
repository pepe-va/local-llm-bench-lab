import stat
from pathlib import Path

import pytest

from llmbench.llamacpp.introspect import binary_flags, binary_version
from tests.helpers import FakeHub, write_gguf

FIXTURES = Path(__file__).parent / "fixtures"
LLAMACPP = FIXTURES / "llamacpp" / "c811cb8"
FAKE_BENCH = FIXTURES / "bin" / "fake-llama-bench"
FAKE_SERVER = FIXTURES / "bin" / "fake-llama-server"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the real $LLB_HOME or ./models from tests."""
    monkeypatch.setenv("LLB_HOME", str(tmp_path / "llbhome"))
    monkeypatch.setenv("LLB_MODELS_DIR", str(tmp_path / "models"))
    # tests never reach the network; a real Hub client would fail fast in offline mode
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    binary_flags.cache_clear()
    binary_version.cache_clear()


@pytest.fixture
def bench_help() -> str:
    return (LLAMACPP / "llama-bench.help.txt").read_text()


@pytest.fixture
def fake_bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """`llama-bench` and `llama-server` stand-ins that behave like the real ones."""
    d = tmp_path / "bin"
    d.mkdir()
    for name, src in (("llama-bench", FAKE_BENCH), ("llama-server", FAKE_SERVER)):
        target = d / name
        target.write_text(src.read_text())
        target.chmod(target.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_BENCH_FIXTURE_DIR", str(LLAMACPP))
    return d


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "tiny.gguf", {"general.file_type": 15, "general.name": "tiny"})


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    """In-memory Hub (real listings captured from huggingface.co) used by every ModelStore."""
    fake = FakeHub.from_fixture(tmp_path / "hf-cache")
    monkeypatch.setattr("llmbench.hub.store.HuggingFaceClient", lambda: fake)
    return fake
