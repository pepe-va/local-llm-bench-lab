import os
import stat
from pathlib import Path

import pytest

from llmbench.llamacpp.introspect import binary_flags, binary_version

FIXTURES = Path(__file__).parent / "fixtures"
LLAMACPP = FIXTURES / "llamacpp" / "c811cb8"
FAKE_BENCH = FIXTURES / "bin" / "fake-llama-bench"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never touch the real $LLB_HOME or ./models from tests."""
    monkeypatch.setenv("LLB_HOME", str(tmp_path / "llbhome"))
    monkeypatch.setenv("LLB_MODELS_DIR", str(tmp_path / "models"))
    binary_flags.cache_clear()
    binary_version.cache_clear()


@pytest.fixture
def bench_help() -> str:
    return (LLAMACPP / "llama-bench.help.txt").read_text()


@pytest.fixture
def fake_bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A directory with a `llama-bench` that behaves like the real one (see the script)."""
    d = tmp_path / "bin"
    d.mkdir()
    target = d / "llama-bench"
    target.write_text(FAKE_BENCH.read_text())
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_BENCH_FIXTURE_DIR", str(LLAMACPP))
    return d


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.gguf"
    p.write_bytes(b"GGUF" + os.urandom(64))
    return p
