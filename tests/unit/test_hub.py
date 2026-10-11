"""Model references, file selection, favorites, offline cache, retention and GGUF metadata."""

import os
from pathlib import Path

import httpx2  # pyright: ignore[reportMissingImports]
import pytest

from llmbench.config.experiment import ModelSpec
from llmbench.hub.client import HubUnavailableError, HuggingFaceClient, RemoteFile, RepoListing
from llmbench.hub.gguf_meta import GGUFError, read_gguf
from llmbench.hub.refs import HfRef, RefError, parse_hf, quant_matches, shard_info
from llmbench.hub.store import (
    Favorites,
    ModelNotFoundError,
    ModelStore,
    combined_sha256,
    pick_files,
)
from tests.helpers import FakeHub, write_gguf

QWEN = "unsloth/Qwen3-1.7B-GGUF"


# -- references -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("hf://org/repo", HfRef("org/repo")),
        ("hf://org/repo:Q4_K_M", HfRef("org/repo", quant="Q4_K_M")),
        ("hf://org/repo@v1.2:q8_0", HfRef("org/repo", "v1.2", "q8_0")),
        (
            "hf://org/re.po@abc123/sub/dir/m.gguf",
            HfRef("org/re.po", "abc123", file="sub/dir/m.gguf"),
        ),
    ],
)
def test_parse_hf(source: str, expected: HfRef) -> None:
    assert parse_hf(source) == expected
    assert str(parse_hf(source)) == source


@pytest.mark.parametrize(
    "bad", ["hf://repo", "hf://org/repo/file.bin", "hf://org/repo:", "org/repo"]
)
def test_parse_hf_rejects(bad: str) -> None:
    with pytest.raises(RefError):
        parse_hf(bad)


def test_quant_matching_respects_boundaries() -> None:
    assert quant_matches("Qwen3-1.7B-Q2_K.gguf", "Q2_K")
    assert not quant_matches("Qwen3-1.7B-Q2_K_L.gguf", "Q2_K")
    assert quant_matches("phi-2/ggml-model-q4_0.gguf", "Q4_0")
    assert quant_matches("m-Q4_K_M-00001-of-00002.gguf", "q4_k_m")
    assert not quant_matches("m-IQ4_NL.gguf", "Q4_NL")


def test_shard_info() -> None:
    assert shard_info("dir/model-Q8_0-00002-of-00003.gguf") == ("model-Q8_0", 2, 3)
    assert shard_info("model.gguf") is None


# -- file selection (real listings captured from huggingface.co) ----------------------


@pytest.fixture
def listings(tmp_path: Path) -> dict[str, RepoListing]:
    return FakeHub.from_fixture(tmp_path).repos


def test_pick_by_quant(listings: dict[str, RepoListing]) -> None:
    files = pick_files(listings[QWEN], HfRef(QWEN, quant="Q2_K"))
    assert [f.name for f in files] == ["Qwen3-1.7B-Q2_K.gguf"]
    assert files[0].sha256 is not None


def test_default_quant_is_q4_k_m(listings: dict[str, RepoListing]) -> None:
    assert [f.name for f in pick_files(listings[QWEN], HfRef(QWEN))] == ["Qwen3-1.7B-Q4_K_M.gguf"]


def test_unknown_quant_lists_what_exists(listings: dict[str, RepoListing]) -> None:
    with pytest.raises(ModelNotFoundError, match=r"(?s)no GGUF matches quant 'Q9_X'.*Q8_0"):
        pick_files(listings[QWEN], HfRef(QWEN, quant="Q9_X"))


def test_ambiguous_quant_across_subfolders(listings: dict[str, RepoListing]) -> None:
    with pytest.raises(ModelNotFoundError, match="ambiguous"):
        pick_files(listings["ggml-org/models"], HfRef("ggml-org/models", quant="F16"))


def test_pick_by_path(listings: dict[str, RepoListing]) -> None:
    ref = HfRef("ggml-org/models", file="tinyllamas/stories260K.gguf")
    assert [f.name for f in pick_files(listings["ggml-org/models"], ref)] == [
        "tinyllamas/stories260K.gguf"
    ]


def listing(*names: str) -> RepoListing:
    return RepoListing("o/r", "c" * 40, [RemoteFile(n, 1, n) for n in names])


def test_split_model_takes_every_shard() -> None:
    lst = listing(
        "m-Q8_0-00002-of-00002.gguf",
        "m-Q8_0-00001-of-00002.gguf",
        "m-Q4_K_M.gguf",
        "mmproj-f16.gguf",
    )
    files = pick_files(lst, HfRef("o/r", quant="Q8_0"))
    assert [f.name for f in files] == ["m-Q8_0-00001-of-00002.gguf", "m-Q8_0-00002-of-00002.gguf"]
    by_path = pick_files(lst, HfRef("o/r", file="m-Q8_0-00002-of-00002.gguf"))
    assert len(by_path) == 2


def test_incomplete_split_is_rejected() -> None:
    with pytest.raises(ModelNotFoundError, match="incomplete"):
        pick_files(listing("m-Q8_0-00001-of-00003.gguf"), HfRef("o/r", quant="Q8_0"))


def test_single_model_repo_without_q4_k_m() -> None:
    assert [f.name for f in pick_files(listing("only-F16.gguf", "mmproj.gguf"), HfRef("o/r"))] == [
        "only-F16.gguf"
    ]


def test_combined_sha256() -> None:
    assert combined_sha256(["a" * 64]) == "a" * 64
    assert combined_sha256(["a", "b"]) != combined_sha256(["b", "a"])


# -- resolution, favorites, offline, retention ------------------------------------------


def spec(source: str, **kw: object) -> ModelSpec:
    return ModelSpec.model_validate({"id": "m", "source": source} | kw)


def test_download_then_cache_hit(hub: FakeHub) -> None:
    store = ModelStore()
    first = store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    second = store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    assert (first.origin, second.origin) == ("download", "cache")
    assert len(hub.downloads) == 1
    assert first.sha256 is not None
    assert first.commit == hub.repos[QWEN].commit


def test_validate_mode_never_downloads(hub: FakeHub) -> None:
    with pytest.raises(ModelNotFoundError, match="is not downloaded"):
        ModelStore(allow_download=False).resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    assert hub.downloads == []


def test_repo_not_found(hub: FakeHub) -> None:
    with pytest.raises(ModelNotFoundError, match="repository not found"):
        ModelStore().resolve(spec("hf://nobody/nothing:Q4_K_M"))


def test_offline_uses_cache_and_hashes_locally(hub: FakeHub) -> None:
    store = ModelStore()
    store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    hub.online = False
    r = store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    assert r.origin == "cache"
    assert r.sha256 is None  # no Hub metadata offline -> hashed by the executor
    with pytest.raises(ModelNotFoundError, match="not available offline"):
        store.resolve(spec(f"hf://{QWEN}:Q8_0"))


def test_favorite_moves_out_of_cache_and_resolves_offline(hub: FakeHub, tmp_path: Path) -> None:
    store = ModelStore()
    fav = store.pull(f"hf://{QWEN}:Q4_K_M", favorite=True)
    assert fav.origin == "favorite"
    assert fav.path == tmp_path / "models" / "unsloth--Qwen3-1.7B-GGUF" / "Qwen3-1.7B-Q4_K_M.gguf"
    assert hub.cached_files() == []
    entry = Favorites().entries()[0]
    assert entry.sha256 == fav.sha256
    hub.online = False
    again = store.resolve(spec(f"hf://{QWEN}:q4_k_m"))
    assert again.origin == "favorite"
    assert again.sha256 == fav.sha256
    assert store.unfavorite(f"hf://{QWEN}:Q4_K_M") == [fav.path]
    assert not fav.path.exists()
    assert Favorites().entries() == []


def test_release_keeps_revision_shared_with_other_quants(hub: FakeHub) -> None:
    store = ModelStore()
    q4 = store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    store.resolve(spec(f"hf://{QWEN}:Q8_0"))
    assert store.release(q4) == 0  # Q8_0 lives in the same revision
    assert len(hub.cached_files()) == 2


def test_release_and_prune(hub: FakeHub) -> None:
    store = ModelStore()
    q4 = store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    assert store.release(q4) > 0
    assert hub.cached_files() == []

    store.resolve(spec(f"hf://{QWEN}:Q4_K_M"))
    assert store.prune(older_than_days=30, repo=None, dry_run=False)[0] == []
    hub.age(45)
    targets, size = store.prune(older_than_days=30, repo=None, dry_run=True)
    assert len(targets) == 1
    assert size > 0
    assert len(hub.cached_files()) == 1  # dry run
    store.prune(older_than_days=30, repo=None, dry_run=False)
    assert hub.cached_files() == []


def test_local_paths_and_favorites_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_gguf(tmp_path / "models" / "some-org--repo" / "nested.gguf")
    r = ModelStore().resolve(spec("nested.gguf"))
    assert r.origin == "local"
    assert r.path.name == "nested.gguf"
    with pytest.raises(ModelNotFoundError, match="file not found"):
        ModelStore().resolve(spec("missing.gguf"))


def test_local_split_model(tmp_path: Path) -> None:
    for i in (1, 2):
        write_gguf(tmp_path / f"big-{i:05d}-of-00002.gguf")
    r = ModelStore().resolve(spec(str(tmp_path / "big-00001-of-00002.gguf")))
    assert len(r.shards) == 2
    (tmp_path / "big-00002-of-00002.gguf").unlink()
    with pytest.raises(ModelNotFoundError, match="missing shards"):
        ModelStore().resolve(spec(str(tmp_path / "big-00001-of-00002.gguf")))


# -- GGUF metadata -------------------------------------------------------------------------


def test_gguf_reader(tmp_path: Path) -> None:
    p = write_gguf(
        tmp_path / "m.gguf",
        {
            "general.architecture": "qwen3",
            "general.file_type": 15,
            "qwen3.context_length": 40960,
            "qwen3.block_count": 28,
            "tokenizer.ggml.tokens": [f"t{i}" for i in range(100)],
            "flag": True,
            "neg": -3,
            "ratio": 0.5,
        },
        [("a", (1024, 2048), 12), ("b", (2048,), 0)],
    )
    info = read_gguf(p)
    s = info.summary()
    assert (s["architecture"], s["file_type"], s["quant"]) == ("qwen3", "Q4_K_M", "Q4_K_M")
    assert (s["context_length"], s["block_count"]) == (40960, 28)
    assert info.n_params == 1024 * 2048 + 2048
    assert info.tensor_types == {"Q4_K": 1024 * 2048, "F32": 2048}
    assert info.metadata["tokenizer.ggml.tokens"] == "<array of 100>"
    assert (info.metadata["flag"], info.metadata["neg"], info.metadata["ratio"]) == (True, -3, 0.5)


def test_reading_a_gguf_does_not_count_as_using_it(tmp_path: Path) -> None:
    """`prune --older-than` reads the blob's atime: inspecting must not refresh it."""
    p = write_gguf(tmp_path / "m.gguf")
    old = 1_600_000_000_000_000_000
    os.utime(p, ns=(old, old))
    read_gguf(p)
    assert p.stat().st_atime_ns == old
    assert p.stat().st_mtime_ns == old


def test_gguf_quant_falls_back_to_dominant_tensor_type(tmp_path: Path) -> None:
    info = read_gguf(write_gguf(tmp_path / "m.gguf", tensors=[("a", (10, 10), 8), ("b", (5,), 0)]))
    assert info.file_type is None
    assert info.quant == "Q8_0"


def test_gguf_shards_are_summed(tmp_path: Path) -> None:
    a = write_gguf(tmp_path / "a.gguf", tensors=[("x", (4, 4), 0)])
    b = write_gguf(tmp_path / "b.gguf", {}, tensors=[("y", (4,), 0)])
    assert read_gguf([a, b]).n_params == 20


def test_gguf_rejects_garbage(tmp_path: Path) -> None:
    (tmp_path / "x.gguf").write_bytes(b"NOPE")
    with pytest.raises(GGUFError, match="bad magic"):
        read_gguf(tmp_path / "x.gguf")
    (tmp_path / "y.gguf").write_bytes(b"GGUF\x03\x00\x00\x00\x01")
    with pytest.raises(GGUFError, match="truncated"):
        read_gguf(tmp_path / "y.gguf")


def test_network_errors_mean_unavailable_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """huggingface_hub 2.x raises httpx2 errors (e.g. behind a blocking proxy)."""
    client = HuggingFaceClient()

    def boom(*_: object, **__: object) -> None:
        raise httpx2.ProxyError("403 Forbidden")

    monkeypatch.setattr(client.api, "model_info", boom)
    with pytest.raises(HubUnavailableError, match="403"):
        client.list_repo("o/r", None)


def test_explicit_path_may_name_an_mmproj() -> None:
    lst = listing("m-Q8_0.gguf", "mmproj-m-f16.gguf")
    assert [f.name for f in pick_files(lst, HfRef("o/r", file="mmproj-m-f16.gguf"))] == [
        "mmproj-m-f16.gguf"
    ]
    with pytest.raises(ModelNotFoundError):  # quant selection never picks a projector
        pick_files(lst, HfRef("o/r", quant="F16"))
