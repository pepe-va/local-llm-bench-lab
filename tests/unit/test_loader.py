from pathlib import Path

import pytest

from llmbench.config.loader import ConfigError, deep_merge, load_experiment

MINIMAL = """
version: 1
experiment: {name: t}
llama_cpp: {builds: [{id: b, bin_dir: /opt/llama}]}
models: [{id: m, source: m.gguf}]
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_minimal_experiment(tmp_path: Path) -> None:
    exp = load_experiment(write(tmp_path, "e.yaml", MINIMAL))
    assert exp.experiment.name == "t"
    assert exp.phases.bench.repetitions == 5
    assert exp.matrix.expand() == [{}]


def test_extends_merges_mappings_and_replaces_lists(tmp_path: Path) -> None:
    write(
        tmp_path,
        "_profiles/base.yaml",
        "defaults: {load: {threads: 4, batch_size: 512}}\n"
        "phases: {bench: {n_prompt: [128, 512]}}\n",
    )
    p = write(
        tmp_path,
        "e.yaml",
        "extends: [_profiles/base.yaml]\n"
        + MINIMAL
        + "defaults: {load: {threads: 2}}\nphases: {bench: {n_prompt: [64]}}\n",
    )
    exp = load_experiment(p)
    assert exp.defaults.load.threads == 2
    assert exp.defaults.load.batch_size == 512
    assert exp.phases.bench.n_prompt == [64]


def test_extends_cycle_is_detected(tmp_path: Path) -> None:
    write(tmp_path, "a.yaml", "extends: b.yaml\n")
    write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_experiment(tmp_path / "a.yaml")


def test_env_interpolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REF", "b6500")
    text = MINIMAL.replace(
        "{id: b, bin_dir: /opt/llama}",
        "{id: b, ref: '${env:REF}'}, {id: c, ref: '${env:NOPE:-master}'}",
    )
    exp = load_experiment(write(tmp_path, "e.yaml", text))
    assert [b.ref for b in exp.llama_cpp.builds] == ["b6500", "master"]


def test_missing_env_without_default_fails(tmp_path: Path) -> None:
    text = MINIMAL.replace("source: m.gguf", "source: '${env:LLB_TEST_UNSET_VAR}'")
    with pytest.raises(ConfigError, match="LLB_TEST_UNSET_VAR is not set"):
        load_experiment(write(tmp_path, "e.yaml", text))


def test_error_has_path_and_line(tmp_path: Path) -> None:
    text = MINIMAL + "defaults:\n  load:\n    threads: -4\n"
    with pytest.raises(ConfigError) as e:
        load_experiment(write(tmp_path, "e.yaml", text))
    assert "defaults.load.threads (line 8)" in str(e.value)
    assert "greater than 0" in str(e.value)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"defaults\.load\.thread"):
        load_experiment(write(tmp_path, "e.yaml", MINIMAL + "defaults: {load: {thread: 4}}\n"))


def test_matrix_error_location(tmp_path: Path) -> None:
    text = MINIMAL + "matrix:\n  load.threads: [1, 0]\n"
    with pytest.raises(ConfigError, match=r"matrix\.load\.threads\[1\]"):
        load_experiment(write(tmp_path, "e.yaml", text))


def test_build_needs_exactly_one_source(tmp_path: Path) -> None:
    text = MINIMAL.replace("{id: b, bin_dir: /opt/llama}", "{id: b}")
    with pytest.raises(ConfigError, match="exactly one of"):
        load_experiment(write(tmp_path, "e.yaml", text))


def test_duplicate_model_ids(tmp_path: Path) -> None:
    text = MINIMAL.replace("[{id: m, source: m.gguf}]", "[{id: m, source: a}, {id: m, source: b}]")
    with pytest.raises(ConfigError, match="duplicate id 'm'"):
        load_experiment(write(tmp_path, "e.yaml", text))


def test_deep_merge() -> None:
    assert deep_merge({"a": {"b": 1, "c": [1]}}, {"a": {"c": [2]}}) == {"a": {"b": 1, "c": [2]}}


def test_repo_examples_are_valid() -> None:
    root = Path(__file__).parents[2] / "experiments"
    for p in root.glob("*.yaml"):
        load_experiment(p)
