import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmbench.config.llama import LlamaConfig
from llmbench.config.matrix import Matrix, apply_variant, variant_label


def m(**data: object) -> Matrix:
    return Matrix.model_validate(data)


def test_empty_matrix_is_one_default_variant() -> None:
    assert Matrix().expand() == [{}]


def test_axes_are_crossed() -> None:
    variants = m(**{"load.threads": [1, 2], "kv.cache_type_k": ["f16", "q8_0"]}).expand()
    assert len(variants) == 4
    assert {"load.threads": 2, "kv.cache_type_k": "q8_0"} in variants


def test_zip_advances_together() -> None:
    variants = m(zip=[{"kv.cache_type_k": ["f16", "q8_0"], "kv.cache_type_v": ["f16", "q8_0"]}])
    assert variants.expand() == [
        {"kv.cache_type_k": "f16", "kv.cache_type_v": "f16"},
        {"kv.cache_type_k": "q8_0", "kv.cache_type_v": "q8_0"},
    ]


def test_exclude_and_include() -> None:
    matrix = m(
        **{"load.threads": [1, 2]},
        zip=[{"kv.cache_type_k": ["f16", "q8_0"], "kv.cache_type_v": ["f16", "q8_0"]}],
        exclude=[{"load.threads": 1, "kv.cache_type_k": "q8_0"}],
        include=[{"load.threads": 4}],
    )
    variants = matrix.expand()
    assert len(variants) == 2 * 2 - 1 + 1
    assert {"load.threads": 4} in variants


def test_include_is_deduplicated() -> None:
    assert m(**{"load.threads": [1]}, include=[{"load.threads": 1}]).expand() == [
        {"load.threads": 1}
    ]


@given(
    a=st.lists(st.integers(1, 64), min_size=1, max_size=5, unique=True),
    b=st.lists(
        st.sampled_from(["f16", "q8_0", "q4_0", "bf16"]), min_size=1, max_size=4, unique=True
    ),
)
def test_count_is_product_of_axes(a: list[int], b: list[str]) -> None:
    variants = m(**{"load.threads": a, "kv.cache_type_k": b}).expand()
    assert len(variants) == math.prod([len(a), len(b)])
    assert len({variant_label(v) for v in variants}) == len(variants)


def test_zip_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="equal length"):
        m(zip=[{"kv.cache_type_k": ["f16"], "kv.cache_type_v": ["f16", "q8_0"]}])


def test_axis_must_be_non_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        m(**{"load.threads": list[int]()})


def test_validate_reports_exact_location() -> None:
    matrix = m(**{"load.threads": [2, -4]})
    with pytest.raises(ValueError, match=r"matrix\.load\.threads\[1\]: .*greater than 0"):
        matrix.validate_against(LlamaConfig())


def test_validate_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match=r"unknown config path 'load\.thread'"):
        m(**{"load.thread": [1]}).validate_against(LlamaConfig())


def test_env_keys_may_be_created() -> None:
    cfg = apply_variant(LlamaConfig(), {"env.GGML_SCHED": "1"})
    assert cfg.env == {"GGML_SCHED": "1"}


def test_apply_variant_keeps_defaults() -> None:
    base = LlamaConfig.model_validate({"load": {"batch_size": 512, "threads": 4}})
    cfg = apply_variant(base, {"load.threads": 2})
    assert cfg.load.threads == 2
    assert cfg.load.batch_size == 512
