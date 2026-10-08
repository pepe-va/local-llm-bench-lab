"""Hugging Face datasets via their parquet conversion, without the `datasets` package.

Every dataset on the Hub has a `refs/convert/parquet` branch with `{config}/{split}/NNNN.parquet`
(large splits may live under `{config}/partial-{split}/`). Reading it needs only
`huggingface_hub` (download, pinned to a commit) and `pyarrow` (extra `hf-datasets`).
"""

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download  # pyright: ignore[reportUnknownVariableType]

from llmbench.config.suite import HfDataset

CONVERT_REF = "refs/convert/parquet"


class DatasetError(Exception):
    pass


@dataclass(frozen=True)
class LoadedDataset:
    repo: str
    config: str
    split: str
    commit: str
    rows: list[dict[str, Any]]


def load_hf_dataset(spec: HfDataset) -> LoadedDataset:
    api = HfApi()
    try:
        commit = spec.revision or _convert_commit(api, spec.repo)
        files = api.list_repo_files(spec.repo, repo_type="dataset", revision=commit)
    except Exception as e:
        raise DatasetError(f"{spec.repo}: cannot list the parquet conversion ({e})") from None
    wanted = [
        f
        for f in sorted(files)
        if f.endswith(".parquet")
        and (
            f.startswith(f"{spec.config}/{spec.split}/")
            or f.startswith(f"{spec.config}/partial-{spec.split}/")
        )
    ]
    if not wanted:
        available = sorted({"/".join(f.split("/")[:2]) for f in files if f.endswith(".parquet")})
        raise DatasetError(
            f"{spec.repo}: no parquet for config={spec.config!r} split={spec.split!r}; "
            f"available: {', '.join(available) or 'none'}"
        )
    rows: list[dict[str, Any]] = []
    for f in wanted:
        try:
            local = hf_hub_download(spec.repo, f, repo_type="dataset", revision=commit)
        except Exception as e:  # network/auth: report, never crash the suite
            raise DatasetError(f"{spec.repo}: cannot download {f} ({e})") from None
        rows += read_table(Path(str(local)))
    if spec.shuffle_seed is not None:
        random.Random(spec.shuffle_seed).shuffle(rows)
    if spec.limit is not None:
        rows = rows[: spec.limit]
    return LoadedDataset(spec.repo, spec.config, spec.split, commit, rows)


def _convert_commit(api: HfApi, repo: str) -> str:
    refs = api.list_repo_refs(repo, repo_type="dataset")
    for ref in refs.converts:
        if ref.ref == CONVERT_REF:
            return ref.target_commit
    raise DatasetError(f"{repo}: the Hub has no parquet conversion for this dataset")


def read_table(path: Path) -> list[dict[str, Any]]:
    """Rows of a `.jsonl`, `.json`, `.csv` or `.parquet` file."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise DatasetError(f"{path}: expected a JSON list of objects")
        return list(data)  # pyright: ignore[reportUnknownArgumentType]
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        except ImportError:
            raise DatasetError(
                "reading parquet needs pyarrow: `uv sync --extra hf-datasets`"
            ) from None
        return list(pq.read_table(path).to_pylist())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    raise DatasetError(f"{path}: unsupported format {suffix!r} (jsonl, json, csv, parquet)")
