"""Suite -> list[Request] for every source except `synthetic` (which needs a live tokenizer)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmbench.config.suite import Columns, Item, Suite
from llmbench.data.hf_datasets import load_hf_dataset, read_table
from llmbench.data.requests import Request


class SuiteError(Exception):
    pass


@dataclass(frozen=True)
class Materialized:
    requests: list[Request]
    provenance: dict[str, Any]
    """Where the data came from (e.g. dataset repo + resolved commit)."""


def materialize(suite: Suite) -> Materialized:
    src = suite.source
    needs_prompt = suite.endpoint in ("completion", "chat")
    if src.items is not None:
        rows = [_item_row(it) for it in src.items]
        provenance: dict[str, Any] = {"source": "items"}
    elif src.file is not None:
        base = src.file.path.parent
        rows = [
            _with_base(_column_row(r, suite.columns, needs_prompt=needs_prompt), base)
            for r in read_table(src.file.path)
        ]
        provenance = {"source": "file", "path": str(src.file.path)}
    elif src.hf_dataset is not None:
        ds = load_hf_dataset(src.hf_dataset)
        rows = [_column_row(r, suite.columns, needs_prompt=needs_prompt) for r in ds.rows]
        provenance = {
            "source": "hf_dataset",
            "repo": ds.repo,
            "config": ds.config,
            "split": ds.split,
            "commit": ds.commit,
            "rows": len(rows),
        }
    else:
        raise SuiteError(f"suite {suite.name!r}: synthetic suites are built with a tokenizer")
    return Materialized([_request(suite, i, row) for i, row in enumerate(rows)], provenance)


def _item_row(it: Item) -> dict[str, Any]:
    return it.model_dump() | {"_evaluator": it.evaluator}


def _column_row(row: dict[str, Any], cols: Columns, *, needs_prompt: bool) -> dict[str, Any]:
    def get(name: str | None) -> Any:
        if name is None or (name == cols.prompt and not needs_prompt):
            return None  # embeddings/rerank rows have no prompt
        if name not in row:
            raise SuiteError(f"column {name!r} not found; columns: {', '.join(row)}")
        return row[name]

    return row | {
        "prompt": str(get(cols.prompt) or ""),
        "image": _opt_str(get(cols.image)),
        "a": _opt_str(get(cols.a)),
        "b": _opt_str(get(cols.b)),
        "label": None if get(cols.label) is None else float(get(cols.label)),
        "query": _opt_str(get(cols.query)),
        "documents": get(cols.documents),
        "relevant": get(cols.relevant),
        "input": _opt_str(get(cols.input)),
        "reference": _opt_str(get(cols.reference)),
        "id": _opt_str(get(cols.id)),
        "category": _opt_str(get(cols.category)),
        "system": _opt_str(get(cols.system)),
    }


def _request(suite: Suite, index: int, row: dict[str, Any]) -> Request:
    prompt = str(row["prompt"])
    if suite.template is not None:
        try:
            prompt = suite.template.format_map(row)
        except (KeyError, IndexError, ValueError) as e:
            raise SuiteError(f"suite {suite.name!r}: template error at item {index}: {e}") from None
    return Request(
        id=f"{suite.name}/{row.get('id') or f'{index:05d}'}",
        prompt=prompt,
        n_predict=int(row.get("max_tokens") or suite.max_tokens),
        system=row.get("system") or suite.system,
        input=row.get("input"),
        reference=row.get("reference"),
        category=row.get("category"),
        evaluator=row.get("_evaluator") or suite.evaluator,
        image=row.get("image"),
        a=row.get("a"),
        b=row.get("b"),
        label=row.get("label"),
        query=row.get("query"),
        documents=tuple(str(d) for d in row.get("documents") or ()),
        relevant=tuple(int(i) for i in row.get("relevant") or ()),
    )


def _with_base(row: dict[str, Any], base: Path) -> dict[str, Any]:
    image = row.get("image")
    if isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
        row["image"] = str((base / image).resolve())
    return row


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)
