# local-llm-bench-lab

Reproducible benchmarking lab for local LLMs on [llama.cpp](https://github.com/ggml-org/llama.cpp), starting on a Raspberry Pi 5.

- **Declarative experiments**: one YAML describes models, the llama.cpp configuration (every flag), the parameter matrix, suites and gates.
- **Hugging Face integration**: `hf://org/repo@rev:Q4_K_M` models and HF datasets as evaluation suites, with pinned revisions.
- **Managed llama.cpp builds** by commit, so the llama.cpp version can be a matrix axis.
- **Hardware telemetry**: temperature, throttling, RAM, CPU frequency, and temperature-based cooldown.
- **Reproducible results** in SQLite, with fingerprints for `--resume`, plus HTML, Markdown and PNG reports and pass/fail gates.

> Status: Phase 0 (skeleton). See [docs/PLAN.md](docs/PLAN.md) (and the original proposal in [docs/PRD.md](docs/PRD.md)) for the design and roadmap.

## Setup

```bash
uv sync              # creates .venv with dev tools
uv run llb --help
uv run llb doctor
```

Optional extras: `hf`, `hf-datasets`, `report`, `parquet`, `nvidia`:

```bash
uv sync --extra hf --extra report
```

## Development

```bash
uv run ruff check
uv run ruff format
uv run basedpyright
uv run pytest
uv run pre-commit install   # run the checks on every commit
```
