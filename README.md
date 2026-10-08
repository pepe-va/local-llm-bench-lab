# local-llm-bench-lab

Reproducible benchmarking lab for local LLMs on [llama.cpp](https://github.com/ggml-org/llama.cpp), starting on a Raspberry Pi 5.

- **Declarative experiments**: one YAML describes models, the llama.cpp configuration (every flag), the parameter matrix, suites and gates.
- **Hugging Face models**: `hf://org/repo[@rev]:Q4_K_M` with pinned provenance (commit + sha256), a `models/` folder for favorites, offline use, and per-model retention to save disk.
- **Managed llama.cpp builds** by commit, with every flag validated against the binary it runs on.
- **Two measurement phases**: `bench` (llama-bench, engine throughput) and `latency` (llama-server: TTFT, inter-token latency, end-to-end, with synthetic prompts of exact token counts). Every llama.cpp knob is configurable, including RoPE/YaRN, speculative decoding, server options and sampling.
- **Hardware telemetry**: temperature, throttling, exact peak RAM, CPU frequency, and temperature-based cooldown.
- **Reproducible results** in SQLite, with fingerprints for `--resume`, plus CSV and JSONL exports.
- **Quality suites**: own prompts (a Spanish writing suite is included), files or Hugging Face datasets, scored by deterministic evaluators (exact, contains, regex, numeric extraction, length, JSON schema, composites).
- **Gates and guidance**: pass/fail thresholds per configuration, the winner, and a ready-to-run `llama-server` command; reports in HTML (offline charts), Markdown and PNG.
- **Modalities**: embeddings (Spearman against labels), reranking (nDCG, MRR) and vision (models with an `mmproj`).
- **Serving and standard benchmarks**: concurrent load (req/s, aggregate tokens/s, p95), lm-evaluation-harness against the same server, and an optional LLM judge whose scores are kept apart from objective ones.
- **Dashboard**: `llb serve`, a read-only web UI with live progress, charts, gates and per-run detail.

> Status: **v0.9.0**: every phase of the [plan](docs/PLAN.md) is implemented. 1.0.0 will follow validation on a Raspberry Pi 5. See [CHANGELOG](CHANGELOG.md).

## Quick start

```bash
uv sync
uv run llb llama build --ref b6500 --backend cpu     # or point a build at an existing bin_dir
uv run llb models pull hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M --favorite   # optional: run downloads on demand
uv run llb models inspect hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M
uv run llb validate experiments/pi5-threads.yaml
uv run llb plan experiments/pi5-threads.yaml
uv run llb run experiments/pi5-threads.yaml          # Ctrl-C safe; continue with --resume
uv run llb run experiments/pi5-latency.yaml          # TTFT / ITL through llama-server
uv run llb guide experiments/pi5-latency.yaml        # gates -> winner -> deploy command
uv run llb report experiments/pi5-latency.yaml -f html -f png
uv run llb serve                                     # dashboard on http://127.0.0.1:8765
```

Results go to `~/.local/share/llmbench/runs/<experiment>/` (`benchmark.sqlite`, `summary.csv`, `latency.csv`, `runs.jsonl`, server logs) unless `output.directory` says otherwise.

A minimal experiment:

```yaml
version: 1
experiment: { name: pi5-threads }
llama_cpp:
  builds: [{ id: pinned, ref: b6500, backend: cpu }]
models:
  - { id: qwen3-1.7b-q4km, source: "hf://unsloth/Qwen3-1.7B-GGUF:Q4_K_M" }   # or a local .gguf
defaults:
  load: { batch_size: 512, flash_attn: auto }
matrix:
  load.threads: [1, 2, 3, 4]
phases:
  bench: { n_prompt: [512], n_gen: [128], repetitions: 5 }
```

## Documentation

| Document | Contents |
|---|---|
| [System overview](docs/SYSTEM_OVERVIEW.md) | What it is, concepts, flow, commands, guarantees |
| [Architecture](docs/ARCHITECTURE.md) | Layers, modules, design rules, testing strategy |
| [Models](docs/MODELS.md) | Config, runtime and result classes; YAML → flag mapping; SQLite schema |
| [ADRs](docs/adr/README.md) | Architecture decisions and their reasons |
| [Plan](docs/PLAN.md) / [PRD](docs/PRD.md) | Roadmap and original proposal |
| [AGENTS.md](AGENTS.md) | Contribution rules: checks, reproducibility, SemVer, commits, releases |

## Development

```bash
uv run ruff check && uv run ruff format --check
uv run basedpyright
uv run pytest --cov
uv run pre-commit install
```

Optional extras: `report` (HTML/PNG), `serve` (dashboard), `hf-datasets` (parquet datasets), `lm-eval`, `parquet`, `nvidia` (`uv sync --extra report --extra serve`).

Commits follow [Conventional Commits](https://www.conventionalcommits.org/) and releases follow [SemVer](https://semver.org/) ([ADR-0009](docs/adr/0009-conventional-commits-semver.md)).
