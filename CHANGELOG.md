# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [Unreleased]

## [0.4.0] - 2026-10-08

Phase 3: llama-server and synthetic data.

### Added

- `phases.latency`: sends suites of requests to `llama-server` one at a time and measures client-side TTFT, inter-token latency and end-to-end latency, plus the server's own prompt and generation throughput. Supports warmup requests and `passes`. Every request runs with `cache_prompt: false` (ADR-0013).
- llama-server lifecycle: free local port on 127.0.0.1 only, health wait, `/props`, per-server log files, clean shutdown, and a remembered start failure per server.
- A `ServerPool` reuses one llama-server process across runs that share a server key (build, model, non-sampling config). Runs that only differ in sampling or suite never restart the server.
- Synthetic suites (`suites/*.yaml`) with prompts of exactly N tokens per model: windows of the model's own tokens, BOS included. `prompt_tokens` and `output_tokens` can be fixed, uniform or normal, with a seed and an optional corpus file. `ignore_eos` is on by default.
- Config sections `rope`, `speculative` (including a draft model resolved like any model), `server` and `sampling` (sent per request), covering the llama-server flags of commit `c811cb8`.
- `llb validate` checks llama-server flags and suites. `llb plan` reports latency runs, server starts and timed requests.
- Speculative decoding acceptance rate from `/metrics`.
- Database migration 0002 (`requests`, `run_metrics`, `runs.suite`) and a `latency.csv` export.
- Example suites `synthetic-512x128` and `synthetic-chat-mix`, the `deterministic` profile, and the `pi5-latency` experiment.
- A fake `llama-server` for integration tests and a flag contract test against the real server help.
- ADR-0012 (peak memory via `VmHWM`) and ADR-0013 (latency methodology).

### Changed

- Peak RSS now comes from `/proc/<pid>/status` `VmHWM`, read every 20 ms. `ru_maxrss` included the Python parent's memory from before `exec`, so a 13 MB llama-bench could show 313 MB (ADR-0012 supersedes ADR-0007).
- The bench fingerprint only covers bench-relevant config (`load`, `kv`, `gpu`, `env`, `extra_args.bench`), so a sampling axis no longer duplicates llama-bench runs. **`--resume` will not recognize bench runs recorded by earlier versions.**
- `server.parallel` defaults to 1 slot (llama.cpp's own default is auto/4).
- The engine is split into `engine/runtime.py` and `engine/phases/{bench,latency}.py`.

### Fixed

- `--help` parsing no longer treats flags marked "the argument has been removed" as supported, nor indented description lines that mention other flags.

## [0.3.0] - 2026-10-08

Phase 2: Hugging Face models.

### Added

- Model references `hf://org/repo[@revision][:QUANT]` and `hf://org/repo[@revision]/path/file.gguf`, with Q4_K_M as the default quant (same as llama.cpp).
- Quant matching respects name boundaries (`Q2_K` does not match `Q2_K_L`). Errors list the available files, report ambiguity across subfolders, and catch incomplete split models.
- Split (sharded) models, both from the Hub and as local files.
- Resolution order: local path → `models/` favorites → Hub (cache hit or download) → offline cache. Each model is resolved lazily, right before its first run.
- Exact provenance per run: repo, resolved commit and sha256 (from the Hub's LFS metadata, without hashing the file), plus GGUF metadata.
- `retention: delete_after_run` frees the model's cached revision after its last run.
- Favorites in `models/<org>--<repo>/`, indexed in `models/.llb.json`; they work offline and are never deleted automatically.
- `llb models pull | ls | inspect | favorite | unfavorite | prune`.
- Pure-Python GGUF header reader (architecture, parameter count, quant, context length, layers, heads, experts, tensor-type mix). It does not need numpy.
- CSV columns `model_quant`, `model_params` and `model_commit`.
- Example experiment `quantization.yaml`; `pi5-threads` and `kv-cache` now use `hf://` sources.
- ADR-0010 (HF cache deletion per revision through the official API) and ADR-0011 (GGUF reader).

### Changed

- `huggingface-hub` is now a core dependency. The `hf` extra is removed.
- `llb validate` resolves models without downloading them.

## [0.2.0] - 2026-10-08

Phase 1: llama-bench MVP.

### Added

- Declarative experiment YAML (`version: 1`) validated with pydantic. It rejects unknown keys and reports errors with the exact path and line number.
- `extends` with recursive deep merge, cycle detection and `${env:VAR:-default}` interpolation.
- llama.cpp configuration models covering every `llama-bench` flag of commit `c811cb8`, grouped in `load`, `kv` and `gpu`, plus per-tool `extra_args` and `env`.
- Matrix over dotted config paths: cartesian axes, `zip` groups, `exclude` and `include`. Every value is validated up front.
- Managed llama.cpp builds per commit, backend and CMake flags (`llb llama build`, `llb llama ls`), and external builds via `bin_dir`.
- Flag introspection from each binary's `--help` and `--version`. Unsupported flags fail validation instead of producing wrong numbers (`llb llama flags`).
- `llama-bench` runner with JSONL parsing, a timeout, OOM detection, and exact peak RSS and CPU time via `wait4`.
- Hardware telemetry (temperature from vcgencmd, sysfs or psutil; CPU frequency; RAM; process RSS), Raspberry Pi throttling decoding, and temperature-based cooldown with a stability window.
- Pre-flight checks for the CPU governor and current throttling.
- SQLite results store with versioned migrations, run fingerprints, `--resume`, telemetry series and a cache for model sha256 hashes.
- Median, CV and an `unstable` flag per test.
- `summary.csv` and `runs.jsonl` exports.
- CLI commands `llb validate`, `llb plan` and `llb run`.
- Example experiments `pi5-threads` and `kv-cache`, and the profile `_profiles/pi5.yaml`.
- Real llama.cpp fixtures (help, version, JSONL), a fake `llama-bench` for integration tests, and a flag contract test.
- Documentation: `ARCHITECTURE.md`, `SYSTEM_OVERVIEW.md`, `MODELS.md` and ADRs 0001–0009.

## [0.1.0] - 2026-10-08

Phase 0: project skeleton.

### Added

- uv package `local-llm-bench-lab` (module `llmbench`, CLI `llb`) with `--version` and `doctor`.
- ruff, basedpyright (strict), pytest, pre-commit, and GitHub Actions on x86_64 and arm64.
- Project proposal (`docs/PRD.md`) and plan (`docs/PLAN.md`).

[Unreleased]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OWNER/local-llm-bench-lab/releases/tag/v0.1.0
