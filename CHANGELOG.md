# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [Unreleased]

### Added

- `hardware.model_retention` (`keep` by default, or `delete_after_run`) sets the retention of every model that does not set its own `retention`, so the machine profile decides whether downloaded models stay in the Hugging Face cache.
- Release workflow (`.github/workflows/release.yml`). Pushing a `vX.Y.Z` tag runs every CI check on the tagged commit. If they pass, it publishes the GitHub Release, with the matching CHANGELOG section as notes and the wheel and sdist attached. It refuses a tag that does not match the `pyproject.toml` version or has no CHANGELOG section (ADR-0021).
- A pre-push hook runs the tests. `uv run pre-commit install` sets it up together with the pre-commit hook.

### Changed

- The `pi5` profile sets `model_retention: delete_after_run`: every Pi experiment now removes each downloaded model from the HF cache after its last run, not only the first three models of `quantization.yaml`. The `pc` profile sets `keep`. A model can still opt out with `retention: keep`, and favorites in `models/` are never deleted.
- CI now runs the pre-commit hooks (file hygiene, lockfile, ruff, basedpyright) as well as the tests on x86_64 and arm64. Until now it skipped the file-hygiene hooks.
- The pre-commit hooks run ruff and uv through `uv run`, with the versions locked in `uv.lock`. Before, they used older pinned copies (ruff 0.12, uv 0.8). Real tool outputs under `tests/fixtures/` and the vendored htmx are excluded from the whitespace hooks, so they stay byte-exact.
- The README is now a user guide: a system diagram, a quick start, the example experiments, the commands and the limitations.

## [0.11.0] - 2026-10-09

Serving runs could ask llama-server for more context than a slot has. llb now reports that before measuring, and fixes the example experiment that ran into it.

### Added

- `llb validate`, `llb plan` and `llb run` warn when a synthetic suite has requests whose prompt plus output exceed the context of one llama-server slot (`ctx_size / parallel`, or the whole `ctx_size` with `kv.unified`). Synthetic shapes are seeded, so this is known before anything runs. Suites of items, files and datasets are not checked.

### Fixed

- `llb validate` ignored the serving phase. When serving (or lm_eval) was the only llama-server phase, its suites were not loaded and llama-server flags were not checked against the build.
- A second `llb run` in the same output directory (for example with `--resume`) overwrote `logs/server-NNN.log` from earlier invocations. Numbering now continues from the last log. The judge's log is numbered too (`logs/judge-server-NNN.log`).
- `experiments/serving.yaml` split 8192 tokens across 8 slots, leaving 1024 per slot, while 3 of its 32 requests need up to 1169. It now uses `ctx_size: 12288` (1536 per slot). Its runs get new fingerprints, so `--resume` does not match earlier results of this experiment.

## [0.10.0] - 2026-10-08

Completes the plan's statistics: every llama-bench test now carries its interquartile range and a 95% confidence interval.

### Added

- IQR and a 95% percentile-bootstrap confidence interval of the mean (`avg_ts`) for every llama-bench test. The bootstrap uses a fixed seed (10,000 resamples), so the same samples always give the same interval. Both are empty for a single repetition (ADR-0020).
- The interval, median and IQR appear in `llb run` output, the HTML and Markdown reports, the dashboard and `summary.csv` (`ci_low_ts`, `ci_high_ts`, `iqr_ts`). Whiskers on the llama-bench chart now show the 95% CI instead of ±1 standard deviation.
- Gate metrics `tps_ci_low:<test>` and `tps_ci_high:<test>`, for conservative thresholds such as "the lower bound of tg128 is at least 8 t/s".
- Migration `0006_measurement_spread.sql`. When an existing results database is upgraded, the new columns are filled in from the stored samples, so old runs get the interval too.

## [0.9.1] - 2026-10-08

First full run on a PC with a real build (`llb llama build --ref b11507`) and a real model (Qwen3-0.6B Q4_K_M vs Q8_0): 20/20 runs across every phase.

### Fixed

- A `contains` evaluator in `any` mode that passed still reported the missing values and a partial score. It now scores 1.0 with no failure detail.

### Added

- `_profiles/pc.yaml` (desktop or WSL: llama.cpp pinned to `b11507`, no Pi-specific checks) and `pc-smoke.yaml`, which runs every phase in minutes with Qwen3-0.6B Q4_K_M vs Q8_0.

## [0.9.0] - 2026-10-08

Phase 8: dashboard. Every phase of the plan is now implemented.

### Added

- `llb serve`: read-only web dashboard (FastAPI + Jinja2 + HTMX, extra `serve`) over one or more results directories.
  - Experiment list.
  - Per-experiment page with the same charts and tables as `llb report`, plus gates when given the experiment YAML.
  - Live run list refreshed every 3 s while `llb run` writes.
  - Run detail with measurements, metrics, a telemetry chart, requests, judge scores and the full config, argv and environment.
  - JSON API (`/api/...`, OpenAPI at `/api/docs`).
- htmx is bundled and plotly.js is served from the package, so the dashboard works without Internet access. It binds to 127.0.0.1 by default and warns otherwise.
- ADR-0019.

### Changed

- CI installs the `serve` extra too.

## [0.8.0] - 2026-10-08

Phase 7: serving, lm-evaluation-harness and LLM judge.

### Added

- `phases.serving`: closed-loop concurrent load, one run per `concurrency` level. Reports requests/s, aggregate prompt and generation tokens/s, TTFT, ITL and end-to-end latency (p50/p95) and the error rate. `server.parallel` is raised to `max(concurrency)` when lower. Writes `serving.csv`.
- `phases.lm_eval`: runs lm-evaluation-harness (`local-chat-completions` or `local-completions`) against each configuration's llama-server and flattens its results into `lm_eval:<task>.<metric>` metrics. Optional extra `lm-eval`.
- `phases.judge` and `llb judge [--force]`: an LLM judge scores responses with `llm_judge` evaluator parts (1-5 per criterion, JSON constrained by a schema), stored apart in `judge_scores` with the judge's identity. Adds `judge.<criterion>_mean` and `judge.overall_mean` metrics.
- Serving and lm-eval sections in reports (table plus a throughput-vs-concurrency chart). Judge scores are labeled as subjective.
- Example `serving.yaml`, and a judge (and optional lm-eval) in `quality-es.yaml`.
- Database migration 0005. ADR-0018.

### Changed

- CI installs only the `report` and `hf-datasets` extras (lm-eval is heavy and is tested with a fake executable).

## [0.7.0] - 2026-10-08

Phase 6: modalities.

### Added

- `embeddings` suites: items or columns `a`, `b`, `label`, plus a `pooling` option. Reports the cosine similarity per pair and Spearman/Pearson correlation with the labels per run.
- `rerank` suites: `query`, `documents`, `relevant`, plus `k`. Reports nDCG@k, MRR, recall@k and top-1 accuracy.
- Vision: chat items with an `image` (local files are inlined as data URLs, and their content hash enters the request digest), and `models[].mmproj` resolved like any model and passed as `-mm`.
- The server mode (generate, embeddings:<pooling>, rerank) is part of the server key and adds `--embeddings`, `--pooling` or `--reranking`.
- `models[].suites` restricts which suites a model runs.
- Example suites `embed-sts-es`, `rerank-es` and `vision-basic` (with images), and the `modalities` experiment.
- Retrieval metrics in reports (spearman, nDCG, MRR columns). Database migration 0004.
- ADR-0017.

### Fixed

- An explicit `hf://repo/mmproj-*.gguf` path can now be resolved; quant-based selection still never picks a projector.

## [0.6.0] - 2026-10-08

Phase 5: reports and gates.

### Added

- Per-configuration analysis view (build x model x variant) across every phase. Bench results are shared with the variants they are identical for.
- `gates:` (metric, op, value, agg) and `select:` in experiments. Metric names include `tps:<test>`, run metrics optionally restricted to one suite with `@<suite>`, `peak_rss_mb`, `temp_peak_c` and `throttled` (ADR-0016).
- `llb guide`: applies the gates, picks the winner and writes `guidance.md` with a ready `llama-server` command (sampling as server flags, bound to 127.0.0.1) and a systemd unit. Exits 1 when nothing passes.
- `llb report`: self-contained HTML (embedded Plotly charts, filterable and sortable tables, light/dark), Markdown and PNG (matplotlib), from an experiment, a results directory or a SQLite file.
- `llb run` writes the reports listed in `output.report` (default `md`) and, when gates are defined, `guidance.md`, exiting 1 if no configuration passes.
- Example gates in `pi5-latency.yaml`.

### Fixed

- Offline mode never triggered with huggingface_hub 2.x: network errors (`httpx2.ProxyError`, `ConnectError`...) are not `OSError`. They now fall back to the local cache instead of crashing. Dataset download failures are reported as suite errors.

### Changed

- CI installs every extra (`uv sync --locked --all-extras`). The `report` extra pins jinja2, plotly and matplotlib.

## [0.5.0] - 2026-10-08

Phase 4: quality.

### Added

- `phases.quality`: every response is scored by its evaluator. Latency metrics are recorded too, and quality runs share llama-server processes with latency runs.
- Suite sources `items` (inline), `file` (jsonl, json, csv, parquet) and `hf_dataset` (read from the Hub's `refs/convert/parquet` branch with a pinned commit, `limit` and `shuffle_seed`; needs the `hf-datasets` extra). Suites also gain `columns` mapping, a `template`, a default `system` prompt and `max_tokens` (ADR-0014).
- `chat` endpoint (`/v1/chat/completions` streaming, with server `timings`; reasoning text kept separately).
- Deterministic evaluators `exact`, `contains`, `not_contains`, `regex`, `regex_extract` (numeric or exact, with an optional reference pattern), `length` (absolute or ratio to the input or reference) and `json_schema`; composites `all`, `any` and `weighted`. `llm_judge` parts never affect objective scores and are marked pending for the judge pass (ADR-0015).
- Per-request prompt, response, reasoning, category, score, pass and failure detail. Per-run `score_mean`, `pass_rate`, per-category scores, and words and chars per second. New `quality.csv`.
- Dataset provenance (repo, config, split, commit, rows) stored with every run. The digest of the materialized requests enters the fingerprint.
- Suites `es-writing` (11 controlled Spanish writing tasks: correction, rewriting, clarity, summary, reduction, style, constraints, no invented facts) and `hf-gsm8k`, plus the `quality-es` experiment.
- Database migration 0003.

### Changed

- `phases/latency.py` becomes `phases/suites.py`, shared by the latency and quality phases. `LatencyPhase` and `QualityPhase` derive from `SuitePhase`.
- The `hf-datasets` extra now installs `pyarrow` instead of `datasets`. `jsonschema` is a core dependency.

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

[Unreleased]: /compare/v0.11.0...HEAD
[0.11.0]: /compare/v0.10.0...v0.11.0
[0.10.0]: /compare/v0.9.1...v0.10.0
[0.9.1]: /compare/v0.9.0...v0.9.1
[0.9.0]: /compare/v0.8.0...v0.9.0
[0.8.0]: /compare/v0.7.0...v0.8.0
[0.7.0]: /compare/v0.6.0...v0.7.0
[0.6.0]: /compare/v0.5.0...v0.6.0
[0.5.0]: /compare/v0.4.0...v0.5.0
[0.4.0]: /compare/v0.3.0...v0.4.0
[0.3.0]: /compare/v0.2.0...v0.3.0
[0.2.0]: /compare/v0.1.0...v0.2.0
[0.1.0]: /releases/tag/v0.1.0
