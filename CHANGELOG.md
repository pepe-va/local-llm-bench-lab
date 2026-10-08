# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Commits follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).

## [Unreleased]

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

[Unreleased]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/OWNER/local-llm-bench-lab/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/OWNER/local-llm-bench-lab/releases/tag/v0.1.0
