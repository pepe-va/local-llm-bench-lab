# AGENTS.md

Instructions for AI coding agents (and humans) working on **local-llm-bench-lab**: a reproducible benchmarking lab for local LLMs on llama.cpp. Package `llmbench`, CLI `llb`. Read this before changing code, and keep it up to date when the rules change.

## Orientation

| Read | For |
|---|---|
| [docs/SYSTEM_OVERVIEW.md](docs/SYSTEM_OVERVIEW.md) | What llb does, concepts, commands, guarantees |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, modules, dependency rules, testing strategy |
| [docs/MODELS.md](docs/MODELS.md) | Config/runtime/result classes, YAML → flag tables, SQLite schema, fingerprints |
| [docs/adr/](docs/adr/README.md) | Why things are the way they are. Check before contradicting a decision. |
| [CHANGELOG.md](CHANGELOG.md) | What changed in each version |
| [docs/PLAN.md](docs/PLAN.md), [docs/PRD.md](docs/PRD.md) | Original plan and proposal (historical) |

## Commands

```bash
uv sync --extra report --extra serve --extra hf-datasets   # dev environment (same extras as CI)
uv run ruff check && uv run ruff format --check            # lint + format
uv run basedpyright                                        # types (strict, src + tests)
uv run pytest --cov                                        # tests (no network)
uv run llb --help
```

All four checks must pass before every commit. CI runs them on x86_64 and arm64.

## Code rules

- **Python 3.12, uv, ruff, basedpyright strict.** No new type errors; use `cast` at untyped library boundaries, never blanket `# type: ignore`.
- **Layering**: `config/` only describes and validates; `engine/` never prints (it reports through the `Events` protocol); `results/` does not depend on the backend; only `cli/` knows rich and typer. See ARCHITECTURE.md.
- **llama.cpp flag spellings live only in `llamacpp/argv.py`.** A new config field needs:
  1. the field, in `config/llama/*.py`;
  2. its mapping in `argv.py`, or a mention in `ignored`;
  3. coverage in the "full config" contract tests (`test_every_*_flag_exists_in_the_real_binary`), updating the syrupy snapshot (`uv run pytest --snapshot-update`) and reviewing the diff;
  4. a row in the MODELS.md flag tables.
- **Never guess llama.cpp behaviour.** Check a real binary (`--help`, an actual request) and store new outputs as fixtures under `tests/fixtures/llamacpp/<commit>/`.
- **Tests never touch the network or the real `$LLB_HOME`/`HF_HOME`.** The autouse fixtures in `tests/conftest.py` set `HF_HUB_OFFLINE=1` and temporary homes. Use the fakes: `tests/fixtures/bin/fake-llama-bench`, `fake-llama-server`, `fake-lm_eval`, and `FakeHub` in `tests/helpers.py`.
- **Peak memory comes from `VmHWM`**, never `ru_maxrss` (ADR-0012). The server binds to `127.0.0.1` only (PRD §37).
- **Hugging Face cache deletion goes only through `huggingface_hub`'s revision API** (ADR-0010). Never delete blobs by hand.

## Reproducibility (do not break it silently)

- **Fingerprints** (`engine/planner.py`, `engine/phases/lm_eval.py`) identify *what is measured*. Any change to what enters a fingerprint changes run identities, so `--resume` stops matching old runs. That is a **breaking change**: say so in the commit footer and the CHANGELOG.
- **SQLite schema**: add a new `src/llmbench/results/migrations/NNNN_description.sql` (next number). **Never edit a migration that has been released**, meaning one included in any tag. Migrations are applied in order through `PRAGMA user_version`.
- **The experiment YAML is `version: 1`.** Adding optional fields with defaults is compatible. Renaming or removing fields, or changing their meaning, is breaking. After 1.0.0 that requires `version: 2` and a major release.
- **Results store provenance** (build commit, model sha256, dataset commit, environment snapshot). Do not remove any of it.

## Versioning: SemVer 2.0.0 ([ADR-0009](docs/adr/0009-conventional-commits-semver.md))

The public API covered by SemVer is the **CLI**, the **experiment and suite YAML schema**, and the **SQLite schema and migrations**.

While in `0.y.z` (current):

| Change | Bump |
|---|---|
| New feature, new phase, or a breaking change | **minor** (`0.9.x` → `0.10.0`) |
| Bug fix, new example experiment or suite, docs-only release | **patch** (`0.9.0` → `0.9.1`) |

From `1.0.0`: a breaking change is a **major** release, a feature is **minor**, and a fix is **patch**. 1.0.0 is cut after the full validation on a Raspberry Pi 5.

## Commits: Conventional Commits 1.0

- **Format**: `type(scope)!: summary`. Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`. Usual scopes: `config`, `llamacpp`, `hub`, `server`, `quality`, `report`, `serve`, `evaluation`, `telemetry`, `experiments`.
- **Breaking changes**: add `!` to the header and a `BREAKING CHANGE: …` footer that explains what stops working and why.
- **Keep commits split by type**: a fix found while building a feature gets its own `fix:` commit before the `feat:`. Documentation goes in its own `docs:` commit.
- **Staging**: stage explicit paths. Never `git add -A` at the repo root, because local tooling may leave files there.
- **Pushing**: do not push, and do not rewrite history once it has been pushed, unless the maintainer asks.

## Release procedure

Do this at the end of a feature, phase or fix batch, once every check passes:

1. **Bump** `version` in `pyproject.toml`, then run `uv lock`.
2. **CHANGELOG.md** ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/)):
   - move `[Unreleased]` items into a new `## [X.Y.Z] - YYYY-MM-DD` section, under `Added`, `Changed`, `Fixed` or `Removed`;
   - mark breaking changes in bold;
   - update the compare links at the bottom.
3. **README.md**: update the `Status:` line.
4. **Commit** in this order:
   - the `feat:`/`fix:` commits with the code, tests, examples and `uv.lock`;
   - `docs: …` with the documentation updates;
   - `chore(release): X.Y.Z` with the CHANGELOG and README.
5. **Tag** it, annotated: `git tag -a vX.Y.Z -m "vX.Y.Z: <one line>"`.

## Documentation to update with every change

| If you change… | Update |
|---|---|
| Behaviour, commands, concepts | `docs/SYSTEM_OVERVIEW.md` (including the "Estado: vX.Y.Z" line) |
| Modules, layers, dependencies | `docs/ARCHITECTURE.md` (and "Diferencias respecto a PLAN.md" if you deviate) |
| Config fields, classes, schema, fingerprints, exports | `docs/MODELS.md` |
| A decision with trade-offs | A new ADR in `docs/adr/NNNN-title.md` from `template.md`, plus a row in `docs/adr/README.md`. Accepted ADRs are never rewritten: supersede them with a new one and update the old one's status line. |
| User-facing usage | `README.md` |

**Languages**:
- **English**: code, comments, `README.md`, `CHANGELOG.md`, commit messages and this file.
- **Spanish**: design docs in `docs/` (overview, architecture, models, ADRs, plan).

## Local paths

- The repository directory is currently `local-llm-bech-lab` (sic). Do not create a sibling `local-llm-bench-lab/` by typing the "correct" name; double-check absolute paths.
- Heavy data never goes into git:
  - builds, sources and results live under `$LLB_HOME` (default `~/.local/share/llmbench`);
  - models live in the Hugging Face cache, or in `models/` for favorites (gitignored).
