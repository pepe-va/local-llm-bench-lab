"""SQLite store: the source of truth for results, `--resume` and reports."""

import hashlib
import json
import sqlite3
from dataclasses import asdict
from importlib import resources
from pathlib import Path
from typing import Any

from llmbench.results.models import RunRecord
from llmbench.telemetry.sampler import Sample

_HASH_CHUNK = 8 * 1024 * 1024


def migrations() -> list[tuple[int, str]]:
    files = resources.files("llmbench.results") / "migrations"
    out: list[tuple[int, str]] = []
    for f in files.iterdir():
        if f.name.endswith(".sql"):
            out.append((int(f.name.split("_", 1)[0]), f.read_text(encoding="utf-8")))
    return sorted(out)


class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.db.close()

    def _migrate(self) -> None:
        current: int = self.db.execute("PRAGMA user_version").fetchone()[0]
        for version, sql in migrations():
            if version > current:
                with self.db:
                    self.db.executescript(sql)
                    self.db.execute(f"PRAGMA user_version = {version}")

    @property
    def schema_version(self) -> int:
        return self.db.execute("PRAGMA user_version").fetchone()[0]

    # -- runs --------------------------------------------------------------

    def is_completed(self, fingerprint: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM runs WHERE fingerprint = ? AND status = 'completed' LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def save_run(self, run: RunRecord, samples: list[Sample] | None = None) -> int:
        j = json.dumps
        with self.db:
            cur = self.db.execute(
                """INSERT INTO runs (fingerprint, experiment, phase, status, error,
                    build_id, build_commit, backend, model_id, model_sha256,
                    build, model, variant, config, argv, ignored, env, telemetry, cooldown,
                    started_at, finished_at, duration_s, suite, suite_provenance, concurrency)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run.fingerprint,
                    run.experiment,
                    run.phase,
                    run.status,
                    run.error,
                    run.build["id"],
                    run.build["commit"],
                    run.build["backend"],
                    run.model["id"],
                    run.model["sha256"],
                    j(run.build),
                    j(run.model),
                    j(run.variant),
                    j(run.config),
                    j(run.argv),
                    j(run.ignored),
                    j(run.env),
                    j(run.telemetry) if run.telemetry is not None else None,
                    j(run.cooldown) if run.cooldown is not None else None,
                    run.started_at,
                    run.finished_at,
                    run.duration_s,
                    run.suite,
                    j(run.suite_provenance) if run.suite_provenance is not None else None,
                    run.concurrency,
                ),
            )
            run_id = cur.lastrowid
            assert run_id is not None
            self.db.executemany(
                """INSERT INTO measurements (run_id, test, label, n_prompt, n_gen, n_depth,
                    avg_ts, stddev_ts, median_ts, cv, unstable, avg_ns, stddev_ns,
                    samples_ts, raw)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id,
                        m.test,
                        m.label,
                        m.n_prompt,
                        m.n_gen,
                        m.n_depth,
                        m.avg_ts,
                        m.stddev_ts,
                        m.median_ts,
                        m.cv,
                        int(m.unstable),
                        m.avg_ns,
                        m.stddev_ns,
                        j(m.samples_ts),
                        j(m.raw),
                    )
                    for m in run.measurements
                ],
            )
            self.db.executemany(
                """INSERT INTO requests (run_id, idx, request_id, pass, ok, error, ttft_ms,
                    e2e_ms, prompt_n, prompt_ms, predicted_n, predicted_ms, prompt_tps, gen_tps,
                    cache_n, stop_type, itl_ms, content, reasoning, prompt_text, category, score,
                    passed, eval_detail, judge_pending, label, similarity, reference, judge_spec)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id,
                        i,
                        r.id,
                        int(r.id.rsplit("#", 1)[1]) if "#" in r.id else 0,
                        int(r.ok),
                        r.error,
                        r.ttft_ms,
                        r.e2e_ms,
                        r.prompt_n,
                        r.prompt_ms,
                        r.predicted_n,
                        r.predicted_ms,
                        r.prompt_tps,
                        r.gen_tps,
                        r.cache_n,
                        r.stop_type,
                        j(r.itl_ms),
                        r.content,
                        r.reasoning or None,
                        r.prompt_text,
                        r.category,
                        r.score,
                        None if r.passed is None else int(r.passed),
                        r.eval_detail,
                        int(r.judge_pending),
                        r.label,
                        r.similarity,
                        r.reference,
                        r.judge_spec,
                    )
                    for i, r in enumerate(run.requests)
                ],
            )
            self.db.executemany(
                "INSERT INTO run_metrics (run_id, name, value) VALUES (?,?,?)",
                [(run_id, k, v) for k, v in run.metrics.items()],
            )
            self.db.executemany(
                """INSERT INTO telemetry_samples
                    (run_id, t, temp_c, cpu_freq_mhz, ram_used_mb, proc_rss_mb)
                VALUES (?,?,?,?,?,?)""",
                [
                    (run_id, s.t, s.temp_c, s.cpu_freq_mhz, s.ram_used_mb, s.proc_rss_mb)
                    for s in samples or []
                ],
            )
        return run_id

    def summary_rows(self, experiment: str) -> list[dict[str, Any]]:
        """Bench runs: latest attempt per fingerprint, one row per measurement."""
        rows = self.db.execute(
            """WITH latest AS (
                   SELECT * FROM runs r
                   WHERE experiment = ? AND phase = 'bench' AND id = (
                       SELECT MAX(id) FROM runs WHERE fingerprint = r.fingerprint)
               )
               SELECT l.*, m.test, m.label, m.n_prompt, m.n_gen, m.n_depth, m.avg_ts,
                      m.stddev_ts, m.median_ts, m.cv, m.unstable
               FROM latest l LEFT JOIN measurements m ON m.run_id = l.id
               ORDER BY l.id, m.rowid""",
            (experiment,),
        ).fetchall()
        return [dict(r) for r in rows]

    def latency_rows(self, experiment: str, phase: str = "latency") -> list[dict[str, Any]]:
        """llama-server runs of a phase: latest attempt per fingerprint, metrics as columns."""
        runs = self.db.execute(
            """SELECT * FROM runs r
               WHERE experiment = ? AND phase = ? AND id = (
                   SELECT MAX(id) FROM runs WHERE fingerprint = r.fingerprint)
               ORDER BY id""",
            (experiment, phase),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in runs:
            row = dict(r)
            for name, value in self.db.execute(
                "SELECT name, value FROM run_metrics WHERE run_id = ?", (r["id"],)
            ):
                row[f"m.{name}"] = value
            out.append(row)
        return out

    def experiments(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT experiment FROM runs ORDER BY 1")]

    def latest_runs(self, experiment: str) -> list[dict[str, Any]]:
        """Latest attempt per fingerprint, every phase, with metrics and measurements decoded."""
        runs = [
            dict(r)
            for r in self.db.execute(
                """SELECT * FROM runs r
                   WHERE experiment = ? AND id = (
                       SELECT MAX(id) FROM runs WHERE fingerprint = r.fingerprint)
                   ORDER BY id""",
                (experiment,),
            )
        ]
        for r in runs:
            for key in ("build", "model", "variant", "config", "argv", "telemetry"):
                r[key] = json.loads(r[key]) if r[key] else None
            r["metrics"] = dict(
                self.db.execute("SELECT name, value FROM run_metrics WHERE run_id = ?", (r["id"],))
            )
            r["measurements"] = [
                dict(m)
                for m in self.db.execute(
                    "SELECT label, avg_ts, stddev_ts, median_ts, cv, unstable FROM measurements "
                    "WHERE run_id = ?",
                    (r["id"],),
                )
            ]
        return runs

    def telemetry(self, run_id: int) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT t, temp_c, cpu_freq_mhz, ram_used_mb, proc_rss_mb FROM telemetry_samples "
                "WHERE run_id = ? ORDER BY t",
                (run_id,),
            )
        ]

    def requests_of(self, run_id: int) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT * FROM requests WHERE run_id = ? ORDER BY idx", (run_id,)
            )
        ]

    # -- LLM judge ----------------------------------------------------------

    def pending_judgements(self, experiment: str) -> list[dict[str, Any]]:
        """Responses of the latest runs that still await the judge."""
        return [
            dict(r)
            for r in self.db.execute(
                """SELECT q.run_id, q.idx, q.request_id, q.prompt_text, q.content, q.reference,
                          q.judge_spec
                   FROM requests q JOIN runs r ON r.id = q.run_id
                   WHERE r.experiment = ? AND q.judge_pending = 1 AND q.ok = 1
                     AND r.id = (SELECT MAX(id) FROM runs WHERE fingerprint = r.fingerprint)
                   ORDER BY q.run_id, q.idx""",
                (experiment,),
            )
        ]

    def reset_judgements(self, experiment: str) -> int:
        """Forget judge scores so every judgeable response is judged again."""
        with self.db:
            self.db.execute(
                "DELETE FROM judge_scores WHERE run_id IN "
                "(SELECT id FROM runs WHERE experiment = ?)",
                (experiment,),
            )
            cur = self.db.execute(
                """UPDATE requests SET judge_pending = 1 WHERE judge_spec IS NOT NULL
                   AND run_id IN (SELECT id FROM runs WHERE experiment = ?)""",
                (experiment,),
            )
        return cur.rowcount

    def save_judgement(
        self,
        *,
        run_id: int,
        idx: int,
        request_id: str,
        scores: dict[str, tuple[float | None, str]],
        judge: dict[str, Any],
        judged_at: str,
    ) -> None:
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO judge_scores VALUES (?,?,?,?,?,?,?,?)",
                [
                    (run_id, idx, request_id, c, s, expl, json.dumps(judge), judged_at)
                    for c, (s, expl) in scores.items()
                ],
            )
            self.db.execute(
                "UPDATE requests SET judge_pending = 0 WHERE run_id = ? AND idx = ?", (run_id, idx)
            )

    def refresh_judge_metrics(self, run_id: int) -> dict[str, float]:
        """Recompute `judge.<criterion>_mean` and `judge.overall_mean` (1-5) of a run."""
        rows = self.db.execute(
            "SELECT criterion, AVG(score) FROM judge_scores WHERE run_id = ? AND score IS NOT NULL "
            "GROUP BY criterion",
            (run_id,),
        ).fetchall()
        metrics = {f"judge.{c}_mean": float(v) for c, v in rows}
        if metrics:
            metrics["judge.overall_mean"] = sum(metrics.values()) / len(metrics)
        pending = self.db.execute(
            "SELECT COUNT(*) FROM requests WHERE run_id = ? AND judge_pending = 1", (run_id,)
        ).fetchone()[0]
        metrics["judge_pending"] = float(pending)
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO run_metrics (run_id, name, value) VALUES (?,?,?)",
                [(run_id, k, v) for k, v in metrics.items()],
            )
        return metrics

    # -- model hashes ------------------------------------------------------

    def remember_sha256(self, path: Path, digest: str) -> None:
        """Record a hash known from elsewhere (e.g. the Hub's LFS metadata)."""
        st = path.stat()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO model_hashes VALUES (?,?,?,?)",
                (str(path.resolve()), st.st_size, st.st_mtime_ns, digest),
            )

    def model_sha256(self, path: Path) -> str:
        st = path.stat()
        key = str(path.resolve())
        row = self.db.execute(
            "SELECT size, mtime_ns, sha256 FROM model_hashes WHERE path = ?", (key,)
        ).fetchone()
        if row and row["size"] == st.st_size and row["mtime_ns"] == st.st_mtime_ns:
            return row["sha256"]
        digest = sha256_file(path)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO model_hashes VALUES (?,?,?,?)",
                (key, st.st_size, st.st_mtime_ns, digest),
            )
        return digest


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["Store", "asdict", "sha256_file"]
