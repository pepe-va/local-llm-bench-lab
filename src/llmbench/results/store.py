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


def _migrations() -> list[tuple[int, str]]:
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
        for version, sql in _migrations():
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
                    started_at, finished_at, duration_s)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
        """Latest attempt per fingerprint, one row per measurement (or one for a failed run)."""
        rows = self.db.execute(
            """WITH latest AS (
                   SELECT * FROM runs r
                   WHERE experiment = ? AND id = (
                       SELECT MAX(id) FROM runs WHERE fingerprint = r.fingerprint)
               )
               SELECT l.*, m.test, m.label, m.n_prompt, m.n_gen, m.n_depth, m.avg_ts,
                      m.stddev_ts, m.median_ts, m.cv, m.unstable
               FROM latest l LEFT JOIN measurements m ON m.run_id = l.id
               ORDER BY l.id, m.rowid""",
            (experiment,),
        ).fetchall()
        return [dict(r) for r in rows]

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
