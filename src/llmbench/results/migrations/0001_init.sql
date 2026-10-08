-- One row per run attempt. A fingerprint can have several attempts (e.g. failed, then completed).
CREATE TABLE runs (
    id           INTEGER PRIMARY KEY,
    fingerprint  TEXT NOT NULL,
    experiment   TEXT NOT NULL,
    phase        TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('completed', 'failed', 'interrupted')),
    error        TEXT,
    build_id     TEXT NOT NULL,
    build_commit TEXT NOT NULL,
    backend      TEXT NOT NULL,
    model_id     TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    build        TEXT NOT NULL,  -- JSON: id, commit, version, backend, cmake, ref
    model        TEXT NOT NULL,  -- JSON: id, source, path, sha256, size
    variant      TEXT NOT NULL,  -- JSON: dotted path -> value (the matrix point)
    config       TEXT NOT NULL,  -- JSON: fully resolved LlamaConfig
    argv         TEXT NOT NULL,  -- JSON list
    ignored      TEXT NOT NULL,  -- JSON list of config paths the tool did not receive
    env          TEXT NOT NULL,  -- JSON machine snapshot
    telemetry    TEXT,           -- JSON summary (peaks, before/after)
    cooldown     TEXT,           -- JSON
    started_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL,
    duration_s   REAL NOT NULL
);
CREATE INDEX runs_fingerprint ON runs (fingerprint, status);
CREATE INDEX runs_experiment ON runs (experiment);

CREATE TABLE measurements (
    run_id     INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    test       TEXT NOT NULL CHECK (test IN ('pp', 'tg', 'pg')),
    label      TEXT NOT NULL,
    n_prompt   INTEGER NOT NULL,
    n_gen      INTEGER NOT NULL,
    n_depth    INTEGER NOT NULL,
    avg_ts     REAL NOT NULL,
    stddev_ts  REAL NOT NULL,
    median_ts  REAL NOT NULL,
    cv         REAL NOT NULL,
    unstable   INTEGER NOT NULL,
    avg_ns     INTEGER NOT NULL,
    stddev_ns  INTEGER NOT NULL,
    samples_ts TEXT NOT NULL,  -- JSON list
    raw        TEXT NOT NULL   -- JSON: the untouched llama-bench row
);
CREATE INDEX measurements_run ON measurements (run_id);

CREATE TABLE telemetry_samples (
    run_id       INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    t            REAL NOT NULL,
    temp_c       REAL,
    cpu_freq_mhz REAL,
    ram_used_mb  REAL NOT NULL,
    proc_rss_mb  REAL
);
CREATE INDEX telemetry_run ON telemetry_samples (run_id);

-- sha256 of large GGUF files is expensive on a Pi: cache it by (path, size, mtime).
CREATE TABLE model_hashes (
    path     TEXT PRIMARY KEY,
    size     INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256   TEXT NOT NULL
);
