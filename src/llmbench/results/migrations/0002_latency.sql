-- Phase 3: llama-server latency runs.
ALTER TABLE runs ADD COLUMN suite TEXT;

-- One row per timed request (warmup requests are not stored).
CREATE TABLE requests (
    run_id       INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    request_id   TEXT NOT NULL,
    pass         INTEGER NOT NULL,
    ok           INTEGER NOT NULL,
    error        TEXT,
    ttft_ms      REAL,
    e2e_ms       REAL,
    prompt_n     INTEGER,
    prompt_ms    REAL,
    predicted_n  INTEGER,
    predicted_ms REAL,
    prompt_tps   REAL,
    gen_tps      REAL,
    cache_n      INTEGER,
    stop_type    TEXT,
    itl_ms       TEXT NOT NULL,  -- JSON list
    content      TEXT
);
CREATE INDEX requests_run ON requests (run_id);

-- Aggregates per run (name -> value): ttft_ms_p50, gen_tps_mean, ...
CREATE TABLE run_metrics (
    run_id INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    name   TEXT NOT NULL,
    value  REAL,
    PRIMARY KEY (run_id, name)
);
