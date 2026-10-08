-- Phase 7: concurrency level of serving runs, and LLM-judge scores (kept apart, PRD §18).
ALTER TABLE runs ADD COLUMN concurrency INTEGER;

CREATE TABLE judge_scores (
    run_id      INTEGER NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    request_idx INTEGER NOT NULL,
    request_id  TEXT NOT NULL,
    criterion   TEXT NOT NULL,
    score       REAL,              -- 1..5
    explanation TEXT,
    judge       TEXT NOT NULL,     -- JSON: judge model source, sha256, build commit
    judged_at   TEXT NOT NULL,
    PRIMARY KEY (run_id, request_idx, criterion)
);

-- What the judge needs for each pending response.
ALTER TABLE requests ADD COLUMN reference TEXT;
ALTER TABLE requests ADD COLUMN judge_spec TEXT;  -- JSON: {criteria: [...], rubric: str | null}
