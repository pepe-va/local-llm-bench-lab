-- Phase 4: quality runs (evaluated responses).
ALTER TABLE requests ADD COLUMN reasoning TEXT;
ALTER TABLE requests ADD COLUMN prompt_text TEXT;
ALTER TABLE requests ADD COLUMN category TEXT;
ALTER TABLE requests ADD COLUMN score REAL;
ALTER TABLE requests ADD COLUMN passed INTEGER;
ALTER TABLE requests ADD COLUMN eval_detail TEXT;
ALTER TABLE requests ADD COLUMN judge_pending INTEGER NOT NULL DEFAULT 0;
ALTER TABLE runs ADD COLUMN suite_provenance TEXT;  -- JSON: dataset repo/commit, file path...
