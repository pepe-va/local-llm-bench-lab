-- Spread of llama-bench repetitions: interquartile range and the 95 % bootstrap CI of avg_ts.
-- NULL with fewer than two samples. Store fills in rows saved before this migration.
ALTER TABLE measurements ADD COLUMN iqr_ts REAL;
ALTER TABLE measurements ADD COLUMN ci_low_ts REAL;
ALTER TABLE measurements ADD COLUMN ci_high_ts REAL;
