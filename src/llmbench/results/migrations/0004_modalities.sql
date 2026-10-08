-- Phase 6: embeddings (pair similarity vs label) and rerank (ranking stored in eval_detail).
ALTER TABLE requests ADD COLUMN label REAL;
ALTER TABLE requests ADD COLUMN similarity REAL;
