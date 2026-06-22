-- Groups all runs that belong to the same execution of run_experiments().
-- Allows filtering 10 runs launched together by a single batch_id UUID.

ALTER TABLE experiments
  ADD COLUMN batch_id TEXT;

CREATE INDEX idx_experiments_batch_id ON experiments (batch_id);