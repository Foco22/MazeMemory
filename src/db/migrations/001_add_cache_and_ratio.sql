-- Add cache token tracking to experiments and agent_runs,
-- and add optimal_steps / optimality_ratio to agent_runs.

ALTER TABLE experiments
  ADD COLUMN total_cache_hit_tokens  INTEGER,
  ADD COLUMN total_cache_miss_tokens INTEGER;

ALTER TABLE agent_runs
  ADD COLUMN optimal_steps     INTEGER,
  ADD COLUMN optimality_ratio  NUMERIC(8, 4),
  ADD COLUMN cache_hit_tokens  INTEGER,
  ADD COLUMN cache_miss_tokens INTEGER;