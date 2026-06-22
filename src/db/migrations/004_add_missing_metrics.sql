-- Add observer cache token tracking to experiments.
ALTER TABLE experiments
  ADD COLUMN observer_cache_hit_tokens  INTEGER,
  ADD COLUMN observer_cache_miss_tokens INTEGER;

-- Add redundant computation metrics to agent_runs.
ALTER TABLE agent_runs
  ADD COLUMN redundant_cells     INTEGER,
  ADD COLUMN total_cells_visited INTEGER,
  ADD COLUMN redundancy_ratio    NUMERIC(8, 4);