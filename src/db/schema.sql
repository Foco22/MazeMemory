-- Run this once in the Supabase SQL editor to create all tables.

CREATE TABLE experiments (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario                TEXT        NOT NULL,
    maze_id                 INTEGER     NOT NULL,
    maze_seed               INTEGER     NOT NULL,
    provider                TEXT        NOT NULL,
    model                   TEXT        NOT NULL,
    model_version           TEXT        NOT NULL,
    run_number              INTEGER     NOT NULL,
    lookahead               INTEGER     NOT NULL,
    started_at              TIMESTAMPTZ NOT NULL,
    completed_at            TIMESTAMPTZ NOT NULL,
    total_prompt_tokens     INTEGER     NOT NULL,
    total_completion_tokens INTEGER     NOT NULL,
    total_tokens            INTEGER     NOT NULL,
    total_cache_hit_tokens  INTEGER,
    total_cache_miss_tokens INTEGER,
    observer_prompt_tokens     INTEGER,
    observer_completion_tokens INTEGER,
    cost_prompt_usd            NUMERIC(12, 6),
    cost_completion_usd        NUMERIC(12, 6),
    cost_agents_usd            NUMERIC(12, 6),
    cost_observer_usd          NUMERIC(12, 6),
    cost_total_usd             NUMERIC(12, 6),
    duration_seconds           NUMERIC(10, 3)
);

CREATE TABLE agent_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID    NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    agent_id          TEXT    NOT NULL,
    steps             INTEGER NOT NULL,
    optimal_steps     INTEGER,
    optimality_ratio  NUMERIC(8, 4),
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    cache_hit_tokens  INTEGER,
    cache_miss_tokens INTEGER,
    reached_exit      BOOLEAN NOT NULL,
    duration_seconds  NUMERIC(10, 3)
);

CREATE TABLE trajectories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id   UUID    NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    agent_id        TEXT    NOT NULL,
    x               INTEGER NOT NULL,
    y               INTEGER NOT NULL,
    timestep        INTEGER NOT NULL
);

-- One row per tool call — captures every decision the agent made
CREATE TABLE agent_actions (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id    UUID    NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    agent_id         TEXT    NOT NULL,
    step             INTEGER NOT NULL,   -- sequential action number within the agent's run
    llm_text         TEXT,               -- LLM reasoning text before the tool call (may be null)
    tool_name        TEXT    NOT NULL,
    tool_args        JSONB,
    tool_result      JSONB   NOT NULL,
    prompt_tokens    INTEGER,            -- tokens used in this LLM call (null for non-first tool calls in same turn)
    completion_tokens INTEGER           -- tokens generated in this LLM call (null for non-first tool calls in same turn)
);