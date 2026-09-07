-- One row per completed request.
--
-- This is a fact table, not a normalized domain model. The profile, the
-- routing decision, and the response are all facts about one finished
-- request; none is ever updated on its own, and none is shared with
-- another row. Splitting them into four tables would buy nothing and cost
-- a four-way join on every analytics query.
--
-- Applied with CREATE TABLE IF NOT EXISTS on startup, which is enough
-- while the schema only ever grows. The moment a column has to change
-- type or be dropped without losing rows, this needs a real migration
-- tool (Alembic) instead -- IF NOT EXISTS silently does nothing to a
-- table that already exists with the wrong shape.

CREATE TABLE IF NOT EXISTS requests (
    request_id   TEXT PRIMARY KEY,
    request_text TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- What the classifier decided the work was.
    task_type          TEXT             NOT NULL,
    difficulty         TEXT             NOT NULL,
    reasoning_required TEXT             NOT NULL,
    context_size       TEXT             NOT NULL,
    output_type        TEXT             NOT NULL,
    confidence         DOUBLE PRECISION NOT NULL,

    -- Who the router chose, and why.
    decision_model  TEXT NOT NULL,
    decision_reason TEXT NOT NULL,

    -- Who actually answered. Differs from decision_model when fallback
    -- fired, which is why original_model is kept alongside.
    response_model TEXT    NOT NULL,
    response_text  TEXT,
    fallback_used  BOOLEAN NOT NULL DEFAULT FALSE,
    original_model TEXT,

    -- Answering usage. Nullable because a provider does not always report
    -- it, and an absent count is not a zero count.
    prompt_tokens   INTEGER,
    output_tokens   INTEGER,
    thinking_tokens INTEGER,
    total_tokens    INTEGER,

    -- Classification usage: the overhead routing pays that a
    -- single-model baseline never does.
    analyzer_model           TEXT,
    analyzer_prompt_tokens   INTEGER,
    analyzer_output_tokens   INTEGER,
    analyzer_thinking_tokens INTEGER,
    analyzer_total_tokens    INTEGER,

    -- NUMERIC, not float: these are sub-cent values summed over many
    -- rows, and binary floating point accumulates error doing exactly
    -- that. NULL means "could not be priced", never "free" -- the same
    -- rule app/pricing.py follows, carried into the schema.
    answering_cost_usd NUMERIC(12, 8),
    analyzer_cost_usd  NUMERIC(12, 8),

    latency_ms DOUBLE PRECISION NOT NULL
);

-- Time-range queries ("what happened in the last hour") and per-model
-- aggregation are the two things this table exists to answer.
CREATE INDEX IF NOT EXISTS requests_created_at_idx
    ON requests (created_at DESC);

CREATE INDEX IF NOT EXISTS requests_response_model_idx
    ON requests (response_model);
