"""Writes completed requests to PostgreSQL.

Persistence lives behind Kafka, not in the request path, for the same
reason publishing does: the caller must never wait on a database. A
consumer reads the topic at its own pace and writes here; if this database
is down, requests keep being answered and the events wait in the topic
until it comes back.

Idempotency is not optional here
--------------------------------
app/consumer.py commits offsets *after* processing, which is at-least-once
delivery: a crash between the write and the commit means the same event
arrives again on restart. A plain INSERT would then raise a duplicate-key
error, or worse, silently double-count in a table without a primary key.

So every write is an upsert keyed on request_id. Seeing an event twice
produces exactly the same row, which is what makes at-least-once delivery
safe to build on. This is the constraint identified back in Phase 4, now
honoured.

ON CONFLICT DO UPDATE rather than DO NOTHING is deliberate. Re-delivering
an unchanged event writes identical values either way, but replay is a
real operational tool -- this project has already changed how cost is
calculated once, and rows written before that change are wrong. DO UPDATE
means resetting a consumer group to the start of the topic repairs those
rows; DO NOTHING would skip them forever.
"""

import logging
from pathlib import Path

import psycopg

from app.config import Settings
from app.schemas import PipelineResult, TokenUsage


logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

UPSERT_REQUEST = """
INSERT INTO requests (
    request_id, request_text,
    task_type, difficulty, reasoning_required, context_size,
    output_type, confidence,
    decision_model, decision_reason,
    response_model, response_text, fallback_used, original_model,
    prompt_tokens, output_tokens, thinking_tokens, total_tokens,
    analyzer_model, analyzer_prompt_tokens, analyzer_output_tokens,
    analyzer_thinking_tokens, analyzer_total_tokens,
    answering_cost_usd, analyzer_cost_usd,
    latency_ms
) VALUES (
    %(request_id)s, %(request_text)s,
    %(task_type)s, %(difficulty)s, %(reasoning_required)s,
    %(context_size)s, %(output_type)s, %(confidence)s,
    %(decision_model)s, %(decision_reason)s,
    %(response_model)s, %(response_text)s, %(fallback_used)s,
    %(original_model)s,
    %(prompt_tokens)s, %(output_tokens)s, %(thinking_tokens)s,
    %(total_tokens)s,
    %(analyzer_model)s, %(analyzer_prompt_tokens)s,
    %(analyzer_output_tokens)s, %(analyzer_thinking_tokens)s,
    %(analyzer_total_tokens)s,
    %(answering_cost_usd)s, %(analyzer_cost_usd)s,
    %(latency_ms)s
)
ON CONFLICT (request_id) DO UPDATE SET
    request_text             = EXCLUDED.request_text,
    task_type                = EXCLUDED.task_type,
    difficulty               = EXCLUDED.difficulty,
    reasoning_required       = EXCLUDED.reasoning_required,
    context_size             = EXCLUDED.context_size,
    output_type              = EXCLUDED.output_type,
    confidence               = EXCLUDED.confidence,
    decision_model           = EXCLUDED.decision_model,
    decision_reason          = EXCLUDED.decision_reason,
    response_model           = EXCLUDED.response_model,
    response_text            = EXCLUDED.response_text,
    fallback_used            = EXCLUDED.fallback_used,
    original_model           = EXCLUDED.original_model,
    prompt_tokens            = EXCLUDED.prompt_tokens,
    output_tokens            = EXCLUDED.output_tokens,
    thinking_tokens          = EXCLUDED.thinking_tokens,
    total_tokens             = EXCLUDED.total_tokens,
    analyzer_model           = EXCLUDED.analyzer_model,
    analyzer_prompt_tokens   = EXCLUDED.analyzer_prompt_tokens,
    analyzer_output_tokens   = EXCLUDED.analyzer_output_tokens,
    analyzer_thinking_tokens = EXCLUDED.analyzer_thinking_tokens,
    analyzer_total_tokens    = EXCLUDED.analyzer_total_tokens,
    answering_cost_usd       = EXCLUDED.answering_cost_usd,
    analyzer_cost_usd        = EXCLUDED.analyzer_cost_usd,
    latency_ms               = EXCLUDED.latency_ms
"""


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(settings.database_url)


def ensure_schema(connection: psycopg.Connection) -> None:
    """Apply the DDL. Safe to call on every startup -- see schema.sql."""
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.commit()


def usage_columns(usage: TokenUsage | None, prefix: str = "") -> dict:
    """Flatten a TokenUsage into columns, or NULLs when there is none."""
    if usage is None:
        return {
            f"{prefix}prompt_tokens": None,
            f"{prefix}output_tokens": None,
            f"{prefix}thinking_tokens": None,
            f"{prefix}total_tokens": None,
        }
    return {
        f"{prefix}prompt_tokens": usage.prompt_tokens,
        f"{prefix}output_tokens": usage.output_tokens,
        f"{prefix}thinking_tokens": usage.thinking_tokens,
        f"{prefix}total_tokens": usage.total_tokens,
    }


def row_for(result: PipelineResult) -> dict:
    """The parameters one PipelineResult contributes to the table."""
    row: dict = {
        "request_id": result.request_id,
        "request_text": result.request_text,
        "task_type": result.profile.task_type,
        "difficulty": result.profile.difficulty,
        "reasoning_required": result.profile.reasoning_required,
        "context_size": result.profile.context_size,
        "output_type": result.profile.output_type,
        "confidence": result.profile.confidence,
        "decision_model": result.decision.model_name,
        "decision_reason": result.decision.reason,
        "response_model": result.response.model_name,
        "response_text": result.response.text,
        "fallback_used": result.response.fallback_used,
        "original_model": result.response.original_model,
        "answering_cost_usd": result.response.estimated_cost_usd,
        "analyzer_model": result.analyzer_model,
        "analyzer_cost_usd": result.analyzer_cost_usd,
        "latency_ms": result.latency_ms,
    }
    row.update(usage_columns(result.response.usage))
    row.update(usage_columns(result.analyzer_usage, prefix="analyzer_"))
    return row


def save_result(
    connection: psycopg.Connection, result: PipelineResult
) -> None:
    """Upsert one completed request. Safe to call twice with the same
    event -- see this module's docstring on at-least-once delivery."""
    with connection.cursor() as cursor:
        cursor.execute(UPSERT_REQUEST, row_for(result))
    connection.commit()
