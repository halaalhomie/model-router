"""Aggregate queries behind the dashboard.

Reads the `requests` table Phase 6 fills. Nothing here touches the
request path -- the dashboard is a reader of history, and a slow query
here can never slow down an answer.

Costs are summed in SQL, where NULL means "could not be priced" and
sum() ignores it. That is the correct behaviour and also the reason
unpriced rows are counted separately: a total that quietly dropped
unpriced requests would understate, which is the one error this
project's cost accounting exists to avoid.
"""

from pydantic import BaseModel


TOTALS_SQL = """
SELECT
    count(*)                                                  AS requests,
    coalesce(sum(answering_cost_usd), 0)                      AS answering_cost_usd,
    coalesce(sum(analyzer_cost_usd), 0)                       AS analyzer_cost_usd,
    coalesce(avg(latency_ms), 0)                              AS avg_latency_ms,
    coalesce(avg(confidence), 0)                              AS avg_confidence,
    count(*) FILTER (WHERE fallback_used)                     AS fallbacks,
    count(*) FILTER (WHERE answering_cost_usd IS NULL)        AS unpriced
FROM requests
"""

BY_MODEL_SQL = """
SELECT
    response_model,
    count(*)                                                       AS requests,
    coalesce(avg(latency_ms), 0)                                   AS avg_latency_ms,
    coalesce(
        percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms), 0
    )                                                              AS p95_latency_ms,
    coalesce(sum(total_tokens), 0)                                 AS tokens,
    coalesce(sum(answering_cost_usd), 0)                           AS cost_usd,
    count(*) FILTER (WHERE answering_cost_usd IS NULL)             AS unpriced
FROM requests
GROUP BY response_model
ORDER BY requests DESC, response_model
"""

BY_TASK_TYPE_SQL = """
SELECT task_type, count(*) AS requests
FROM requests
GROUP BY task_type
ORDER BY requests DESC, task_type
"""

RECENT_SQL = """
SELECT request_id, task_type, response_model, fallback_used,
       latency_ms,
       coalesce(answering_cost_usd, 0) + coalesce(analyzer_cost_usd, 0)
           AS total_cost_usd,
       created_at
FROM requests
ORDER BY created_at DESC
LIMIT %(limit)s
"""


class Totals(BaseModel):
    requests: int
    answering_cost_usd: float
    analyzer_cost_usd: float
    total_cost_usd: float
    avg_latency_ms: float
    avg_confidence: float
    fallbacks: int
    fallback_rate: float
    unpriced: int


class ModelRow(BaseModel):
    model: str
    requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    tokens: int
    cost_usd: float
    unpriced: int


class TaskTypeRow(BaseModel):
    task_type: str
    requests: int


class RecentRow(BaseModel):
    request_id: str
    task_type: str
    model: str
    fallback_used: bool
    latency_ms: float
    total_cost_usd: float
    created_at: str


class DashboardStats(BaseModel):
    totals: Totals
    by_model: list[ModelRow]
    by_task_type: list[TaskTypeRow]
    recent: list[RecentRow]


def read_totals(connection) -> Totals:
    with connection.cursor() as cursor:
        cursor.execute(TOTALS_SQL)
        row = cursor.fetchone()

    requests, answering, analyzer, latency, confidence, fallbacks, unpriced = row
    return Totals(
        requests=requests,
        answering_cost_usd=float(answering),
        analyzer_cost_usd=float(analyzer),
        total_cost_usd=float(answering) + float(analyzer),
        avg_latency_ms=float(latency),
        avg_confidence=float(confidence),
        fallbacks=fallbacks,
        fallback_rate=fallbacks / requests if requests else 0.0,
        unpriced=unpriced,
    )


def read_by_model(connection) -> list[ModelRow]:
    with connection.cursor() as cursor:
        cursor.execute(BY_MODEL_SQL)
        rows = cursor.fetchall()

    return [
        ModelRow(
            model=name,
            requests=requests,
            avg_latency_ms=float(avg_latency),
            p95_latency_ms=float(p95_latency),
            tokens=tokens,
            cost_usd=float(cost),
            unpriced=unpriced,
        )
        for name, requests, avg_latency, p95_latency, tokens, cost, unpriced
        in rows
    ]


def read_by_task_type(connection) -> list[TaskTypeRow]:
    with connection.cursor() as cursor:
        cursor.execute(BY_TASK_TYPE_SQL)
        rows = cursor.fetchall()

    return [
        TaskTypeRow(task_type=task_type, requests=requests)
        for task_type, requests in rows
    ]


def read_recent(connection, limit: int = 10) -> list[RecentRow]:
    with connection.cursor() as cursor:
        cursor.execute(RECENT_SQL, {"limit": limit})
        rows = cursor.fetchall()

    return [
        RecentRow(
            request_id=request_id,
            task_type=task_type,
            model=model,
            fallback_used=fallback_used,
            latency_ms=float(latency),
            total_cost_usd=float(cost),
            created_at=created_at.isoformat(),
        )
        for request_id, task_type, model, fallback_used, latency, cost,
        created_at in rows
    ]


def read_dashboard(connection, recent_limit: int = 10) -> DashboardStats:
    return DashboardStats(
        totals=read_totals(connection),
        by_model=read_by_model(connection),
        by_task_type=read_by_task_type(connection),
        recent=read_recent(connection, limit=recent_limit),
    )
