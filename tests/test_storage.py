import unittest

from app.storage import (
    SCHEMA_PATH,
    UPSERT_REQUEST,
    row_for,
    save_result,
    usage_columns,
)
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
    TokenUsage,
)


def make_result(
    request_id: str = "req-abc",
    *,
    fallback_used: bool = False,
    answering_cost: float | None = 0.01,
    analyzer_cost: float | None = 0.0002,
    usage: TokenUsage | None = None,
    analyzer_usage: TokenUsage | None = None,
) -> PipelineResult:
    return PipelineResult(
        request_id=request_id,
        request_text="Explain sharding.",
        profile=TaskProfile(
            task_type="reasoning",
            difficulty="high",
            reasoning_required="high",
            context_size="small",
            output_type="explanation",
            confidence=0.8,
        ),
        decision=RoutingDecision(
            model_name="demo-reasoning-model", reason="needs reasoning"
        ),
        response=ModelResponse(
            model_name="demo-fast-model" if fallback_used else "demo-reasoning-model",
            text="Sharding splits data.",
            usage=usage,
            fallback_used=fallback_used,
            original_model="demo-reasoning-model" if fallback_used else None,
            estimated_cost_usd=answering_cost,
        ),
        latency_ms=1234.5,
        analyzer_model="demo-analyzer-model",
        analyzer_usage=analyzer_usage,
        analyzer_cost_usd=analyzer_cost,
    )


class FakeCursor:
    def __init__(self, recorder: list) -> None:
        self.recorder = recorder

    def execute(self, sql: str, params: object = None) -> None:
        self.recorder.append((sql, params))

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeConnection:
    """Stands in for psycopg.Connection: records SQL, counts commits."""

    def __init__(self) -> None:
        self.executed: list = []
        self.commits = 0

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.executed)

    def commit(self) -> None:
        self.commits += 1


class UsageColumnsTests(unittest.TestCase):
    def test_missing_usage_becomes_nulls_not_zeroes(self) -> None:
        """An absent token count is unknown, not zero -- the same rule
        pricing follows, carried into the table."""
        columns = usage_columns(None)

        self.assertEqual(set(columns.values()), {None})

    def test_flattens_a_usage_into_columns(self) -> None:
        usage = TokenUsage(
            prompt_tokens=11,
            output_tokens=841,
            thinking_tokens=621,
            total_tokens=1473,
        )

        columns = usage_columns(usage)

        self.assertEqual(columns["prompt_tokens"], 11)
        self.assertEqual(columns["thinking_tokens"], 621)
        self.assertEqual(columns["total_tokens"], 1473)

    def test_prefix_separates_analyzer_columns(self) -> None:
        usage = TokenUsage(prompt_tokens=99, output_tokens=61, total_tokens=160)

        columns = usage_columns(usage, prefix="analyzer_")

        self.assertEqual(columns["analyzer_prompt_tokens"], 99)
        self.assertNotIn("prompt_tokens", columns)


class RowForTests(unittest.TestCase):
    def test_maps_every_column_the_insert_names(self) -> None:
        """The row and the statement must agree, or psycopg raises at
        execute time on a live connection rather than here."""
        row = row_for(
            make_result(
                usage=TokenUsage(
                    prompt_tokens=1, output_tokens=2, total_tokens=3
                ),
                analyzer_usage=TokenUsage(
                    prompt_tokens=4, output_tokens=5, total_tokens=9
                ),
            )
        )

        named = set()
        for fragment in UPSERT_REQUEST.split("%(")[1:]:
            named.add(fragment.split(")s")[0])

        self.assertEqual(named, set(row))

    def test_records_who_actually_answered_and_who_was_chosen(self) -> None:
        row = row_for(make_result(fallback_used=True))

        self.assertEqual(row["decision_model"], "demo-reasoning-model")
        self.assertEqual(row["response_model"], "demo-fast-model")
        self.assertEqual(row["original_model"], "demo-reasoning-model")
        self.assertTrue(row["fallback_used"])

    def test_unpriced_costs_stay_null(self) -> None:
        row = row_for(
            make_result(answering_cost=None, analyzer_cost=None)
        )

        self.assertIsNone(row["answering_cost_usd"])
        self.assertIsNone(row["analyzer_cost_usd"])

    def test_carries_the_profile_through(self) -> None:
        row = row_for(make_result())

        self.assertEqual(row["task_type"], "reasoning")
        self.assertEqual(row["confidence"], 0.8)
        self.assertEqual(row["latency_ms"], 1234.5)


class SaveResultTests(unittest.TestCase):
    def test_upserts_rather_than_inserts(self) -> None:
        """At-least-once delivery redelivers events, so a plain INSERT
        would fail on a duplicate key the second time around."""
        connection = FakeConnection()

        save_result(connection, make_result())  # type: ignore[arg-type]

        sql, _ = connection.executed[0]
        self.assertIn("ON CONFLICT (request_id) DO UPDATE", sql)

    def test_the_same_event_twice_writes_the_same_parameters(self) -> None:
        connection = FakeConnection()
        result = make_result("req-repeat")

        save_result(connection, result)  # type: ignore[arg-type]
        save_result(connection, result)  # type: ignore[arg-type]

        (_, first), (_, second) = connection.executed
        self.assertEqual(first, second)
        self.assertEqual(first["request_id"], "req-repeat")

    def test_commits_each_write(self) -> None:
        connection = FakeConnection()

        save_result(connection, make_result())  # type: ignore[arg-type]

        self.assertEqual(connection.commits, 1)


class SchemaTests(unittest.TestCase):
    def test_ddl_is_re_runnable(self) -> None:
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")

        self.assertIn("CREATE TABLE IF NOT EXISTS requests", ddl)
        self.assertEqual(ddl.count("CREATE INDEX IF NOT EXISTS"), 2)

    def test_money_is_numeric_not_floating_point(self) -> None:
        """Summing sub-cent floats accumulates error; NUMERIC does not."""
        ddl = SCHEMA_PATH.read_text(encoding="utf-8")

        self.assertIn("answering_cost_usd NUMERIC(12, 8)", ddl)
        self.assertIn("analyzer_cost_usd  NUMERIC(12, 8)", ddl)
