import unittest
from datetime import datetime, timezone
from decimal import Decimal

from app.stats import (
    read_by_model,
    read_by_task_type,
    read_dashboard,
    read_recent,
    read_totals,
)


class FakeCursor:
    def __init__(self, results: list) -> None:
        self.results = results
        self.executed: list = []

    def execute(self, sql: str, params: object = None) -> None:
        self.executed.append((sql, params))

    def fetchone(self):
        return self.results[0]

    def fetchall(self):
        return self.results

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeConnection:
    """Serves queued result sets in order, one per cursor()."""

    def __init__(self, *result_sets: list) -> None:
        self.result_sets = list(result_sets)
        self.executed: list = []

    def cursor(self) -> FakeCursor:
        results = self.result_sets.pop(0) if self.result_sets else []
        cursor = FakeCursor(results)
        self.executed.append(cursor)
        return cursor


class ReadTotalsTests(unittest.TestCase):
    def test_splits_routing_from_answering_and_adds_them(self) -> None:
        connection = FakeConnection(
            [(10, Decimal("0.05"), Decimal("0.002"), 2500.0, 0.9, 2, 1)]
        )

        totals = read_totals(connection)  # type: ignore[arg-type]

        self.assertEqual(totals.requests, 10)
        self.assertAlmostEqual(totals.answering_cost_usd, 0.05)
        self.assertAlmostEqual(totals.analyzer_cost_usd, 0.002)
        self.assertAlmostEqual(totals.total_cost_usd, 0.052)

    def test_computes_fallback_rate(self) -> None:
        connection = FakeConnection(
            [(8, Decimal("0"), Decimal("0"), 0.0, 0.0, 2, 0)]
        )

        totals = read_totals(connection)  # type: ignore[arg-type]

        self.assertAlmostEqual(totals.fallback_rate, 0.25)

    def test_an_empty_table_does_not_divide_by_zero(self) -> None:
        connection = FakeConnection(
            [(0, Decimal("0"), Decimal("0"), 0.0, 0.0, 0, 0)]
        )

        totals = read_totals(connection)  # type: ignore[arg-type]

        self.assertEqual(totals.requests, 0)
        self.assertEqual(totals.fallback_rate, 0.0)

    def test_reports_unpriced_rather_than_hiding_them(self) -> None:
        """A total that silently dropped unpriced rows would understate."""
        connection = FakeConnection(
            [(5, Decimal("0.01"), Decimal("0.001"), 100.0, 1.0, 0, 3)]
        )

        totals = read_totals(connection)  # type: ignore[arg-type]

        self.assertEqual(totals.unpriced, 3)


class ReadByModelTests(unittest.TestCase):
    def test_maps_rows_including_percentiles(self) -> None:
        connection = FakeConnection([
            ("fast", 3, 2500.0, 2653.0, 51, Decimal("0.000025"), 2),
            ("slow", 1, 12693.0, 12693.0, 1748, Decimal("0.01565"), 0),
        ])

        rows = read_by_model(connection)  # type: ignore[arg-type]

        self.assertEqual(rows[0].model, "fast")
        self.assertAlmostEqual(rows[0].p95_latency_ms, 2653.0)
        self.assertEqual(rows[0].unpriced, 2)
        self.assertEqual(rows[1].tokens, 1748)

    def test_handles_no_rows(self) -> None:
        connection = FakeConnection([])

        self.assertEqual(read_by_model(connection), [])  # type: ignore[arg-type]


class ReadByTaskTypeTests(unittest.TestCase):
    def test_maps_counts(self) -> None:
        connection = FakeConnection([("factual", 3), ("coding", 1)])

        rows = read_by_task_type(connection)  # type: ignore[arg-type]

        self.assertEqual([(r.task_type, r.requests) for r in rows],
                         [("factual", 3), ("coding", 1)])


class ReadRecentTests(unittest.TestCase):
    def test_serialises_timestamps_and_passes_the_limit(self) -> None:
        when = datetime(2026, 9, 8, 12, 30, tzinfo=timezone.utc)
        connection = FakeConnection([
            ("req-1", "factual", "fast", False, 2500.0, Decimal("0.0002"), when),
        ])

        rows = read_recent(connection, limit=5)  # type: ignore[arg-type]

        self.assertEqual(rows[0].request_id, "req-1")
        self.assertIn("2026-09-08", rows[0].created_at)
        _, params = connection.executed[0].executed[0]
        self.assertEqual(params, {"limit": 5})


class ReadDashboardTests(unittest.TestCase):
    def test_assembles_every_section(self) -> None:
        connection = FakeConnection(
            [(1, Decimal("0.01"), Decimal("0.001"), 100.0, 0.9, 0, 0)],
            [("fast", 1, 100.0, 100.0, 10, Decimal("0.01"), 0)],
            [("factual", 1)],
            [],
        )

        dashboard = read_dashboard(connection)  # type: ignore[arg-type]

        self.assertEqual(dashboard.totals.requests, 1)
        self.assertEqual(len(dashboard.by_model), 1)
        self.assertEqual(len(dashboard.by_task_type), 1)
        self.assertEqual(dashboard.recent, [])
