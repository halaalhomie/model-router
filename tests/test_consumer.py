import logging
import unittest

from confluent_kafka import KafkaError

from app.consumer import RequestStats, consume_forever, describe, parse_event
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
    TokenUsage,
)
from tests.fakes import FakeKafkaConsumer, FakeKafkaError, FakeMessage


def make_result(
    request_id: str = "req-abc",
    *,
    served_by: str = "demo-fast-model",
    latency_ms: float = 100.0,
    total_tokens: int = 50,
    fallback_used: bool = False,
    confidence: float = 0.9,
    estimated_cost_usd: float | None = 0.001,
    analyzer_cost: float | None = 0.0001,
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
            confidence=confidence,
        ),
        decision=RoutingDecision(
            model_name="demo-reasoning-model",
            reason="The request needs high-level reasoning.",
        ),
        response=ModelResponse(
            model_name=served_by,
            text="Sharding splits data.",
            usage=TokenUsage(
                prompt_tokens=10,
                output_tokens=40,
                total_tokens=total_tokens,
            ),
            fallback_used=fallback_used,
            original_model="demo-reasoning-model" if fallback_used else None,
            estimated_cost_usd=estimated_cost_usd,
        ),
        latency_ms=latency_ms,
        analyzer_cost_usd=analyzer_cost,
    )


def event(result: PipelineResult, offset: int = 0) -> FakeMessage:
    return FakeMessage(value=result.model_dump_json().encode(), offset=offset)


class RequestStatsTests(unittest.TestCase):
    def test_starts_empty(self) -> None:
        stats = RequestStats()

        self.assertEqual(stats.total, 0)
        self.assertEqual(stats.avg_confidence, 0.0)
        self.assertEqual(stats.fallback_rate, 0.0)
        self.assertIn("No events", stats.summary())

    def test_groups_totals_by_the_model_that_answered(self) -> None:
        stats = RequestStats()

        stats.record(make_result(served_by="fast", latency_ms=100))
        stats.record(make_result(served_by="fast", latency_ms=300))
        stats.record(make_result(served_by="slow", latency_ms=1000))

        self.assertEqual(stats.total, 3)
        self.assertEqual(stats.by_model["fast"].requests, 2)
        self.assertEqual(stats.by_model["fast"].avg_latency_ms, 200)
        self.assertEqual(stats.by_model["slow"].requests, 1)

    def test_attributes_a_fallback_to_the_model_that_actually_answered(
        self,
    ) -> None:
        """decision.model_name says who was chosen; response.model_name says
        who did the work. Cost and latency belong to the latter."""
        stats = RequestStats()

        stats.record(make_result(served_by="demo-fast-model", fallback_used=True))

        self.assertIn("demo-fast-model", stats.by_model)
        self.assertNotIn("demo-reasoning-model", stats.by_model)
        self.assertEqual(stats.fallbacks, 1)
        self.assertEqual(stats.fallback_rate, 1.0)

    def test_sums_tokens_and_averages_confidence(self) -> None:
        stats = RequestStats()

        stats.record(make_result(total_tokens=100, confidence=1.0))
        stats.record(make_result(total_tokens=50, confidence=0.5))

        self.assertEqual(stats.by_model["demo-fast-model"].tokens_total, 150)
        self.assertEqual(stats.avg_confidence, 0.75)

    def test_tolerates_a_result_without_usage(self) -> None:
        stats = RequestStats()
        result = make_result()
        result = result.model_copy(
            update={"response": result.response.model_copy(update={"usage": None})}
        )

        stats.record(result)

        self.assertEqual(stats.by_model["demo-fast-model"].tokens_total, 0)

    def test_sums_cost_per_model_and_overall(self) -> None:
        stats = RequestStats()

        stats.record(make_result(served_by="cheap", estimated_cost_usd=0.001))
        stats.record(make_result(served_by="cheap", estimated_cost_usd=0.002))
        stats.record(make_result(served_by="dear", estimated_cost_usd=0.05))

        self.assertAlmostEqual(stats.by_model["cheap"].cost_usd_total, 0.003)
        self.assertAlmostEqual(stats.by_model["dear"].cost_usd_total, 0.05)
        self.assertAlmostEqual(stats.answering_cost_usd_total, 0.053)
        # cost_usd_total also carries three classifier calls at 0.0001
        self.assertAlmostEqual(stats.cost_usd_total, 0.0533)

    def test_tracks_routing_overhead_apart_from_answering(self) -> None:
        """The classifier's cost belongs to no single model, and it is
        exactly the overhead a single-model baseline never pays."""
        stats = RequestStats()

        stats.record(
            make_result(
                served_by="fast", estimated_cost_usd=0.01, analyzer_cost=0.002
            )
        )
        stats.record(
            make_result(
                served_by="fast", estimated_cost_usd=0.01, analyzer_cost=0.002
            )
        )

        self.assertAlmostEqual(stats.answering_cost_usd_total, 0.02)
        self.assertAlmostEqual(stats.analyzer_cost_usd_total, 0.004)
        self.assertAlmostEqual(stats.cost_usd_total, 0.024)
        self.assertNotIn(
            0.002, [s.cost_usd_total for s in stats.by_model.values()]
        )

    def test_counts_unpriced_analyzer_calls_separately(self) -> None:
        stats = RequestStats()

        stats.record(make_result(estimated_cost_usd=0.01, analyzer_cost=None))

        self.assertEqual(stats.analyzer_unpriced, 1)
        self.assertAlmostEqual(stats.analyzer_cost_usd_total, 0.0)
        self.assertAlmostEqual(stats.cost_usd_total, 0.01)

    def test_counts_unpriced_requests_instead_of_treating_them_as_free(
        self,
    ) -> None:
        """A None cost must not deflate the total by counting as zero."""
        stats = RequestStats()

        stats.record(
            make_result(
                served_by="priced", estimated_cost_usd=0.01, analyzer_cost=None
            )
        )
        stats.record(
            make_result(
                served_by="mystery", estimated_cost_usd=None, analyzer_cost=None
            )
        )

        self.assertAlmostEqual(stats.cost_usd_total, 0.01)
        self.assertEqual(stats.unpriced_requests, 1)
        self.assertEqual(stats.by_model["mystery"].cost_usd_total, 0.0)

    def test_shows_na_rather_than_zero_for_a_fully_unpriced_model(
        self,
    ) -> None:
        """A 0.000000 in a cost column reads as free, which is exactly the
        misreading None is meant to prevent."""
        stats = RequestStats()
        stats.record(make_result(served_by="mystery", estimated_cost_usd=None))

        self.assertEqual(stats.by_model["mystery"].cost_display, "n/a")
        self.assertIn("n/a", stats.summary())

    def test_shows_a_number_when_any_request_was_priced(self) -> None:
        stats = RequestStats()
        stats.record(make_result(served_by="mixed", estimated_cost_usd=0.01))
        stats.record(make_result(served_by="mixed", estimated_cost_usd=None))

        self.assertEqual(stats.by_model["mixed"].cost_display, "0.010000")

    def test_summary_flags_unpriced_requests(self) -> None:
        stats = RequestStats()
        stats.record(make_result(estimated_cost_usd=None))

        self.assertIn("no pricing", stats.summary())

    def test_summary_omits_the_note_when_everything_is_priced(self) -> None:
        stats = RequestStats()
        stats.record(make_result(estimated_cost_usd=0.01))

        self.assertNotIn("no pricing", stats.summary())

    def test_summary_lists_each_model(self) -> None:
        stats = RequestStats()
        stats.record(make_result(served_by="model-a"))
        stats.record(make_result(served_by="model-b"))

        summary = stats.summary()

        self.assertIn("model-a", summary)
        self.assertIn("model-b", summary)
        self.assertIn("requests       : 2", summary)


class ParseEventTests(unittest.TestCase):
    def test_round_trips_what_the_producer_serialized(self) -> None:
        original = make_result("req-42")

        parsed = parse_event(original.model_dump_json().encode())

        self.assertEqual(parsed.request_id, "req-42")
        self.assertEqual(parsed.latency_ms, original.latency_ms)

    def test_rejects_a_payload_that_is_not_a_pipeline_result(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            parse_event(b'{"nonsense": true}')


class DescribeTests(unittest.TestCase):
    def test_marks_a_fallback(self) -> None:
        line = describe(make_result(served_by="fast", fallback_used=True))

        self.assertIn("(fallback)", line)

    def test_omits_the_marker_otherwise(self) -> None:
        self.assertNotIn("fallback", describe(make_result()))


class ConsumeForeverTests(unittest.TestCase):
    def test_records_each_event(self) -> None:
        consumer = FakeKafkaConsumer(
            [event(make_result("a")), event(make_result("b"))]
        )
        stats = RequestStats()

        consume_forever(consumer, stats, max_events=2)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 2)

    def test_commits_after_processing_each_event(self) -> None:
        """At-least-once: the commit follows the work, so a crash in
        between redelivers rather than silently drops."""
        messages = [event(make_result("a"), offset=0)]
        consumer = FakeKafkaConsumer(messages)

        consume_forever(  # type: ignore[arg-type]
            consumer, RequestStats(), max_events=1
        )

        self.assertEqual(len(consumer.committed), 1)

    def test_skips_and_commits_past_a_malformed_event(self) -> None:
        """A poison message must not block the partition forever."""
        consumer = FakeKafkaConsumer(
            [FakeMessage(value=b"not json", offset=7)]
        )
        stats = RequestStats()

        with self.assertLogs("app.consumer", level=logging.WARNING):
            consume_forever(consumer, stats, max_events=1)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 0)
        self.assertEqual(len(consumer.committed), 1)

    def test_ignores_end_of_partition(self) -> None:
        """_PARTITION_EOF means "caught up", not a failure."""
        consumer = FakeKafkaConsumer(
            [
                FakeMessage(error=FakeKafkaError(KafkaError._PARTITION_EOF)),
                event(make_result("a")),
            ]
        )
        stats = RequestStats()

        consume_forever(consumer, stats, max_events=1)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 1)
        self.assertEqual(len(consumer.committed), 1)

    def test_waits_quietly_for_a_topic_that_does_not_exist_yet(self) -> None:
        """A consumer that starts before the first publish sees this until
        the topic is created. It resolves itself, so it is not an error."""
        consumer = FakeKafkaConsumer(
            [
                FakeMessage(
                    error=FakeKafkaError(KafkaError.UNKNOWN_TOPIC_OR_PART)
                ),
                event(make_result("a")),
            ]
        )
        stats = RequestStats()

        with self.assertLogs("app.consumer", level=logging.INFO) as captured:
            consume_forever(consumer, stats, max_events=1)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 1)
        self.assertNotIn("ERROR", captured.output[0])

    def test_logs_a_real_kafka_error_without_stopping(self) -> None:
        consumer = FakeKafkaConsumer(
            [
                FakeMessage(error=FakeKafkaError(KafkaError.BROKER_NOT_AVAILABLE)),
                event(make_result("a")),
            ]
        )
        stats = RequestStats()

        with self.assertLogs("app.consumer", level=logging.ERROR):
            consume_forever(consumer, stats, max_events=1)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 1)

    def test_skips_idle_polls(self) -> None:
        consumer = FakeKafkaConsumer([None, None, event(make_result("a"))])
        stats = RequestStats()

        consume_forever(consumer, stats, max_events=1)  # type: ignore[arg-type]

        self.assertEqual(stats.total, 1)

    def test_calls_the_on_event_hook(self) -> None:
        seen: list[str] = []
        consumer = FakeKafkaConsumer([event(make_result("req-9"))])

        consume_forever(  # type: ignore[arg-type]
            consumer,
            RequestStats(),
            on_event=lambda result, _stats: seen.append(result.request_id),
            max_events=1,
        )

        self.assertEqual(seen, ["req-9"])
