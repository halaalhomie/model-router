import json
import logging
import unittest

from app.events import (
    REQUEST_EVENTS_TOPIC,
    KafkaEventPublisher,
    publish_safely,
)
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
    TokenUsage,
)
from tests.fakes import FakeEventPublisher, FakeKafkaProducer


def make_result(request_id: str = "req-abc") -> PipelineResult:
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
            model_name="demo-reasoning-model",
            reason="The request needs high-level reasoning.",
        ),
        response=ModelResponse(
            model_name="demo-reasoning-model",
            text="Sharding splits data.",
            usage=TokenUsage(
                prompt_tokens=11,
                output_tokens=841,
                thinking_tokens=621,
                total_tokens=1473,
            ),
        ),
        latency_ms=1234.5,
    )


class KafkaEventPublisherTests(unittest.TestCase):
    def test_publishes_to_the_requests_topic_keyed_by_request_id(self) -> None:
        producer = FakeKafkaProducer()

        KafkaEventPublisher(producer).publish(make_result("req-42"))

        self.assertEqual(len(producer.produced), 1)
        sent = producer.produced[0]
        self.assertEqual(sent["topic"], REQUEST_EVENTS_TOPIC)
        self.assertEqual(sent["key"], "req-42")

    def test_publishes_the_whole_result_as_json(self) -> None:
        producer = FakeKafkaProducer()

        KafkaEventPublisher(producer).publish(make_result())

        payload = json.loads(producer.produced[0]["value"])
        self.assertEqual(payload["request_id"], "req-abc")
        self.assertEqual(payload["decision"]["model_name"], "demo-reasoning-model")
        self.assertEqual(payload["response"]["usage"]["total_tokens"], 1473)
        self.assertEqual(payload["latency_ms"], 1234.5)

    def test_polls_without_blocking_after_producing(self) -> None:
        """poll(0) services delivery callbacks already completed; it must
        never be flush(), which would block the request on the broker."""
        producer = FakeKafkaProducer()

        KafkaEventPublisher(producer).publish(make_result())

        self.assertEqual(producer.poll_calls, [0])
        self.assertEqual(producer.flush_calls, [])


class PublishSafelyTests(unittest.TestCase):
    def test_publishes_when_a_publisher_is_given(self) -> None:
        publisher = FakeEventPublisher()
        result = make_result()

        publish_safely(publisher, result)

        self.assertEqual(publisher.published, [result])

    def test_does_nothing_when_no_publisher_is_configured(self) -> None:
        publish_safely(None, make_result())  # must not raise

    def test_swallows_and_logs_a_publish_failure(self) -> None:
        """A broken Kafka must never become a failed request."""
        publisher = FakeEventPublisher(error=RuntimeError("broker down"))

        with self.assertLogs("app.events", level=logging.WARNING) as captured:
            publish_safely(publisher, make_result("req-99"))

        self.assertIn("req-99", captured.output[0])
