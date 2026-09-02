"""Publishes one event per completed request to Kafka.

This is the asynchronous half of the architecture: the synchronous request
path is CONFIG -> ANALYZE -> ROUTE -> EXECUTE -> RESPOND, and this module is
never part of it. A request must succeed or fail on its own merits --
whether Kafka is up, slow, or completely unreachable must never change
whether the caller gets their answer. pipeline.py enforces that by treating
every failure from publish() as a log line, never a raised exception.

Concretely, this rests on three confluent_kafka.Producer methods that are
easy to confuse:
  produce()  queues the message locally and returns immediately -- it does
             not wait for the broker. This is what makes publishing "async."
  poll(0)    services any delivery-report callbacks for messages that have
             already been sent, without blocking for new ones. Cheap,
             called after every produce().
  flush()    blocks until every queued message is actually sent (or a
             timeout elapses). The only place this belongs is where a
             process is about to exit and would otherwise silently drop
             whatever is still queued -- see main.py (after printing the
             answer) and api.py's lifespan shutdown. Never call it per
             request: that would turn "async" back into "sync."
"""

import logging
from typing import Protocol

from confluent_kafka import Producer

from app.config import Settings
from app.schemas import PipelineResult


logger = logging.getLogger(__name__)

REQUEST_EVENTS_TOPIC = "model-router.requests"


class EventPublisher(Protocol):
    def publish(self, result: PipelineResult) -> None: ...


class KafkaEventPublisher:
    """Publishes the full PipelineResult, keyed by request_id.

    Keying by request_id (rather than leaving the key unset) means every
    event for the same request -- if this ever grows into multiple events
    per request -- lands in the same partition, preserving their order.
    With one event per request today this mostly just spreads load evenly
    across partitions, but costs nothing to set up correctly now.
    """

    def __init__(
        self, producer: Producer, topic: str = REQUEST_EVENTS_TOPIC
    ) -> None:
        self._producer = producer
        self._topic = topic

    def publish(self, result: PipelineResult) -> None:
        self._producer.produce(
            self._topic,
            key=result.request_id,
            value=result.model_dump_json(),
        )
        self._producer.poll(0)


def build_producer(settings: Settings) -> Producer:
    return Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})


def publish_safely(
    publisher: EventPublisher | None, result: PipelineResult
) -> None:
    """Publish if a publisher was given, and never let it fail the request.

    This is the one place the "Kafka failures are not request failures"
    policy is enforced -- see the module docstring. Every caller (main.py,
    api.py) goes through this instead of calling publisher.publish()
    directly, so the policy can't be accidentally skipped by a new caller.
    """
    if publisher is None:
        return
    try:
        publisher.publish(result)
    except Exception:
        logger.warning(
            "Failed to publish event for request %s -- the response to "
            "the caller is unaffected.",
            result.request_id,
            exc_info=True,
        )
