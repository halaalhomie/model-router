"""Analytics consumer: reads request events and aggregates them.

Run it in a second terminal while making requests in the first:

    venv\\Scripts\\python.exe -m app.consumer

This is the "many consumers, none in the caller's way" half of Phase 4.
It reads the same topic app/events.py writes to, at its own pace, and the
router neither knows nor waits for it. Stopping this consumer does not
affect the router at all; restarting it picks up exactly where it left
off, because Kafka stores this group's offsets server-side.

Delivery semantics, concretely
------------------------------
This consumer commits offsets *after* processing each event, which is
at-least-once: if it crashes between processing and committing, the event
is redelivered on restart and counted twice. The alternative -- committing
before processing -- is at-most-once, where that same crash loses the
event entirely.

At-least-once is the right default here, and everywhere that "losing data
silently" is worse than "seeing it twice." The cost is that processing
must tolerate duplicates. In-memory counters below do not (a duplicate
double-counts), which is acceptable for a restartable local aggregate but
is exactly why Phase 6's Postgres writes will need to be idempotent --
upsert on request_id, never a blind insert.
"""

import logging
import sys
from dataclasses import dataclass, field

from confluent_kafka import Consumer, KafkaError
from pydantic import ValidationError

from app.config import ConfigurationError, Settings, load_settings
from app.console import use_utf8_stdio
from app.events import REQUEST_EVENTS_TOPIC, base_client_config
from app.schemas import PipelineResult


logger = logging.getLogger(__name__)

POLL_TIMEOUT_SECONDS = 1.0


@dataclass
class ModelStats:
    """Per-model totals. Attributed to the model that actually answered
    (response.model_name), not the one routing picked -- when fallback
    fires those differ, and cost/latency belong to whoever did the work."""

    requests: int = 0
    latency_ms_total: float = 0.0
    tokens_total: int = 0
    cost_usd_total: float = 0.0
    # Requests whose cost could not be determined. Counted rather than
    # folded into cost_usd_total as zero, so a gap in pricing shows up as
    # a gap instead of quietly deflating the total.
    unpriced_requests: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.requests if self.requests else 0.0

    @property
    def cost_display(self) -> str:
        """"n/a" rather than 0.000000 when nothing here could be priced --
        a zero in a cost column reads as "this model is free", which is
        the precise misreading estimate_cost_usd returns None to avoid."""
        if self.requests and self.unpriced_requests == self.requests:
            return "n/a"
        return f"{self.cost_usd_total:.6f}"


@dataclass
class RequestStats:
    """Running totals across every event seen. Deliberately pure: it takes
    PipelineResult objects and knows nothing about Kafka, so the numbers
    can be tested without a broker."""

    total: int = 0
    fallbacks: int = 0
    confidence_total: float = 0.0
    by_model: dict[str, ModelStats] = field(default_factory=dict)
    # Routing overhead: what the classifier cost, tracked apart from the
    # models that did the answering. It belongs to no one model, and it is
    # the price the router pays that a single-model baseline does not.
    analyzer_cost_usd_total: float = 0.0
    analyzer_unpriced: int = 0

    def record(self, result: PipelineResult) -> None:
        self.total += 1
        self.confidence_total += result.profile.confidence
        if result.response.fallback_used:
            self.fallbacks += 1

        model = self.by_model.setdefault(
            result.response.model_name, ModelStats()
        )
        model.requests += 1
        model.latency_ms_total += result.latency_ms
        if result.response.usage is not None:
            model.tokens_total += result.response.usage.total_tokens

        cost = result.response.estimated_cost_usd
        if cost is None:
            model.unpriced_requests += 1
        else:
            model.cost_usd_total += cost

        if result.analyzer_cost_usd is None:
            self.analyzer_unpriced += 1
        else:
            self.analyzer_cost_usd_total += result.analyzer_cost_usd

    @property
    def avg_confidence(self) -> float:
        return self.confidence_total / self.total if self.total else 0.0

    @property
    def answering_cost_usd_total(self) -> float:
        return sum(stats.cost_usd_total for stats in self.by_model.values())

    @property
    def cost_usd_total(self) -> float:
        """Everything the router spent: answering plus routing overhead."""
        return self.answering_cost_usd_total + self.analyzer_cost_usd_total

    @property
    def unpriced_requests(self) -> int:
        return sum(
            stats.unpriced_requests for stats in self.by_model.values()
        )

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.total if self.total else 0.0

    def summary(self) -> str:
        if not self.total:
            return "No events consumed."

        lines = [
            f"requests       : {self.total}",
            f"avg confidence : {self.avg_confidence:.2f}",
            f"fallback rate  : {self.fallback_rate:.0%} "
            f"({self.fallbacks}/{self.total})",
            f"est. cost      : ${self.cost_usd_total:.6f} "
            f"(${self.analyzer_cost_usd_total:.6f} routing + "
            f"${self.answering_cost_usd_total:.6f} answering)",
            "",
            f"{'model':<26} {'reqs':>5} {'avg ms':>8} {'tokens':>8} "
            f"{'cost $':>10}",
        ]
        for name, stats in sorted(self.by_model.items()):
            lines.append(
                f"{name:<26} {stats.requests:>5} "
                f"{stats.avg_latency_ms:>8.0f} {stats.tokens_total:>8} "
                f"{stats.cost_display:>10}"
            )

        if self.unpriced_requests:
            lines.append(
                f"\nNote: {self.unpriced_requests} request(s) had no "
                f"pricing and are excluded from the cost totals."
            )
        return "\n".join(lines)


def parse_event(payload: bytes) -> PipelineResult:
    """Deserialize one event, validating it against the same schema the
    producer serialized from. This is the consumer's trust boundary: the
    bytes came off a topic anyone could have written to, so they get
    validated rather than assumed well-formed."""
    return PipelineResult.model_validate_json(payload)


def describe(result: PipelineResult) -> str:
    served_by = result.response.model_name
    note = " (fallback)" if result.response.fallback_used else ""
    return (
        f"{result.request_id[:8]}  {result.profile.task_type:<12} "
        f"-> {served_by}{note}  {result.latency_ms:.0f} ms"
    )


def build_consumer(
    settings: Settings, group_id: str | None = None
) -> Consumer:
    """Build a consumer, optionally in a group other than the configured
    one. Each group tracks its own offsets, so a second group reads the
    same topic without disturbing the first -- which is how the analytics
    and persistence consumers coexist."""
    return Consumer(
        base_client_config(settings)
        | {
            "group.id": group_id or settings.kafka_consumer_group,
            # Where a group with no committed offset starts. "earliest"
            # replays the whole topic, which is what an analytics consumer
            # wants; "latest" would ignore everything before it started.
            "auto.offset.reset": "earliest",
            # Off, so offsets are committed explicitly after processing --
            # see this module's docstring on delivery semantics.
            "enable.auto.commit": False,
        }
    )


def consume_forever(
    consumer: Consumer,
    stats: RequestStats,
    *,
    on_event: object = None,
    poll_timeout: float = POLL_TIMEOUT_SECONDS,
    max_events: int | None = None,
) -> RequestStats:
    """Poll, process, then commit -- in that order, for at-least-once.

    max_events exists so tests can run the loop to completion; left None,
    this runs until interrupted.
    """
    seen = 0

    while max_events is None or seen < max_events:
        message = consumer.poll(poll_timeout)

        if message is None:
            continue

        error = message.error()
        if error is not None:
            # _PARTITION_EOF just means "caught up with this partition",
            # which is normal and not a failure worth reporting.
            if error.code() != KafkaError._PARTITION_EOF:
                logger.error("Kafka error while consuming: %s", error)
            continue

        try:
            result = parse_event(message.value())
        except ValidationError as invalid:
            # Commit past a malformed event instead of retrying it forever
            # -- one poison message would otherwise block this partition
            # permanently, since the offset would never advance. Logged
            # without a traceback: this path is expected and handled, so
            # the field count is the useful part, not the stack.
            logger.warning(
                "Skipping malformed event at offset %s (%d validation "
                "errors); committing past it.",
                message.offset(),
                invalid.error_count(),
            )
            consumer.commit(message=message, asynchronous=False)
            seen += 1
            continue

        stats.record(result)
        if on_event is not None:
            on_event(result, stats)

        # After processing, never before: a crash above redelivers the
        # event rather than silently dropping it.
        consumer.commit(message=message, asynchronous=False)
        seen += 1

    return stats


def main() -> int:
    use_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    consumer = build_consumer(settings)
    consumer.subscribe([REQUEST_EVENTS_TOPIC])

    print(
        f"Consuming {REQUEST_EVENTS_TOPIC} as group "
        f"{settings.kafka_consumer_group!r} from "
        f"{settings.kafka_bootstrap_servers}. Ctrl+C to stop.\n"
    )

    stats = RequestStats()
    try:
        consume_forever(
            consumer,
            stats,
            on_event=lambda result, _stats: print(describe(result)),
        )
    except KeyboardInterrupt:
        print("\n")
    finally:
        # close() commits final offsets and leaves the group cleanly, so
        # Kafka rebalances immediately instead of waiting for this
        # consumer's session to time out.
        consumer.close()

    print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
