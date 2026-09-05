import sys

from app.client import build_client
from app.config import ConfigurationError, load_settings
from app.console import use_utf8_stdio
from app.events import KafkaEventPublisher, build_producer
from app.pipeline import run_pipeline
from app.schemas import PipelineResult


USAGE = 'Usage: python -m app.main "your request here"'

# How long to wait, at process exit, for any queued Kafka message to
# actually reach the broker (see app/events.py's module docstring for why
# this -- and only this -- is where flush() belongs). Bounded so a
# down/unreachable Kafka delays exit by at most this long, never hangs it.
KAFKA_FLUSH_TIMEOUT_SECONDS = 5.0


def read_request(argv: list[str]) -> str:
    """Take the request from the command line, or from a pipe."""
    if argv:
        return " ".join(argv)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def format_result(result: PipelineResult) -> str:
    """Show the routing decision above the answer, so the choice is visible."""
    profile = result.profile
    lines = [
        f"request id   : {result.request_id}",
        f"task type    : {profile.task_type}",
        f"difficulty   : {profile.difficulty}",
        f"reasoning    : {profile.reasoning_required}",
        f"context size : {profile.context_size}",
        f"output type  : {profile.output_type}",
        f"confidence   : {profile.confidence:.2f}",
        f"model        : {result.decision.model_name}",
        f"reason       : {result.decision.reason}",
    ]

    if result.response.fallback_used:
        lines.append(
            f"fallback     : {result.response.original_model} failed, "
            f"used {result.response.model_name} instead"
        )

    usage = result.response.usage
    if usage is not None:
        lines.append(
            f"tokens       : {usage.prompt_tokens} in / "
            f"{usage.output_tokens} out / "
            f"{usage.thinking_tokens} thinking / "
            f"{usage.total_tokens} total"
        )

    cost = result.response.estimated_cost_usd
    if cost is not None:
        # Six decimals because a flash-lite call lands around $0.00002 --
        # fewer would round most requests to $0.00 and hide the thing we
        # are trying to measure.
        lines.append(f"est. cost    : ${cost:.6f} (paid-tier rates)")

    lines.append(f"latency      : {result.latency_ms:.0f} ms")
    lines.extend(["", result.response.text])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdio()

    request_text = read_request(sys.argv[1:] if argv is None else argv)

    if not request_text.strip():
        print(USAGE, file=sys.stderr)
        return 2

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    client = build_client(settings)
    producer = build_producer(settings)
    publisher = KafkaEventPublisher(producer)

    result = run_pipeline(
        request_text, client=client, settings=settings, publisher=publisher
    )

    print(format_result(result))

    # The process is about to exit -- flush now, or whatever publish()
    # queued but hadn't sent yet is silently lost. This is a one-shot CLI
    # process, not a long-lived server, so blocking briefly here (after the
    # user already has their answer) is the right tradeoff; api.py flushes
    # at server shutdown instead, for the same reason.
    still_queued = producer.flush(KAFKA_FLUSH_TIMEOUT_SECONDS)
    if still_queued > 0:
        print(
            f"Warning: {still_queued} Kafka message(s) were not delivered "
            f"before exit.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())