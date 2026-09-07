"""Consumer that writes request events into PostgreSQL.

    venv\\Scripts\\python.exe -m app.persist

This is the second consumer group on the same topic, and the point where
consumer groups stop being theory. The analytics consumer
(model-router.analytics) and this one (model-router.persistence) read the
identical stream, hold separate offsets, and neither can affect the
other's progress. Stop this for an hour and analytics keeps running;
start it again and it resumes from its own last commit, catching up on
everything it missed. That is the property Kafka was chosen for.

It reuses consume_forever unchanged: the poll/parse/commit loop, the
skip-past-poison-messages rule, and the at-least-once ordering are all
the same. Only what happens per event differs -- and because that loop
commits after processing, the write it performs has to be idempotent.
See app/storage.py.
"""

import logging
import sys

import psycopg

from app import storage
from app.config import ConfigurationError, load_settings
from app.console import use_utf8_stdio
from app.consumer import RequestStats, build_consumer, consume_forever
from app.events import REQUEST_EVENTS_TOPIC
from app.schemas import PipelineResult


logger = logging.getLogger(__name__)

PERSISTENCE_GROUP = "model-router.persistence"


def describe(result: PipelineResult, stored: int) -> str:
    return (
        f"{result.request_id[:8]}  {result.profile.task_type:<12} "
        f"-> {result.response.model_name}   (stored {stored})"
    )


def main() -> int:
    use_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    try:
        connection = storage.connect(settings)
    except psycopg.OperationalError as error:
        # Unlike Kafka, there is no point starting without the database:
        # this process exists only to write to it.
        print(f"Cannot reach PostgreSQL: {error}", file=sys.stderr)
        print("Is `docker compose up -d` running?", file=sys.stderr)
        return 2

    storage.ensure_schema(connection)

    consumer = build_consumer(settings, group_id=PERSISTENCE_GROUP)
    consumer.subscribe([REQUEST_EVENTS_TOPIC])

    print(
        f"Persisting {REQUEST_EVENTS_TOPIC} as group "
        f"{PERSISTENCE_GROUP!r} into PostgreSQL. Ctrl+C to stop.\n"
    )

    stored = 0

    def store(result: PipelineResult, _stats: RequestStats) -> None:
        nonlocal stored
        storage.save_result(connection, result)
        stored += 1
        print(describe(result, stored))

    try:
        consume_forever(consumer, RequestStats(), on_event=store)
    except KeyboardInterrupt:
        print("\n")
    finally:
        consumer.close()
        connection.close()

    print(f"Stored {stored} request(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
