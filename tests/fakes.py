"""Fakes and fixtures so tests never touch Gemini or Kafka for real."""

import json

from app.config import Settings
from app.schemas import ModelCatalog


SETTINGS = Settings(
    gemini_api_key="test-key",
    analyzer_model="demo-analyzer-model",
    catalog=ModelCatalog(
        fast_model="demo-fast-model",
        code_model="demo-code-model",
        reasoning_model="demo-reasoning-model",
        long_context_model="demo-long-context-model",
    ),
    request_timeout_seconds=30.0,
    kafka_bootstrap_servers="localhost:9092",
    kafka_consumer_group="test-group",
    database_url="postgresql://test:test@localhost:5432/test",
)


def profile_json(**changes: object) -> str:
    """The JSON an analyzer call would return, with per-test overrides."""
    values: dict[str, object] = {
        "task_type": "general",
        "difficulty": "low",
        "reasoning_required": "low",
        "context_size": "small",
        "output_type": "text",
        "confidence": 0.9,
    }
    values.update(changes)
    return json.dumps(values)


class FakeUsage:
    def __init__(
        self,
        prompt_token_count: int = 0,
        candidates_token_count: int = 0,
        total_token_count: int = 0,
        thoughts_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.thoughts_token_count = thoughts_token_count


class FakeResponse:
    def __init__(self, text: str, usage: FakeUsage | None = None) -> None:
        self.text = text
        if usage is not None:
            self.usage_metadata = usage


class ScriptedModels:
    """Return queued responses in order and record every request.

    A queued item that is an Exception is raised instead of returned, so
    tests can script "fails twice, then succeeds" without a real network
    call.
    """

    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "generate_content called more times than queued."
            )
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ScriptedClient:
    def __init__(self, responses: list[FakeResponse | Exception]) -> None:
        self.models = ScriptedModels(responses)


class FakeEventPublisher:
    """Satisfies the EventPublisher Protocol without a Kafka broker.

    Set `error` to make publish() raise, which is how tests check that a
    broken Kafka never breaks a request.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.published: list[object] = []
        self.error = error

    def publish(self, result: object) -> None:
        if self.error is not None:
            raise self.error
        self.published.append(result)


class FakeKafkaError:
    """Stands in for confluent_kafka.KafkaError."""

    def __init__(self, code: int) -> None:
        self._code = code

    def code(self) -> int:
        return self._code

    def __str__(self) -> str:
        return f"FakeKafkaError({self._code})"


class FakeMessage:
    """Stands in for confluent_kafka.Message (a C extension type)."""

    def __init__(
        self,
        value: bytes | None = None,
        error: FakeKafkaError | None = None,
        offset: int = 0,
        partition: int = 0,
    ) -> None:
        self._value = value
        self._error = error
        self._offset = offset
        self._partition = partition

    def value(self) -> bytes | None:
        return self._value

    def error(self) -> FakeKafkaError | None:
        return self._error

    def offset(self) -> int:
        return self._offset

    def partition(self) -> int:
        return self._partition


class FakeKafkaConsumer:
    """Returns queued poll() results in order and records commits.

    A queued None models "poll timed out with nothing available", which is
    the normal idle case the real client returns constantly.
    """

    def __init__(self, messages: list[FakeMessage | None]) -> None:
        self.messages = list(messages)
        self.committed: list[FakeMessage] = []
        self.closed = False
        self.subscribed: list[str] = []

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed = topics

    def poll(self, timeout: float) -> FakeMessage | None:
        if not self.messages:
            return None
        return self.messages.pop(0)

    def commit(self, message: FakeMessage, asynchronous: bool = True) -> None:
        self.committed.append(message)

    def close(self) -> None:
        self.closed = True


class FakeKafkaProducer:
    """Stands in for confluent_kafka.Producer, which is a C extension type
    that can't be meaningfully subclassed or inspected in a unit test."""

    def __init__(self) -> None:
        self.produced: list[dict[str, object]] = []
        self.poll_calls: list[float] = []
        self.flush_calls: list[float] = []

    def produce(self, topic: str, key: object = None, value: object = None) -> None:
        self.produced.append({"topic": topic, "key": key, "value": value})

    def poll(self, timeout: float) -> int:
        self.poll_calls.append(timeout)
        return 0

    def flush(self, timeout: float) -> int:
        self.flush_calls.append(timeout)
        return 0
