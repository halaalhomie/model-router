import unittest

from app.main import format_result, read_request, use_utf8
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
    TokenUsage,
)


def make_result(text: str) -> PipelineResult:
    return PipelineResult(
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
            text=text,
            usage=TokenUsage(
                prompt_tokens=11,
                output_tokens=841,
                thinking_tokens=621,
                total_tokens=1473,
            ),
        ),
    )


class FakeStream:
    def __init__(self) -> None:
        self.encoding: str | None = None
        self.errors: str | None = None

    def reconfigure(self, *, encoding: str, errors: str) -> None:
        self.encoding = encoding
        self.errors = errors


class UseUtf8Tests(unittest.TestCase):
    def test_switches_the_stream_to_utf8(self) -> None:
        stream = FakeStream()

        use_utf8(stream)

        self.assertEqual(stream.encoding, "utf-8")
        self.assertEqual(stream.errors, "replace")

    def test_ignores_a_stream_that_cannot_be_reconfigured(self) -> None:
        use_utf8(object())  # must not raise


class ReadRequestTests(unittest.TestCase):
    def test_joins_command_line_arguments(self) -> None:
        self.assertEqual(
            read_request(["Explain", "sharding."]), "Explain sharding."
        )


class FormatResultTests(unittest.TestCase):
    def test_shows_the_routing_decision_and_the_answer(self) -> None:
        output = format_result(make_result("Sharding splits data."))

        self.assertIn("model        : demo-reasoning-model", output)
        self.assertIn("confidence   : 0.80", output)
        self.assertIn("621 thinking", output)
        self.assertIn("Sharding splits data.", output)

    def test_survives_characters_a_cp1252_console_cannot_encode(self) -> None:
        """Regression: model output with em dashes crashed print() on Windows
        before main() forced UTF-8 on stdout."""
        output = format_result(make_result("Shard — then route → done."))

        self.assertIn("—", output)
        output.encode("utf-8")
