import unittest

from app.main import format_result, read_request
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
    TokenUsage,
)


def make_result(text: str) -> PipelineResult:
    return PipelineResult(
        request_id="req-abc",
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
        latency_ms=1234.5,
    )


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

    def test_shows_a_fallback_line_when_fallback_was_used(self) -> None:
        result = make_result("Fallback answer.")
        result = result.model_copy(
            update={
                "response": result.response.model_copy(
                    update={
                        "fallback_used": True,
                        "original_model": "demo-reasoning-model",
                        "model_name": "demo-fast-model",
                    }
                )
            }
        )

        output = format_result(result)

        self.assertIn(
            "fallback     : demo-reasoning-model failed, "
            "used demo-fast-model instead",
            output,
        )

    def test_omits_the_fallback_line_when_not_used(self) -> None:
        output = format_result(make_result("Answer."))

        self.assertNotIn("fallback", output)
