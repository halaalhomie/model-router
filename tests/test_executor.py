import unittest

from app.executor import execute_request
from tests.fakes import FakeResponse, FakeUsage, ScriptedClient


class ExecuteRequestTests(unittest.TestCase):
    def test_sends_the_request_to_the_routed_model(self) -> None:
        client = ScriptedClient([FakeResponse("The answer.")])

        response = execute_request(
            "What is a model router?",
            client=client,  # type: ignore[arg-type]
            model_name="demo-code-model",
        )

        self.assertEqual(response.text, "The answer.")
        self.assertEqual(response.model_name, "demo-code-model")

        request = client.models.calls[0]
        self.assertEqual(request["model"], "demo-code-model")
        self.assertEqual(request["contents"], "What is a model router?")

    def test_captures_token_usage_when_the_provider_reports_it(self) -> None:
        usage = FakeUsage(
            prompt_token_count=12,
            candidates_token_count=34,
            total_token_count=46,
        )
        client = ScriptedClient([FakeResponse("Answer.", usage)])

        response = execute_request(
            "Summarize this.",
            client=client,  # type: ignore[arg-type]
            model_name="demo-fast-model",
        )

        self.assertIsNotNone(response.usage)
        assert response.usage is not None
        self.assertEqual(response.usage.prompt_tokens, 12)
        self.assertEqual(response.usage.output_tokens, 34)
        self.assertEqual(response.usage.total_tokens, 46)
        self.assertEqual(response.usage.thinking_tokens, 0)

    def test_captures_thinking_tokens_from_a_reasoning_model(self) -> None:
        """Real Gemini usage: total exceeds prompt + output by the thinking
        tokens, which are billed as output."""
        usage = FakeUsage(
            prompt_token_count=11,
            candidates_token_count=841,
            total_token_count=1473,
            thoughts_token_count=621,
        )
        client = ScriptedClient([FakeResponse("Answer.", usage)])

        response = execute_request(
            "Explain sharding.",
            client=client,  # type: ignore[arg-type]
            model_name="demo-reasoning-model",
        )

        assert response.usage is not None
        self.assertEqual(response.usage.thinking_tokens, 621)
        self.assertGreater(
            response.usage.total_tokens,
            response.usage.prompt_tokens + response.usage.output_tokens,
        )

    def test_usage_is_none_when_the_provider_omits_it(self) -> None:
        client = ScriptedClient([FakeResponse("Answer.")])

        response = execute_request(
            "Summarize this.",
            client=client,  # type: ignore[arg-type]
            model_name="demo-fast-model",
        )

        self.assertIsNone(response.usage)

    def test_rejects_an_empty_request(self) -> None:
        client = ScriptedClient([FakeResponse("unused")])

        with self.assertRaises(ValueError):
            execute_request(
                "   ",
                client=client,  # type: ignore[arg-type]
                model_name="demo-fast-model",
            )

    def test_rejects_an_empty_model_reply(self) -> None:
        client = ScriptedClient([FakeResponse("")])

        with self.assertRaises(ValueError):
            execute_request(
                "Summarize this.",
                client=client,  # type: ignore[arg-type]
                model_name="demo-fast-model",
            )
