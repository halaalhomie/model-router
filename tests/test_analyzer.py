import unittest

from pydantic import ValidationError

from app.analyzer import analyze_task
from tests.fakes import FakeResponse, FakeUsage, ScriptedClient, profile_json


class TaskAnalyzerTests(unittest.TestCase):
    def test_returns_a_validated_task_profile(self) -> None:
        client = ScriptedClient([FakeResponse(profile_json(task_type="coding"))])

        analysis = analyze_task(
            "Fix a Python bug that reverses a list.",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-test-model",
        )

        self.assertEqual(analysis.profile.task_type, "coding")
        self.assertEqual(analysis.model_name, "gemini-test-model")
        self.assertEqual(client.models.calls[0]["model"], "gemini-test-model")

    def test_rejects_an_unknown_task_type_from_the_model(self) -> None:
        client = ScriptedClient(
            [FakeResponse(profile_json(task_type="debugging"))]
        )

        with self.assertRaises(ValidationError):
            analyze_task(
                "Fix a Python bug.",
                client=client,  # type: ignore[arg-type]
                model_name="gemini-test-model",
            )

    def test_rejects_an_empty_request(self) -> None:
        client = ScriptedClient([FakeResponse(profile_json())])

        with self.assertRaises(ValueError):
            analyze_task(
                "   ",
                client=client,  # type: ignore[arg-type]
                model_name="gemini-test-model",
            )


class AnalyzerCostTests(unittest.TestCase):
    """The classifier runs on every request, so what it costs is part of
    what routing costs -- see AnalysisResult's docstring."""

    def test_captures_what_the_classification_cost(self) -> None:
        usage = FakeUsage(
            prompt_token_count=100_000,
            candidates_token_count=100_000,
            total_token_count=200_000,
        )
        client = ScriptedClient([FakeResponse(profile_json(), usage)])

        analysis = analyze_task(
            "Say hello.",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        assert analysis.usage is not None
        self.assertEqual(analysis.usage.prompt_tokens, 100_000)
        # 0.1M in at $0.30 + 0.1M out at $2.50 = 0.03 + 0.25
        self.assertAlmostEqual(analysis.estimated_cost_usd, 0.28)

    def test_reports_no_cost_for_an_unpriced_analyzer_model(self) -> None:
        usage = FakeUsage(prompt_token_count=10, candidates_token_count=10)
        client = ScriptedClient([FakeResponse(profile_json(), usage)])

        analysis = analyze_task(
            "Say hello.",
            client=client,  # type: ignore[arg-type]
            model_name="some-unlisted-model",
        )

        self.assertIsNone(analysis.estimated_cost_usd)

    def test_survives_a_response_without_usage_metadata(self) -> None:
        client = ScriptedClient([FakeResponse(profile_json())])

        analysis = analyze_task(
            "Say hello.",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        self.assertIsNone(analysis.usage)
        self.assertIsNone(analysis.estimated_cost_usd)
