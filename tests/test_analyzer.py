import json
import unittest

from pydantic import ValidationError

from app.analyzer import analyze_task


class FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.last_request: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.last_request = kwargs
        return FakeResponse(self.response_text)


class FakeClient:
    def __init__(self, response_text: str) -> None:
        self.models = FakeModels(response_text)


class TaskAnalyzerTests(unittest.TestCase):
    def test_returns_a_validated_task_profile(self) -> None:
        response_text = json.dumps(
            {
                "task_type": "coding",
                "difficulty": "medium",
                "reasoning_required": "medium",
                "context_size": "small",
                "output_type": "code",
                "confidence": 0.9,
            }
        )
        client = FakeClient(response_text)

        profile = analyze_task(
            "Fix a Python bug that reverses a list.",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-test-model",
        )

        self.assertEqual(profile.task_type, "coding")
        self.assertEqual(client.models.last_request["model"], "gemini-test-model")

    def test_rejects_an_unknown_task_type_from_the_model(self) -> None:
        response_text = json.dumps(
            {
                "task_type": "debugging",
                "difficulty": "medium",
                "reasoning_required": "medium",
                "context_size": "small",
                "output_type": "code",
                "confidence": 0.9,
            }
        )

        with self.assertRaises(ValidationError):
            analyze_task(
                "Fix a Python bug.",
                client=FakeClient(response_text),  # type: ignore[arg-type]
                model_name="gemini-test-model",
            )
