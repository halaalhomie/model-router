"""Tests for the HTTP transport.

Deliberately does NOT use `with TestClient(app) as client:` -- that form
runs FastAPI's lifespan startup, which would call the real load_settings()
and depend on the developer's own .env file being present and valid. Using
TestClient(app) directly skips lifespan entirely (verified: app.state is
never populated), so /health works standalone and /route is only safe
because every test below overrides get_settings/get_client -- the real
ones, which read app.state, are never actually called.
"""

import json
import unittest

from fastapi.testclient import TestClient
from google.genai import errors

from app.api import app, get_client, get_settings
from app.config import Settings
from app.schemas import ModelCatalog
from tests.fakes import FakeResponse, ScriptedClient


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
)


def profile_json(**changes: object) -> str:
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


class ApiTestCase(unittest.TestCase):
    """Wires a ScriptedClient in for get_client/get_settings on every test,
    via FastAPI's dependency_overrides -- the standard way to swap out
    what a Depends() resolves to during a test, without touching app.state
    or lifespan at all."""

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(app.dependency_overrides.clear)

    def override(self, scripted: ScriptedClient) -> None:
        app.dependency_overrides[get_settings] = lambda: SETTINGS
        app.dependency_overrides[get_client] = lambda: scripted


class HealthTests(unittest.TestCase):
    def test_returns_ok_with_no_dependencies_needed(self) -> None:
        response = TestClient(app).get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


class RouteEndpointTests(ApiTestCase):
    def test_returns_the_pipeline_result_as_json(self) -> None:
        scripted = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding", output_type="code")),
                FakeResponse("def solve(): ..."),
            ]
        )
        self.override(scripted)

        response = self.client.post("/route", json={"text": "Fix a bug."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["profile"]["task_type"], "coding")
        self.assertEqual(body["decision"]["model_name"], "demo-code-model")
        self.assertEqual(body["response"]["text"], "def solve(): ...")

    def test_rejects_blank_text_before_touching_the_model(self) -> None:
        scripted = ScriptedClient([])  # would raise AssertionError if called
        self.override(scripted)

        response = self.client.post("/route", json={"text": "   "})

        self.assertEqual(response.status_code, 422)

    def test_rejects_a_request_body_missing_the_text_field(self) -> None:
        self.override(ScriptedClient([]))

        response = self.client.post("/route", json={})

        self.assertEqual(response.status_code, 422)

    def test_returns_502_when_the_model_and_fallback_both_fail(self) -> None:
        """factual routes to fast_model directly -- when fast_model fails
        there is no further fallback to attempt, so run_pipeline's own
        exception should surface as a 502, not crash the server as a 500."""
        scripted = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="factual")),
                errors.ClientError(400, {"error": {"message": "bad request"}}),
            ]
        )
        self.override(scripted)

        response = self.client.post("/route", json={"text": "What is 2+2?"})

        self.assertEqual(response.status_code, 502)
        self.assertIn("fallback did not recover", response.json()["detail"])

    def test_reports_fallback_in_the_response_body(self) -> None:
        scripted = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding")),
                errors.ClientError(400, {"error": {"message": "bad request"}}),
                FakeResponse("Fallback answer."),
            ]
        )
        self.override(scripted)

        response = self.client.post("/route", json={"text": "Fix this bug."})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["response"]["fallback_used"])
        self.assertEqual(body["response"]["original_model"], "demo-code-model")
