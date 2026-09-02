import unittest

from google.genai import errors

from app.pipeline import run_pipeline
from tests.fakes import (
    SETTINGS,
    FakeEventPublisher,
    FakeResponse,
    FakeUsage,
    ScriptedClient,
    profile_json,
)


class RunPipelineTests(unittest.TestCase):
    def test_analyzes_routes_and_executes_in_order(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(
                    profile_json(task_type="coding", output_type="code")
                ),
                FakeResponse("def solve(): ...", FakeUsage(10, 20, 30)),
            ]
        )

        result = run_pipeline(
            "Write a Python function that reverses a list.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertEqual(result.profile.task_type, "coding")
        self.assertEqual(result.decision.model_name, "demo-code-model")
        self.assertEqual(result.response.text, "def solve(): ...")
        self.assertEqual(
            result.request_text,
            "Write a Python function that reverses a list.",
        )

    def test_classifier_uses_the_analyzer_model(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding")),
                FakeResponse("Answer."),
            ]
        )

        run_pipeline(
            "Fix this bug.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        analyzer_call, executor_call = client.models.calls
        self.assertEqual(analyzer_call["model"], "demo-analyzer-model")
        self.assertEqual(executor_call["model"], "demo-code-model")

    def test_sends_the_original_request_to_the_routed_model(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json(difficulty="high")),
                FakeResponse("Answer."),
            ]
        )

        run_pipeline(
            "Plan a migration strategy.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        executor_call = client.models.calls[1]
        self.assertEqual(
            executor_call["contents"], "Plan a migration strategy."
        )
        self.assertEqual(executor_call["model"], "demo-reasoning-model")

    def test_carries_token_usage_through_to_the_result(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json()),
                FakeResponse("Answer.", FakeUsage(5, 7, 12)),
            ]
        )

        result = run_pipeline(
            "Say hello.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        assert result.response.usage is not None
        self.assertEqual(result.response.usage.total_tokens, 12)

    def test_stops_before_executing_when_the_profile_is_invalid(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="debugging")),
                FakeResponse("should never be reached"),
            ]
        )

        with self.assertRaises(Exception):
            run_pipeline(
                "Fix this bug.",
                client=client,  # type: ignore[arg-type]
                settings=SETTINGS,
            )

        self.assertEqual(len(client.models.calls), 1)


class RunPipelineFallbackTests(unittest.TestCase):
    def test_falls_back_to_the_fast_model_when_the_routed_model_fails(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding")),
                errors.ClientError(400, {"error": {"message": "bad request"}}),
                FakeResponse("Fallback answer."),
            ]
        )

        result = run_pipeline(
            "Fix this bug.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertEqual(result.response.text, "Fallback answer.")
        self.assertEqual(result.response.model_name, "demo-fast-model")
        self.assertTrue(result.response.fallback_used)
        self.assertEqual(result.response.original_model, "demo-code-model")
        # decision still records what the router actually chose
        self.assertEqual(result.decision.model_name, "demo-code-model")

        second_call = client.models.calls[1]
        third_call = client.models.calls[2]
        self.assertEqual(second_call["model"], "demo-code-model")
        self.assertEqual(third_call["model"], "demo-fast-model")

    def test_falls_back_when_the_routed_model_returns_an_empty_reply(self) -> None:
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding")),
                FakeResponse(""),
                FakeResponse("Fallback answer."),
            ]
        )

        result = run_pipeline(
            "Fix this bug.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertTrue(result.response.fallback_used)
        self.assertEqual(result.response.text, "Fallback answer.")

    def test_does_not_fall_back_again_when_already_routed_to_fast_model(
        self,
    ) -> None:
        """fast_model is the fallback target -- if it's also the routed
        model and it fails, there is nowhere safer left to fall back to."""
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="factual")),
                errors.ClientError(400, {"error": {"message": "bad request"}}),
            ]
        )

        with self.assertRaises(errors.ClientError):
            run_pipeline(
                "What is 2+2?",
                client=client,  # type: ignore[arg-type]
                settings=SETTINGS,
            )

        self.assertEqual(len(client.models.calls), 2)  # no third, fallback call

    def test_does_not_fall_back_on_success(self) -> None:
        client = ScriptedClient(
            [FakeResponse(profile_json(task_type="coding")), FakeResponse("ok")]
        )

        result = run_pipeline(
            "Fix this bug.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertFalse(result.response.fallback_used)
        self.assertIsNone(result.response.original_model)


class PipelineTelemetryTests(unittest.TestCase):
    def scripted(self) -> ScriptedClient:
        return ScriptedClient(
            [FakeResponse(profile_json()), FakeResponse("Answer.")]
        )

    def test_assigns_a_unique_request_id_per_run(self) -> None:
        first = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
        )
        second = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertTrue(first.request_id)
        self.assertNotEqual(first.request_id, second.request_id)

    def test_records_latency(self) -> None:
        result = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertGreater(result.latency_ms, 0)

    def test_publishes_the_result_when_a_publisher_is_given(self) -> None:
        publisher = FakeEventPublisher()

        result = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
            publisher=publisher,
        )

        self.assertEqual(publisher.published, [result])

    def test_runs_without_a_publisher(self) -> None:
        result = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertEqual(result.response.text, "Answer.")

    def test_a_broken_publisher_does_not_fail_the_request(self) -> None:
        """The whole point of keeping Kafka off the synchronous path: the
        caller still gets their answer even if publishing blows up."""
        publisher = FakeEventPublisher(error=RuntimeError("broker down"))

        result = run_pipeline(
            "Say hello.",
            client=self.scripted(),  # type: ignore[arg-type]
            settings=SETTINGS,
            publisher=publisher,
        )

        self.assertEqual(result.response.text, "Answer.")

    def test_publishes_only_after_the_response_is_complete(self) -> None:
        """The event carries the finished result, fallback flags included --
        so a consumer never sees a half-built request."""
        client = ScriptedClient(
            [
                FakeResponse(profile_json(task_type="coding")),
                errors.ClientError(400, {"error": {"message": "bad request"}}),
                FakeResponse("Fallback answer."),
            ]
        )
        publisher = FakeEventPublisher()

        run_pipeline(
            "Fix this bug.",
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
            publisher=publisher,
        )

        published = publisher.published[0]
        self.assertTrue(published.response.fallback_used)
        self.assertEqual(published.response.text, "Fallback answer.")
