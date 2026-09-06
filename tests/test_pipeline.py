import unittest
from dataclasses import replace

from google.genai import errors

from app.pipeline import run_pipeline
from app.schemas import (
    ModelResponse,
    PipelineResult,
    RoutingDecision,
    TaskProfile,
)
from tests.fakes import (
    SETTINGS,
    FakeEventPublisher,
    FakeResponse,
    FakeUsage,
    ScriptedClient,
    profile_json,
)


def make_priced_result(
    *, analyzer: float | None, answering: float | None
) -> PipelineResult:
    """A PipelineResult with only the two cost components set, for
    exercising total_cost_usd without running the pipeline."""
    return PipelineResult(
        request_id="req-cost",
        request_text="Say hello.",
        profile=TaskProfile(
            task_type="general",
            difficulty="low",
            reasoning_required="low",
            context_size="small",
            output_type="text",
            confidence=0.9,
        ),
        decision=RoutingDecision(model_name="m", reason="because"),
        response=ModelResponse(
            model_name="m", text="hi", estimated_cost_usd=answering
        ),
        latency_ms=1.0,
        analyzer_cost_usd=analyzer,
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

    def test_records_what_the_classifier_cost(self) -> None:
        """Routing overhead is a real cost a single-model baseline never
        pays, so the pipeline has to carry it, not discard it."""
        client = ScriptedClient(
            [
                FakeResponse(profile_json(), FakeUsage(100_000, 100_000)),
                FakeResponse("Answer.", FakeUsage(10, 10)),
            ]
        )
        settings = replace(SETTINGS, analyzer_model="gemini-3.5-flash-lite")

        result = run_pipeline(
            "Say hello.",
            client=client,  # type: ignore[arg-type]
            settings=settings,
        )

        self.assertEqual(result.analyzer_model, "gemini-3.5-flash-lite")
        assert result.analyzer_usage is not None
        self.assertEqual(result.analyzer_usage.prompt_tokens, 100_000)
        self.assertAlmostEqual(result.analyzer_cost_usd, 0.28)

    def test_total_cost_adds_routing_to_answering(self) -> None:
        result = make_priced_result(analyzer=0.001, answering=0.01)

        self.assertAlmostEqual(result.total_cost_usd, 0.011)

    def test_total_cost_is_unknown_when_either_half_is(self) -> None:
        """A total missing one component is not a smaller total, it is an
        unknown one -- the same reasoning as estimate_cost_usd."""
        self.assertIsNone(
            make_priced_result(analyzer=None, answering=0.01).total_cost_usd
        )
        self.assertIsNone(
            make_priced_result(analyzer=0.001, answering=None).total_cost_usd
        )

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
