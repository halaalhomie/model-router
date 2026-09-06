import unittest
from pathlib import Path

from google.genai import errors

from app.evaluation import (
    DEFAULT_DATASET,
    CaseOutcome,
    Dataset,
    EvaluationCase,
    StrategySummary,
    evaluate,
    format_report,
    load_dataset,
    parse_args,
    run_baseline_case,
    run_router_case,
)
from tests.fakes import SETTINGS, FakeResponse, ScriptedClient, profile_json


def case(case_id: str = "c1", expected: str = "factual") -> EvaluationCase:
    return EvaluationCase(
        id=case_id, text="What is the capital of Denmark?",
        expected_task_type=expected,  # type: ignore[arg-type]
    )


def outcome(**changes: object) -> CaseOutcome:
    values: dict[str, object] = {
        "case_id": "c1",
        "ok": True,
        "model_name": "demo-fast-model",
        "cost_usd": 0.001,
        "latency_ms": 100.0,
    }
    values.update(changes)
    return CaseOutcome(**values)  # type: ignore[arg-type]


class DatasetTests(unittest.TestCase):
    def test_the_shipped_dataset_loads_and_is_labelled(self) -> None:
        dataset = load_dataset(DEFAULT_DATASET)

        self.assertEqual(dataset.label, "routing-eval v1")
        self.assertTrue(dataset.cases)

    def test_every_case_has_a_valid_expected_task_type(self) -> None:
        """expected_task_type is typed as TaskType, so a typo in the JSON
        fails here rather than silently scoring every case as misrouted."""
        dataset = load_dataset(DEFAULT_DATASET)

        for c in dataset.cases:
            with self.subTest(case=c.id):
                self.assertTrue(c.text.strip())
                self.assertTrue(c.expected_task_type)

    def test_case_ids_are_unique(self) -> None:
        ids = [c.id for c in load_dataset(DEFAULT_DATASET).cases]

        self.assertEqual(len(ids), len(set(ids)))

    def test_rejects_an_unknown_task_type(self) -> None:
        bad = '{"name":"x","version":1,"cases":[{"id":"a","text":"t",' \
              '"expected_task_type":"nonsense"}]}'

        with self.assertRaises(Exception):
            Dataset.model_validate_json(bad)


class StrategySummaryTests(unittest.TestCase):
    def test_totals_cost_and_averages_latency_over_successes(self) -> None:
        summary = StrategySummary("router", [
            outcome(cost_usd=0.001, latency_ms=100),
            outcome(cost_usd=0.002, latency_ms=300),
        ])

        self.assertAlmostEqual(summary.total_cost_usd, 0.003)
        self.assertEqual(summary.avg_latency_ms, 200)

    def test_failures_are_excluded_from_cost_and_latency(self) -> None:
        summary = StrategySummary("router", [
            outcome(cost_usd=0.001, latency_ms=100),
            outcome(ok=False, cost_usd=None, latency_ms=9999, error="boom"),
        ])

        self.assertAlmostEqual(summary.total_cost_usd, 0.001)
        self.assertEqual(summary.avg_latency_ms, 100)
        self.assertEqual(summary.failures, 1)
        self.assertAlmostEqual(summary.failure_rate, 0.5)

    def test_total_cost_is_unknown_if_any_success_was_unpriced(self) -> None:
        """A partial total understates, and understating cost is the one
        error this whole exercise exists to avoid."""
        summary = StrategySummary("router", [
            outcome(cost_usd=0.001),
            outcome(cost_usd=None),
        ])

        self.assertIsNone(summary.total_cost_usd)

    def test_routing_accuracy_counts_only_graded_cases(self) -> None:
        summary = StrategySummary("router", [
            outcome(predicted_task_type="factual", expected_task_type="factual"),
            outcome(predicted_task_type="coding", expected_task_type="factual"),
            outcome(),  # baseline-style outcome, ungraded
        ])

        self.assertAlmostEqual(summary.routing_accuracy, 0.5)

    def test_routing_accuracy_is_none_without_predictions(self) -> None:
        self.assertIsNone(StrategySummary("baseline", [outcome()]).routing_accuracy)

    def test_counts_models_and_fallbacks(self) -> None:
        summary = StrategySummary("router", [
            outcome(model_name="a"),
            outcome(model_name="a", fallback_used=True),
            outcome(model_name="b"),
        ])

        self.assertEqual(summary.model_distribution, {"a": 2, "b": 1})
        self.assertEqual(summary.fallbacks, 1)

    def test_empty_summary_does_not_divide_by_zero(self) -> None:
        summary = StrategySummary("router")

        self.assertEqual(summary.avg_latency_ms, 0.0)
        self.assertEqual(summary.failure_rate, 0.0)
        self.assertIsNone(summary.routing_accuracy)


class RunCaseTests(unittest.TestCase):
    def test_router_case_records_classification_and_cost(self) -> None:
        client = ScriptedClient([
            FakeResponse(profile_json(task_type="factual")),
            FakeResponse("Copenhagen."),
        ])

        result = run_router_case(
            case(expected="factual"),
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.predicted_task_type, "factual")
        self.assertIs(result.classified_correctly, True)

    def test_router_case_marks_a_misclassification(self) -> None:
        client = ScriptedClient([
            FakeResponse(profile_json(task_type="coding")),
            FakeResponse("Copenhagen."),
        ])

        result = run_router_case(
            case(expected="factual"),
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertIs(result.classified_correctly, False)

    def test_router_case_records_a_failure_without_raising(self) -> None:
        """One bad case must not abort the whole run."""
        client = ScriptedClient([
            FakeResponse(profile_json(task_type="factual")),
            errors.ClientError(400, {"error": {"message": "bad"}}),
            errors.ClientError(400, {"error": {"message": "bad"}}),
        ])

        result = run_router_case(
            case(),
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
        )

        self.assertFalse(result.ok)
        self.assertIn("ClientError", result.error)

    def test_baseline_case_makes_one_call_with_no_classification(self) -> None:
        client = ScriptedClient([FakeResponse("Copenhagen.")])

        result = run_baseline_case(
            case(),
            client=client,  # type: ignore[arg-type]
            model_name="demo-reasoning-model",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.model_name, "demo-reasoning-model")
        self.assertIsNone(result.predicted_task_type)
        self.assertEqual(len(client.models.calls), 1)

    def test_baseline_case_records_a_failure(self) -> None:
        client = ScriptedClient([
            errors.ClientError(400, {"error": {"message": "bad"}})
        ])

        result = run_baseline_case(
            case(),
            client=client,  # type: ignore[arg-type]
            model_name="demo-reasoning-model",
        )

        self.assertFalse(result.ok)


class EvaluateTests(unittest.TestCase):
    def test_runs_both_strategies_over_every_case(self) -> None:
        dataset = Dataset(
            name="t", version=1, cases=[case("c1"), case("c2")]
        )
        # Per case: classify, routed execute, baseline execute.
        client = ScriptedClient([
            FakeResponse(profile_json(task_type="factual")),
            FakeResponse("A."),
            FakeResponse("A."),
            FakeResponse(profile_json(task_type="factual")),
            FakeResponse("B."),
            FakeResponse("B."),
        ])

        router, baseline = evaluate(
            dataset,
            client=client,  # type: ignore[arg-type]
            settings=SETTINGS,
            baseline_model="demo-reasoning-model",
        )

        self.assertEqual(len(router.outcomes), 2)
        self.assertEqual(len(baseline.outcomes), 2)
        self.assertEqual(len(client.models.calls), 6)

    def test_does_not_publish_events(self) -> None:
        """An evaluation run must not pour synthetic events into the topic
        real traffic uses. Asserted on the signature rather than the body:
        run_router_case takes no publisher, so it cannot publish."""
        import inspect

        params = inspect.signature(run_router_case).parameters

        self.assertNotIn("publisher", params)
        self.assertEqual(set(params) - {"case"}, {"client", "settings"})


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = Dataset(name="t", version=1, cases=[case("c1")])

    def report(self, router: StrategySummary, baseline: StrategySummary) -> str:
        return format_report(self.dataset, router, baseline, "big-model")

    def test_reports_a_saving_when_routing_is_cheaper(self) -> None:
        router = StrategySummary("router", [outcome(cost_usd=0.001)])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.004)])

        text = self.report(router, baseline)

        self.assertIn("cheaper", text)
        self.assertIn("75.0%", text)

    def test_says_plainly_when_routing_costs_more(self) -> None:
        router = StrategySummary("router", [outcome(cost_usd=0.005)])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.001)])

        self.assertIn("MORE EXPENSIVE", self.report(router, baseline))

    def test_always_disclaims_quality(self) -> None:
        """Cost saving is not quality. A router that always picked the
        worst model would score perfectly on every number here."""
        router = StrategySummary("router", [outcome(cost_usd=0.001)])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.004)])

        self.assertIn(
            "nothing about whether the answers were",
            self.report(router, baseline),
        )

    def test_refuses_to_compare_when_a_cost_is_unknown(self) -> None:
        router = StrategySummary("router", [outcome(cost_usd=None)])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.004)])

        self.assertIn("unavailable", self.report(router, baseline))

    def test_lists_misclassified_cases(self) -> None:
        router = StrategySummary("router", [
            outcome(predicted_task_type="coding", expected_task_type="factual")
        ])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.004)])

        text = self.report(router, baseline)

        self.assertIn("expected factual, got coding", text)

    def test_lists_failures(self) -> None:
        router = StrategySummary("router", [
            outcome(ok=False, cost_usd=None, error="ClientError: nope")
        ])
        baseline = StrategySummary("baseline", [outcome(cost_usd=0.004)])

        text = self.report(router, baseline)

        self.assertIn("Failures:", text)
        self.assertIn("ClientError: nope", text)


class ArgumentTests(unittest.TestCase):
    def test_defaults(self) -> None:
        args = parse_args([])

        self.assertIsNone(args.limit)
        self.assertIsNone(args.baseline_model)
        self.assertFalse(args.dry_run)
        self.assertEqual(args.dataset, DEFAULT_DATASET)

    def test_overrides(self) -> None:
        args = parse_args(
            ["--limit", "3", "--baseline-model", "m", "--dry-run",
             "--dataset", "other.json"]
        )

        self.assertEqual(args.limit, 3)
        self.assertEqual(args.baseline_model, "m")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.dataset, Path("other.json"))
