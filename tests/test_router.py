import unittest

from app.router import select_model
from app.schemas import ModelCatalog, TaskProfile


CATALOG = ModelCatalog(
    fast_model="demo-fast-model",
    code_model="demo-code-model",
    reasoning_model="demo-reasoning-model",
    long_context_model="demo-long-context-model",
)


def make_profile(**changes: str | float) -> TaskProfile:
    values: dict[str, str | float] = {
        "task_type": "general",
        "difficulty": "low",
        "reasoning_required": "low",
        "context_size": "small",
        "output_type": "text",
        "confidence": 0.9,
    }
    values.update(changes)
    return TaskProfile(**values)


class RouterTests(unittest.TestCase):
    def test_large_context_has_highest_priority(self) -> None:
        decision = select_model(
            make_profile(task_type="coding", context_size="large"), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-long-context-model")

    def test_coding_uses_the_code_model(self) -> None:
        decision = select_model(make_profile(task_type="coding"), CATALOG)

        self.assertEqual(decision.model_name, "demo-code-model")

    def test_high_reasoning_uses_the_reasoning_model(self) -> None:
        decision = select_model(
            make_profile(task_type="planning", reasoning_required="high"), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-reasoning-model")

    def test_simple_request_uses_the_fast_model(self) -> None:
        decision = select_model(make_profile(task_type="factual"), CATALOG)

        self.assertEqual(decision.model_name, "demo-fast-model")


class ConfidenceAwareRoutingTests(unittest.TestCase):
    """confidence describes the whole classification, not one field, so a
    low score should override every rule above -- including large_context,
    which otherwise has top priority."""

    def test_low_confidence_overrides_large_context(self) -> None:
        decision = select_model(
            make_profile(context_size="large", confidence=0.4), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-reasoning-model")

    def test_low_confidence_overrides_coding(self) -> None:
        decision = select_model(
            make_profile(task_type="coding", confidence=0.4), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-reasoning-model")

    def test_low_confidence_overrides_the_fast_model_default(self) -> None:
        decision = select_model(
            make_profile(task_type="factual", confidence=0.4), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-reasoning-model")
        self.assertIn("0.40", decision.reason)

    def test_confidence_exactly_at_the_threshold_is_trusted(self) -> None:
        """The threshold is an exclusive lower bound: AT 0.6, the profile
        is trusted and normal routing applies."""
        decision = select_model(
            make_profile(task_type="coding", confidence=0.6), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-code-model")

    def test_confidence_just_below_the_threshold_is_not_trusted(self) -> None:
        decision = select_model(
            make_profile(task_type="coding", confidence=0.59), CATALOG
        )

        self.assertEqual(decision.model_name, "demo-reasoning-model")
