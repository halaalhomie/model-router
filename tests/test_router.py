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
