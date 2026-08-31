import unittest

from pydantic import ValidationError

from app.schemas import TaskProfile


class TaskProfileTests(unittest.TestCase):
    def test_accepts_a_valid_profile(self) -> None:
        profile = TaskProfile(
            task_type="coding",
            difficulty="medium",
            reasoning_required="medium",
            context_size="small",
            output_type="code",
            confidence=0.9,
        )

        self.assertEqual(profile.task_type, "coding")
        self.assertEqual(profile.confidence, 0.9)

    def test_rejects_confidence_outside_the_allowed_range(self) -> None:
        with self.assertRaises(ValidationError):
            TaskProfile(
                task_type="coding",
                difficulty="medium",
                reasoning_required="medium",
                context_size="small",
                output_type="code",
                confidence=1.1,
            )
