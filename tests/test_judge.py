import json
import unittest

from app.judge import (
    BASELINE,
    INCONSISTENT,
    ROUTER,
    TIE,
    JudgeVerdict,
    QualityOutcome,
    QualitySummary,
    compare_once,
    judge_pair,
    resolve,
)
from tests.fakes import FakeResponse, FakeUsage, ScriptedClient


def verdict_json(winner: str, reason: str = "because") -> str:
    return json.dumps({"winner": winner, "reason": reason})


def quality(result: str, **changes: object) -> QualityOutcome:
    values: dict[str, object] = {
        "case_id": "c1",
        "result": result,
        "reason": "because",
        "cost_usd": 0.001,
        "router_answer_chars": 100,
        "baseline_answer_chars": 100,
    }
    values.update(changes)
    return QualityOutcome(**values)  # type: ignore[arg-type]


class ResolveTests(unittest.TestCase):
    """The router's answer is A in the first pass and B in the second, so
    agreement means the winner letter flips between passes."""

    def test_router_wins_when_both_orders_agree(self) -> None:
        self.assertEqual(resolve("a", "b"), ROUTER)

    def test_baseline_wins_when_both_orders_agree(self) -> None:
        self.assertEqual(resolve("b", "a"), BASELINE)

    def test_mutual_tie_is_a_tie(self) -> None:
        self.assertEqual(resolve("tie", "tie"), TIE)

    def test_same_letter_twice_is_position_bias_not_a_win(self) -> None:
        """Picking "a" both times means the judge preferred the first slot,
        not an answer -- the whole reason both orders are run."""
        self.assertEqual(resolve("a", "a"), INCONSISTENT)
        self.assertEqual(resolve("b", "b"), INCONSISTENT)

    def test_a_tie_in_only_one_order_is_inconsistent(self) -> None:
        self.assertEqual(resolve("tie", "a"), INCONSISTENT)
        self.assertEqual(resolve("b", "tie"), INCONSISTENT)


class CompareOnceTests(unittest.TestCase):
    def test_returns_a_validated_verdict_and_its_cost(self) -> None:
        usage = FakeUsage(
            prompt_token_count=100_000, candidates_token_count=100_000
        )
        client = ScriptedClient([FakeResponse(verdict_json("a"), usage)])

        result, cost = compare_once(
            "Q?", "answer one", "answer two",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.winner, "a")
        # 0.1M in at $0.30 + 0.1M out at $2.50
        self.assertAlmostEqual(cost, 0.28)

    def test_rejects_a_winner_outside_the_allowed_set(self) -> None:
        client = ScriptedClient([FakeResponse(verdict_json("neither"))])

        with self.assertRaises(Exception):
            compare_once(
                "Q?", "one", "two",
                client=client,  # type: ignore[arg-type]
                model_name="gemini-3.5-flash-lite",
            )

    def test_shows_the_judge_both_answers_without_naming_the_models(
        self,
    ) -> None:
        client = ScriptedClient([FakeResponse(verdict_json("tie"))])

        compare_once(
            "Why is the sky blue?", "ROUTER_ANSWER", "BASELINE_ANSWER",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        prompt = client.models.calls[0]["contents"]
        self.assertIn("ROUTER_ANSWER", prompt)
        self.assertIn("BASELINE_ANSWER", prompt)
        self.assertIn("Why is the sky blue?", prompt)
        self.assertNotIn("gemini", prompt)


class JudgePairTests(unittest.TestCase):
    def test_judges_both_orders_and_swaps_the_answers(self) -> None:
        client = ScriptedClient([
            FakeResponse(verdict_json("a")),
            FakeResponse(verdict_json("b")),
        ])

        result = judge_pair(
            "c1", "Q?", "ROUTER_ANSWER", "BASELINE_ANSWER",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.result, ROUTER)
        first, second = (c["contents"] for c in client.models.calls)
        # Router's answer leads the first prompt and trails the second.
        self.assertLess(
            first.index("ROUTER_ANSWER"), first.index("BASELINE_ANSWER")
        )
        self.assertGreater(
            second.index("ROUTER_ANSWER"), second.index("BASELINE_ANSWER")
        )

    def test_flags_an_order_dependent_verdict(self) -> None:
        client = ScriptedClient([
            FakeResponse(verdict_json("a", "first is better")),
            FakeResponse(verdict_json("a", "first is better")),
        ])

        result = judge_pair(
            "c1", "Q?", "one", "two",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.result, INCONSISTENT)
        self.assertIn("order-dependent", result.reason)
        self.assertEqual(result.router_first_winner, "a")
        self.assertEqual(result.baseline_first_winner, "a")

    def test_records_answer_lengths_for_verbosity_checking(self) -> None:
        client = ScriptedClient([
            FakeResponse(verdict_json("a")),
            FakeResponse(verdict_json("b")),
        ])

        result = judge_pair(
            "c1", "Q?", "x" * 500, "y" * 100,
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        self.assertEqual(result.router_answer_chars, 500)
        self.assertEqual(result.baseline_answer_chars, 100)

    def test_sums_the_cost_of_both_passes(self) -> None:
        usage = FakeUsage(prompt_token_count=1_000_000)
        client = ScriptedClient([
            FakeResponse(verdict_json("a"), usage),
            FakeResponse(verdict_json("b"), usage),
        ])

        result = judge_pair(
            "c1", "Q?", "one", "two",
            client=client,  # type: ignore[arg-type]
            model_name="gemini-3.5-flash-lite",
        )

        # 1M input at $0.30, twice.
        self.assertAlmostEqual(result.cost_usd, 0.60)


class QualitySummaryTests(unittest.TestCase):
    def test_counts_each_verdict(self) -> None:
        summary = QualitySummary([
            quality(ROUTER), quality(ROUTER), quality(BASELINE),
            quality(TIE), quality(INCONSISTENT),
        ])

        self.assertEqual(summary.count(ROUTER), 2)
        self.assertEqual(summary.count(BASELINE), 1)
        self.assertEqual(summary.decided, 4)

    def test_consistency_is_the_share_that_survived_the_swap(self) -> None:
        summary = QualitySummary([
            quality(ROUTER), quality(TIE),
            quality(INCONSISTENT), quality(INCONSISTENT),
        ])

        self.assertAlmostEqual(summary.consistency, 0.5)

    def test_verbosity_gap_is_the_mean_length_difference(self) -> None:
        summary = QualitySummary([
            quality(ROUTER, router_answer_chars=300, baseline_answer_chars=100),
            quality(TIE, router_answer_chars=100, baseline_answer_chars=100),
        ])

        self.assertAlmostEqual(summary.verbosity_gap, 100)

    def test_total_cost_is_unknown_if_any_pair_was_unpriced(self) -> None:
        summary = QualitySummary([quality(ROUTER), quality(TIE, cost_usd=None)])

        self.assertIsNone(summary.total_cost_usd)

    def test_empty_summary_does_not_divide_by_zero(self) -> None:
        summary = QualitySummary([])

        self.assertIsNone(summary.consistency)
        self.assertIsNone(summary.verbosity_gap)
        self.assertEqual(summary.decided, 0)


class VerdictSchemaTests(unittest.TestCase):
    def test_accepts_the_three_allowed_winners(self) -> None:
        for winner in ("a", "b", "tie"):
            with self.subTest(winner=winner):
                self.assertEqual(
                    JudgeVerdict(winner=winner, reason="r").winner, winner
                )
