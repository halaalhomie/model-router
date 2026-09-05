import logging
import unittest

from app.pricing import PRICING, ModelPricing, estimate_cost_usd
from app.schemas import TokenUsage


class EstimateCostTests(unittest.TestCase):
    def test_prices_input_and_output_separately(self) -> None:
        # gemini-3.5-flash-lite: $0.30 in / $2.50 out per 1M tokens.
        # 1M input + 1M output = 0.30 + 2.50 = 2.80
        usage = TokenUsage(
            prompt_tokens=1_000_000,
            output_tokens=1_000_000,
            total_tokens=2_000_000,
        )

        cost = estimate_cost_usd("gemini-3.5-flash-lite", usage)

        self.assertAlmostEqual(cost, 2.80)

    def test_bills_thinking_tokens_at_the_output_rate(self) -> None:
        """Google's pricing page bills thinking tokens as output. Ignoring
        them understates reasoning-model costs badly, since they routinely
        exceed the visible output."""
        with_thinking = TokenUsage(
            prompt_tokens=0,
            output_tokens=500_000,
            thinking_tokens=500_000,
            total_tokens=1_000_000,
        )
        without_thinking = TokenUsage(
            prompt_tokens=0,
            output_tokens=500_000,
            total_tokens=500_000,
        )

        billed = estimate_cost_usd("gemini-3.5-flash-lite", with_thinking)
        ignored = estimate_cost_usd("gemini-3.5-flash-lite", without_thinking)

        self.assertAlmostEqual(billed, 2.50)  # 1M output-billed tokens
        self.assertAlmostEqual(ignored, 1.25)
        self.assertGreater(billed, ignored)

    def test_uses_real_measured_usage(self) -> None:
        """The live gemini-3.6-flash call from the README's measurements:
        11 in, 841 out, 621 thinking. At $0.75/$3.75 per 1M that is
        (11*0.75 + 1462*3.75) / 1e6."""
        usage = TokenUsage(
            prompt_tokens=11,
            output_tokens=841,
            thinking_tokens=621,
            total_tokens=1473,
        )

        cost = estimate_cost_usd("gemini-3.6-flash", usage)

        expected = (11 * 0.75 + (841 + 621) * 3.75) / 1_000_000
        self.assertAlmostEqual(cost, expected)

    def test_returns_none_for_an_unpriced_model(self) -> None:
        """None, not 0.0: an unknown model still costs something, and a
        silent zero would deflate every total built on top of it."""
        usage = TokenUsage(prompt_tokens=10, output_tokens=10, total_tokens=20)

        with self.assertLogs("app.pricing", level=logging.WARNING):
            cost = estimate_cost_usd("some-unlisted-model", usage)

        self.assertIsNone(cost)

    def test_returns_none_without_usage(self) -> None:
        self.assertIsNone(estimate_cost_usd("gemini-3.5-flash-lite", None))

    def test_zero_usage_costs_zero(self) -> None:
        usage = TokenUsage(prompt_tokens=0, output_tokens=0, total_tokens=0)

        self.assertEqual(estimate_cost_usd("gemini-3.5-flash-lite", usage), 0.0)


class PricingTableTests(unittest.TestCase):
    def test_covers_every_model_the_default_catalog_routes_to(self) -> None:
        """.env.example's catalog must be fully priced, or evaluation
        silently loses requests to the unpriced bucket."""
        for model in (
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.6-flash",
        ):
            with self.subTest(model=model):
                self.assertIn(model, PRICING)

    def test_output_is_dearer_than_input_everywhere(self) -> None:
        for model, pricing in PRICING.items():
            with self.subTest(model=model):
                self.assertGreater(
                    pricing.output_per_1m_usd, pricing.input_per_1m_usd
                )

    def test_the_reasoning_tier_is_not_the_priciest_per_token(self) -> None:
        """A deliberate check on an assumption the router quietly makes:
        gemini-3.6-flash is the heavier tier but is cheaper per token than
        gemini-3.5-flash. Tier by capability != tier by price."""
        reasoning = PRICING["gemini-3.6-flash"]
        code = PRICING["gemini-3.5-flash"]

        self.assertLess(reasoning.output_per_1m_usd, code.output_per_1m_usd)

    def test_pricing_is_immutable(self) -> None:
        with self.assertRaises(Exception):
            PRICING["gemini-3.5-flash"].input_per_1m_usd = 999  # type: ignore

    def test_model_pricing_holds_both_rates(self) -> None:
        pricing = ModelPricing(1.0, 2.0)

        self.assertEqual(pricing.input_per_1m_usd, 1.0)
        self.assertEqual(pricing.output_per_1m_usd, 2.0)
