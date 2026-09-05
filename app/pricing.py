"""Per-model prices, and the cost of one model call.

Prices are reference data about a model's identity, not deployment
configuration, so they live here in version control rather than in .env --
you want a diff and a review when a price changes, and you want the same
numbers on every machine. Which models to *use* stays in .env; what they
*cost* lives here.

Source: https://ai.google.dev/gemini-api/docs/pricing (paid tier, per 1M
tokens), checked 2026-09-05.

Two things worth knowing about these numbers:

1. Thinking tokens are billed as output. Google's pricing page says so
   explicitly ("Output price (including thinking tokens)"), which is why
   TokenUsage tracks thinking_tokens separately -- a cost calculation that
   ignores them understates the bill badly on reasoning models, where they
   routinely exceed the visible output.

2. gemini-3.6-flash is *cheaper* per token than gemini-3.5-flash
   ($0.75/$3.75 vs $1.50/$9.00), even though the router treats it as the
   heavier "reasoning" tier. Tier ordering by capability does not imply
   tier ordering by price. Whether routing to it actually costs more per
   request depends on how many thinking tokens it burns -- which is the
   sort of question this data exists to answer instead of assume.

These are what a request *would* cost at paid-tier rates. This project's
key is free-tier, so nothing here is an actual bill; it is a modelled cost,
which is the right basis for comparing routing strategies but must not be
reported as money genuinely spent.
"""

import logging
from dataclasses import dataclass

from app.schemas import TokenUsage


logger = logging.getLogger(__name__)

TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m_usd: float
    output_per_1m_usd: float


PRICING: dict[str, ModelPricing] = {
    "gemini-3.5-flash-lite": ModelPricing(0.30, 2.50),
    "gemini-3.5-flash": ModelPricing(1.50, 9.00),
    "gemini-3.6-flash": ModelPricing(0.75, 3.75),
    "gemini-3.7-flash": ModelPricing(0.75, 3.75),
}


def estimate_cost_usd(
    model_name: str, usage: TokenUsage | None
) -> float | None:
    """Cost of one call, or None when it genuinely cannot be priced.

    None rather than 0.0 on purpose: a model missing from PRICING costs
    something, we just don't know what. Returning 0.0 would silently
    understate every total built on top of it, and an evaluation that
    quietly undercounts cost is worse than one that admits a gap.
    """
    if usage is None:
        return None

    pricing = PRICING.get(model_name)
    if pricing is None:
        logger.warning(
            "No pricing for model %r -- its cost is omitted rather than "
            "counted as zero. Add it to app/pricing.py.",
            model_name,
        )
        return None

    # Thinking tokens bill as output, so they belong on the output side of
    # this sum, not left out of it.
    billable_output = usage.output_tokens + usage.thinking_tokens

    input_cost = usage.prompt_tokens * pricing.input_per_1m_usd
    output_cost = billable_output * pricing.output_per_1m_usd
    return (input_cost + output_cost) / TOKENS_PER_PRICE_UNIT
