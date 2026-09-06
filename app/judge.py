"""Judges whether the cheaper answer was actually as good.

Cost and latency are objective; quality is not. Without this, the
evaluation harness can only say routing is *cheaper* -- and a router that
always picked the worst model would score perfectly on cheapness. This
module is what stops "cheaper" being mistaken for "better".

Design, and why each part is there
----------------------------------
**Pairwise, not a 1-5 score.** The question is comparative -- did the
routed answer lose anything against the baseline's? -- and LLM judges are
substantially more reliable choosing between two answers than assigning an
absolute score, which drifts between runs and skews lenient.

**Every pair is judged twice, with the order swapped.** Judges carry a
position bias: they favour whichever answer they see first (or second),
independent of content. Running A/B and then B/A and requiring the two
passes to agree turns that bias from a silent thumb on the scale into a
visible disagreement. A pair the judge scores inconsistently is reported
as INCONSISTENT rather than resolved by a coin flip, because the honest
reading is "this judge cannot reliably separate these two answers".

**The judge is blind.** It never learns which model wrote which answer,
so it cannot prefer a name.

Biases this does NOT solve, and must be read alongside every result
------------------------------------------------------------------
- **Self-preference.** The strongest model reachable on this key is
  gemini-3.6-flash, which is also one of the models being judged. A model
  grading its own family is not a neutral referee. Escaping this needs a
  judge from a different provider, which this key cannot reach.
- **Verbosity.** Judges tend to reward longer answers. The prompt tells
  the judge to ignore length, which helps and does not cure it; answer
  lengths are recorded so a correlation is at least detectable.
- **It is one judge, not a panel.** A single model's taste is not ground
  truth. Human review remains the only real gold standard here.
"""

import logging
from dataclasses import dataclass
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.pricing import estimate_cost_usd
from app.retry import call_with_retry
from app.usage import read_usage


logger = logging.getLogger(__name__)

JUDGE_INSTRUCTIONS = """
You are comparing two answers to the same request.

Judge only how well each answer serves the request: correctness,
completeness, and clarity. A longer answer is not automatically better --
ignore length except where it genuinely affects those three. Ignore
differences in formatting or tone. You are not told which system produced
either answer; do not speculate.

Reply with the winner ("a", "b", or "tie") and one sentence of reasoning.
Choose "tie" when neither answer is meaningfully better, not as a way to
avoid deciding.
""".strip()

# What the two swapped passes together mean.
ROUTER = "router"
BASELINE = "baseline"
TIE = "tie"
INCONSISTENT = "inconsistent"


class JudgeVerdict(BaseModel):
    """One judgement of one ordering. Structured output, validated -- the
    same pattern the analyzer uses, for the same reason: a free-text reply
    saying "well, it depends" is not something we can count."""

    winner: Literal["a", "b", "tie"]
    reason: str


@dataclass
class QualityOutcome:
    """The consistency-checked result for one pair of answers."""

    case_id: str
    result: str  # ROUTER | BASELINE | TIE | INCONSISTENT
    reason: str
    cost_usd: float | None
    router_answer_chars: int
    baseline_answer_chars: int
    # The raw winner from each ordering, kept so an inconsistent verdict
    # can be inspected rather than just counted.
    router_first_winner: str | None = None
    baseline_first_winner: str | None = None


def compare_once(
    question: str,
    answer_a: str,
    answer_b: str,
    *,
    client: genai.Client,
    model_name: str,
) -> tuple[JudgeVerdict, float | None]:
    """One judgement of one ordering, with what it cost."""
    prompt = (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"Request:\n{question}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}"
    )

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JudgeVerdict,
            ),
        )
    )

    if not response.text:
        raise ValueError("The judge returned no text.")

    usage = read_usage(response)
    return (
        JudgeVerdict.model_validate_json(response.text),
        estimate_cost_usd(model_name, usage),
    )


def resolve(router_first: str, baseline_first: str) -> str:
    """Combine the two orderings into one verdict.

    In the first pass the router's answer is A; in the second it is B. So
    the passes agree that the router won only when the first says "a" and
    the second says "b". Anything else that is not a mutual tie means the
    judge changed its mind when the order changed -- which is exactly the
    position bias this exists to catch.
    """
    if router_first == "a" and baseline_first == "b":
        return ROUTER
    if router_first == "b" and baseline_first == "a":
        return BASELINE
    if router_first == "tie" and baseline_first == "tie":
        return TIE
    return INCONSISTENT


def judge_pair(
    case_id: str,
    question: str,
    router_answer: str,
    baseline_answer: str,
    *,
    client: genai.Client,
    model_name: str,
) -> QualityOutcome:
    """Judge one pair in both orders and report the combined verdict."""
    first, cost_a = compare_once(
        question, router_answer, baseline_answer,
        client=client, model_name=model_name,
    )
    second, cost_b = compare_once(
        question, baseline_answer, router_answer,
        client=client, model_name=model_name,
    )

    total_cost = (
        None if cost_a is None or cost_b is None else cost_a + cost_b
    )
    result = resolve(first.winner, second.winner)

    return QualityOutcome(
        case_id=case_id,
        result=result,
        reason=first.reason if result != INCONSISTENT else (
            f"order-dependent: {first.reason} / {second.reason}"
        ),
        cost_usd=total_cost,
        router_answer_chars=len(router_answer),
        baseline_answer_chars=len(baseline_answer),
        router_first_winner=first.winner,
        baseline_first_winner=second.winner,
    )


@dataclass
class QualitySummary:
    outcomes: list[QualityOutcome]

    def count(self, result: str) -> int:
        return sum(1 for o in self.outcomes if o.result == result)

    @property
    def total_cost_usd(self) -> float | None:
        if any(o.cost_usd is None for o in self.outcomes):
            return None
        return sum(o.cost_usd for o in self.outcomes)

    @property
    def decided(self) -> int:
        """Pairs the judge ranked the same way in both orderings."""
        return self.count(ROUTER) + self.count(BASELINE) + self.count(TIE)

    @property
    def consistency(self) -> float | None:
        """How often the judge survived its own order swap. A low number
        means the verdicts below are largely noise."""
        if not self.outcomes:
            return None
        return self.decided / len(self.outcomes)

    @property
    def verbosity_gap(self) -> float | None:
        """Mean router-minus-baseline answer length, in characters.

        A large positive gap alongside a router win is a reason to suspect
        the judge rewarded length rather than quality.
        """
        if not self.outcomes:
            return None
        gaps = [
            o.router_answer_chars - o.baseline_answer_chars
            for o in self.outcomes
        ]
        return sum(gaps) / len(gaps)
