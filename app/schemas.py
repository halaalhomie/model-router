from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal[
    "coding",
    "reasoning",
    "factual",
    "math",
    "data_analysis",
    "research",
    "summarization",
    "translation",
    "creative_writing",
    "information_extraction",
    "classification",
    "planning",
    "general",
]

Difficulty = Literal["low", "medium", "high"]
ContextSize = Literal["small", "medium", "large"]
OutputType = Literal["text", "code", "json", "table", "list", "explanation"]


class TaskProfile(BaseModel):
    """A validated description of the work requested from the router."""

    task_type: TaskType
    difficulty: Difficulty
    reasoning_required: Difficulty
    context_size: ContextSize
    output_type: OutputType
    confidence: float = Field(ge=0.0, le=1.0)


class ModelCatalog(BaseModel):
    """Model names grouped by the capability each routing rule needs."""

    fast_model: str
    code_model: str
    reasoning_model: str
    long_context_model: str


class RoutingDecision(BaseModel):
    """The router's chosen model and its human-readable rationale."""

    model_name: str
    reason: str


class TokenUsage(BaseModel):
    """Token counts reported by the provider for one model call.

    Reasoning models bill hidden "thinking" tokens as output, which is why
    total_tokens is usually larger than prompt_tokens + output_tokens. Cost
    calculations must use thinking_tokens or they will understate the bill.
    """

    prompt_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    thinking_tokens: int = Field(ge=0, default=0)
    total_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """One model's reply, plus the usage data the evaluation platform needs.

    model_name is whichever model actually produced `text` -- if the routed
    model failed and the pipeline fell back, this is the fallback model, not
    the one RoutingDecision originally chose. original_model preserves that
    original choice so telemetry can tell "routed correctly but the model
    was down" apart from "routed to the wrong model".
    """

    model_name: str
    text: str
    usage: TokenUsage | None = None
    fallback_used: bool = False
    original_model: str | None = None
    # None means "could not be priced" (unknown model, or no usage
    # reported), never "free" -- see app/pricing.py. Stored alongside the
    # raw token counts rather than derived later, so an event keeps the
    # cost as calculated at the time even if prices change afterwards.
    estimated_cost_usd: float | None = None


class AnalysisResult(BaseModel):
    """The classifier's verdict, plus what the classification itself cost.

    The router pays for this call on every request; a baseline that always
    uses one model does not. Discarding its cost would make any comparison
    between them flatter the router by exactly this much.
    """

    profile: TaskProfile
    model_name: str
    usage: TokenUsage | None = None
    estimated_cost_usd: float | None = None


class PipelineResult(BaseModel):
    """Everything one request produced, from classification to final reply.

    request_id and latency_ms exist for telemetry, not for the pipeline's
    own logic: request_id is what a Kafka event, and later a Postgres row,
    correlates back to one request; latency_ms is the first real evaluation
    metric this project collects.

    Cost is split in two on purpose. response.estimated_cost_usd is what
    answering cost; analyzer_cost_usd is the routing overhead paid to decide
    who should answer. total_cost_usd adds them, and only that total is a
    fair number to compare against a non-routing baseline.
    """

    request_id: str
    request_text: str
    profile: TaskProfile
    decision: RoutingDecision
    response: ModelResponse
    latency_ms: float
    # Added after events were already in the topic, so both default --
    # older events parse as "analyzer cost unknown" instead of failing.
    analyzer_model: str | None = None
    analyzer_usage: TokenUsage | None = None
    analyzer_cost_usd: float | None = None

    @property
    def total_cost_usd(self) -> float | None:
        """Classification plus execution, or None if either is unpriced.

        None rather than a partial sum: a total missing one of its two
        components is not a smaller total, it is an unknown one, and
        reporting it as a number invites exactly the undercount that
        estimate_cost_usd returns None to prevent.
        """
        answering = self.response.estimated_cost_usd
        if answering is None or self.analyzer_cost_usd is None:
            return None
        return answering + self.analyzer_cost_usd
