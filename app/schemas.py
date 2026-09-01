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
    """One model's reply, plus the usage data the evaluation platform needs."""

    model_name: str
    text: str
    usage: TokenUsage | None = None


class PipelineResult(BaseModel):
    """Everything one request produced, from classification to final reply."""

    request_text: str
    profile: TaskProfile
    decision: RoutingDecision
    response: ModelResponse
