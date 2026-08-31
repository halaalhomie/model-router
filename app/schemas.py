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
