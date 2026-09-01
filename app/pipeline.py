from google import genai

from app.analyzer import analyze_task
from app.config import Settings
from app.executor import execute_request
from app.router import select_model
from app.schemas import PipelineResult


def run_pipeline(
    request_text: str,
    *,
    client: genai.Client,
    settings: Settings,
) -> PipelineResult:
    """Run one request through analyze -> route -> execute.

    This function is deliberately transport-agnostic. The CLI calls it today
    and the FastAPI handler will call the same function in Phase 3, so neither
    transport owns the orchestration logic.
    """
    profile = analyze_task(
        request_text,
        client=client,
        model_name=settings.analyzer_model,
    )

    decision = select_model(profile, settings.catalog)

    response = execute_request(
        request_text,
        client=client,
        model_name=decision.model_name,
    )

    return PipelineResult(
        request_text=request_text,
        profile=profile,
        decision=decision,
        response=response,
    )
