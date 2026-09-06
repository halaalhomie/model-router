import time
import uuid

from google import genai
from google.genai import errors as genai_errors
import httpx2

from app.analyzer import analyze_task
from app.config import Settings
from app.events import EventPublisher, publish_safely
from app.executor import execute_request
from app.router import select_model
from app.schemas import PipelineResult


# What counts as "the routed model failed" for fallback purposes: the
# provider errors execute_request's retries could not resolve, plus its own
# ValueError for an empty reply. A bare `except Exception` would also catch
# programming errors (e.g. a bad kwarg) and silently mask them as a model
# failure, which is worse than letting the pipeline crash on those.
EXECUTION_FAILURES = (genai_errors.APIError, httpx2.TransportError, ValueError)


def run_pipeline(
    request_text: str,
    *,
    client: genai.Client,
    settings: Settings,
    publisher: EventPublisher | None = None,
) -> PipelineResult:
    """Run one request through analyze -> route -> execute -> publish.

    This function is deliberately transport-agnostic. The CLI and the
    FastAPI handler both call this same function, so neither transport
    owns the orchestration logic.

    If the routed model still fails after execute_request's own retries
    (a 429/5xx that didn't clear, or an empty reply), the request falls back
    to the catalog's fast model once rather than failing outright -- fast_model
    is the one tier that has been reliable in practice (see README "Model
    availability"). A request already routed to fast_model has nowhere safer
    to fall back to, so its failure propagates.

    publisher is optional and best-effort: see app/events.py for why a
    Kafka failure must never become a request failure.
    """
    request_id = str(uuid.uuid4())
    started_at = time.perf_counter()

    analysis = analyze_task(
        request_text,
        client=client,
        model_name=settings.analyzer_model,
    )
    profile = analysis.profile

    decision = select_model(profile, settings.catalog)

    try:
        response = execute_request(
            request_text,
            client=client,
            model_name=decision.model_name,
        )
    except EXECUTION_FAILURES:
        if decision.model_name == settings.catalog.fast_model:
            raise

        response = execute_request(
            request_text,
            client=client,
            model_name=settings.catalog.fast_model,
        )
        response = response.model_copy(
            update={"fallback_used": True, "original_model": decision.model_name}
        )

    result = PipelineResult(
        request_id=request_id,
        request_text=request_text,
        profile=profile,
        decision=decision,
        response=response,
        latency_ms=(time.perf_counter() - started_at) * 1000,
        analyzer_model=analysis.model_name,
        analyzer_usage=analysis.usage,
        analyzer_cost_usd=analysis.estimated_cost_usd,
    )

    publish_safely(publisher, result)

    return result
