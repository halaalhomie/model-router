from google import genai
from google.genai import types

from app.pricing import estimate_cost_usd
from app.retry import call_with_retry
from app.schemas import AnalysisResult, TaskProfile
from app.usage import read_usage


ANALYSIS_INSTRUCTIONS = """
You are a task classifier for an LLM model router.
Analyze the user's request and return one TaskProfile as JSON.

Choose the single primary task_type. Treat implementation and debugging requests
as coding. Use general only when no more specific task type applies.
Estimate difficulty, required reasoning, context size, expected output type, and
your confidence in the classification. Do not include an explanation or Markdown.
""".strip()


def analyze_task(
    request_text: str,
    *,
    client: genai.Client,
    model_name: str,
) -> AnalysisResult:
    """Classify one request, reporting what the classification cost."""
    if not request_text.strip():
        raise ValueError("request_text must not be empty.")

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=model_name,
            contents=f"{ANALYSIS_INSTRUCTIONS}\n\nUser request:\n{request_text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TaskProfile,
            ),
        )
    )

    if not response.text:
        raise ValueError("The task analyzer returned no text.")

    usage = read_usage(response)
    return AnalysisResult(
        profile=TaskProfile.model_validate_json(response.text),
        model_name=model_name,
        usage=usage,
        estimated_cost_usd=estimate_cost_usd(model_name, usage),
    )
