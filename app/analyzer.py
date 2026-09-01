from google import genai
from google.genai import types

from app.retry import call_with_retry
from app.schemas import TaskProfile


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
) -> TaskProfile:
    """Classify one user request into a validated TaskProfile."""
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

    return TaskProfile.model_validate_json(response.text)
