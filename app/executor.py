from google import genai

from app.pricing import estimate_cost_usd
from app.retry import call_with_retry
from app.schemas import ModelResponse
from app.usage import read_usage


def execute_request(
    request_text: str,
    *,
    client: genai.Client,
    model_name: str,
) -> ModelResponse:
    """Send one request to the routed model and capture its reply and usage."""
    if not request_text.strip():
        raise ValueError("request_text must not be empty.")

    response = call_with_retry(
        lambda: client.models.generate_content(
            model=model_name,
            contents=request_text,
        )
    )

    if not response.text:
        raise ValueError(f"Model {model_name} returned no text.")

    usage = read_usage(response)
    return ModelResponse(
        model_name=model_name,
        text=response.text,
        usage=usage,
        estimated_cost_usd=estimate_cost_usd(model_name, usage),
    )
