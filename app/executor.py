from google import genai

from app.schemas import ModelResponse, TokenUsage


def execute_request(
    request_text: str,
    *,
    client: genai.Client,
    model_name: str,
) -> ModelResponse:
    """Send one request to the routed model and capture its reply and usage."""
    if not request_text.strip():
        raise ValueError("request_text must not be empty.")

    response = client.models.generate_content(
        model=model_name,
        contents=request_text,
    )

    if not response.text:
        raise ValueError(f"Model {model_name} returned no text.")

    return ModelResponse(
        model_name=model_name,
        text=response.text,
        usage=read_usage(response),
    )


def read_usage(response: object) -> TokenUsage | None:
    """Read token counts defensively: usage metadata is not always present."""
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    return TokenUsage(
        prompt_tokens=getattr(metadata, "prompt_token_count", None) or 0,
        output_tokens=getattr(metadata, "candidates_token_count", None) or 0,
        thinking_tokens=getattr(metadata, "thoughts_token_count", None) or 0,
        total_tokens=getattr(metadata, "total_token_count", None) or 0,
    )
