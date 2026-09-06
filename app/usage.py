from app.schemas import TokenUsage


def read_usage(response: object) -> TokenUsage | None:
    """Read token counts off a Gemini response, defensively.

    Shared by the analyzer and the executor: both make a model call, and
    both must account for what it cost. Usage metadata is not guaranteed to
    be present, and a missing count is reported as None rather than zero so
    downstream pricing can tell "nothing was used" apart from "we don't
    know what was used".
    """
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None

    return TokenUsage(
        prompt_tokens=getattr(metadata, "prompt_token_count", None) or 0,
        output_tokens=getattr(metadata, "candidates_token_count", None) or 0,
        thinking_tokens=getattr(metadata, "thoughts_token_count", None) or 0,
        total_tokens=getattr(metadata, "total_token_count", None) or 0,
    )
