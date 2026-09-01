import sys

from google import genai

from app.config import ConfigurationError, load_settings
from app.pipeline import run_pipeline
from app.schemas import PipelineResult


USAGE = 'Usage: python -m app.main "your request here"'


def use_utf8(stream: object) -> None:
    """Force UTF-8 on a console stream.

    Windows consoles default to cp1252, which cannot encode characters that
    models emit constantly (em dashes, arrows, box drawing). Without this the
    answer is lost to a UnicodeEncodeError after we have already paid for it.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def read_request(argv: list[str]) -> str:
    """Take the request from the command line, or from a pipe."""
    if argv:
        return " ".join(argv)
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def format_result(result: PipelineResult) -> str:
    """Show the routing decision above the answer, so the choice is visible."""
    profile = result.profile
    lines = [
        f"task type    : {profile.task_type}",
        f"difficulty   : {profile.difficulty}",
        f"reasoning    : {profile.reasoning_required}",
        f"context size : {profile.context_size}",
        f"output type  : {profile.output_type}",
        f"confidence   : {profile.confidence:.2f}",
        f"model        : {result.decision.model_name}",
        f"reason       : {result.decision.reason}",
    ]

    usage = result.response.usage
    if usage is not None:
        lines.append(
            f"tokens       : {usage.prompt_tokens} in / "
            f"{usage.output_tokens} out / "
            f"{usage.thinking_tokens} thinking / "
            f"{usage.total_tokens} total"
        )

    lines.extend(["", result.response.text])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

    request_text = read_request(sys.argv[1:] if argv is None else argv)

    if not request_text.strip():
        print(USAGE, file=sys.stderr)
        return 2

    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    client = genai.Client(api_key=settings.gemini_api_key)
    result = run_pipeline(request_text, client=client, settings=settings)

    print(format_result(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())