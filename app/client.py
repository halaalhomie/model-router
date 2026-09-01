from google import genai
from google.genai import types

from app.config import Settings


def build_client(settings: Settings) -> genai.Client:
    """Construct the one Gemini client every transport shares.

    Both the CLI and the API need the same client, built the same way (API
    key, per-request timeout). Defining it once here means neither transport
    can drift from the other by forgetting to pass http_options.
    """
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.request_timeout_seconds * 1000)
        ),
    )
