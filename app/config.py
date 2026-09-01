import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.schemas import ModelCatalog


class ConfigurationError(ValueError):
    """Raised when a required local setting is missing."""


@dataclass(frozen=True)
class Settings:
    """Every value the pipeline needs, resolved once at startup.

    The analyzer model is kept separate from the catalog on purpose: the
    classifier runs on every request, so it should be the cheapest model we
    trust, never whichever model the router happens to pick.
    """

    gemini_api_key: str
    analyzer_model: str
    catalog: ModelCatalog


def require(name: str) -> str:
    """Read one required environment variable, or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"{name} is missing. Add it to your .env file."
        )
    return value


def load_settings() -> Settings:
    """Load required settings from .env into one validated object."""
    load_dotenv()

    return Settings(
        gemini_api_key=require("GEMINI_API_KEY"),
        analyzer_model=require("GEMINI_ANALYZER_MODEL"),
        catalog=ModelCatalog(
            fast_model=require("GEMINI_FAST_MODEL"),
            code_model=require("GEMINI_CODE_MODEL"),
            reasoning_model=require("GEMINI_REASONING_MODEL"),
            long_context_model=require("GEMINI_LONG_CONTEXT_MODEL"),
        ),
    )
