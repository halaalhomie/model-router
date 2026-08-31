import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when a required local setting is missing."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str


def load_settings() -> Settings:
    """Load required settings from .env into one validated object."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key:
        raise ConfigurationError("GEMINI_API_KEY is missing. Add it to your .env file.")
    if not model:
        raise ConfigurationError("GEMINI_MODEL is missing. Add it to your .env file.")

    return Settings(gemini_api_key=api_key, gemini_model=model)
