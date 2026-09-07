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
    request_timeout_seconds: float
    kafka_bootstrap_servers: str
    kafka_consumer_group: str
    database_url: str


# Live-measured request times on the configured catalog run 2-19s (see
# README "Model availability"). 30s gives a normal request headroom without
# leaving a stuck request hanging for the SDK's 10-minute default.
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0

# Matches docker-compose.yml's PLAINTEXT_HOST listener -- what Kafka is
# reachable at from the host, not from inside another container.
DEFAULT_KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# The consumer group the analytics consumer joins. Kafka tracks committed
# offsets per group, so changing this name gives you a fresh reader that
# re-reads the topic from the beginning, independent of the old one.
DEFAULT_KAFKA_CONSUMER_GROUP = "model-router.analytics"

# Gemini itself rejects a deadline under 10s with a 400 INVALID_ARGUMENT
# ("Manually set deadline Ns is too short") -- discovered by live testing,
# not documented anywhere obvious. Enforced here so a too-low override fails
# at startup with a clear message instead of as a cryptic 400 mid-request.
MIN_REQUEST_TIMEOUT_SECONDS = 10.0


def require(name: str) -> str:
    """Read one required environment variable, or fail with a clear message."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"{name} is missing. Add it to your .env file."
        )
    return value


def optional_float(name: str, default: float) -> float:
    """Read an optional numeric environment variable, or fall back to default."""
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        raise ConfigurationError(f"{name} must be a number, got {value!r}.")


def optional_str(name: str, default: str) -> str:
    """Read an optional environment variable, or fall back to default."""
    return os.getenv(name) or default


def load_request_timeout_seconds() -> float:
    timeout = optional_float(
        "REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    if timeout < MIN_REQUEST_TIMEOUT_SECONDS:
        raise ConfigurationError(
            f"REQUEST_TIMEOUT_SECONDS must be at least "
            f"{MIN_REQUEST_TIMEOUT_SECONDS:g}: Gemini itself rejects a "
            f"shorter deadline. Got {timeout:g}."
        )
    return timeout


def load_database_url() -> str:
    """Build the connection URL from the POSTGRES_* vars.

    docker-compose.yml already reads POSTGRES_USER/PASSWORD/DB to create
    the database; deriving the client's URL from those same values keeps
    one source of truth rather than a second copy of the password that can
    drift out of step. DATABASE_URL still wins when set, which is what a
    deployment pointing at a managed database would use.
    """
    explicit = os.getenv("DATABASE_URL")
    if explicit:
        return explicit

    user = require("POSTGRES_USER")
    password = require("POSTGRES_PASSWORD")
    database = require("POSTGRES_DB")
    host = optional_str("POSTGRES_HOST", "localhost")
    port = optional_str("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


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
        request_timeout_seconds=load_request_timeout_seconds(),
        kafka_bootstrap_servers=optional_str(
            "KAFKA_BOOTSTRAP_SERVERS", DEFAULT_KAFKA_BOOTSTRAP_SERVERS
        ),
        kafka_consumer_group=optional_str(
            "KAFKA_CONSUMER_GROUP", DEFAULT_KAFKA_CONSUMER_GROUP
        ),
        database_url=load_database_url(),
    )
