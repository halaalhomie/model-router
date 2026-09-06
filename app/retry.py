import logging
import re
import time
from typing import Callable, TypeVar

import httpx2
from google.genai import errors


logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0

# A per-minute quota can ask us to wait about a minute, which is worth
# honouring. A much longer hint means the quota will not clear inside this
# request's lifetime, so failing fast beats hanging on.
MAX_RETRY_DELAY_SECONDS = 75.0

# 429 is a ClientError in google-genai's hierarchy, not a ServerError, but
# it means "retry later," not "this request is malformed" -- so it belongs
# in the retryable set alongside 5xx.
RETRYABLE_CLIENT_STATUS_CODES = {429}


def is_retryable(error: Exception) -> bool:
    """A transient failure worth retrying, vs. one that will just repeat.

    Retryable: 5xx server errors, 429 rate limits, and network/timeout
    errors -- the same failures a human retrying by hand would expect to
    sometimes succeed on. Everything else (400 bad request, 404 unknown
    model, ...) is a defect in the request itself; retrying it three times
    only wastes the timeout budget for an outcome that was never in doubt.
    """
    if isinstance(error, errors.ServerError):
        return True
    if isinstance(error, errors.ClientError):
        return error.code in RETRYABLE_CLIENT_STATUS_CODES
    if isinstance(error, httpx2.TransportError):
        return True
    return False


def retry_after_seconds(error: Exception) -> float | None:
    """The server's own "wait this long" hint, if it sent one.

    A rate limit is the one failure where the server knows exactly how
    long the client should wait, and Gemini says so: a 429 carries a
    google.rpc.RetryInfo entry like {"retryDelay": "56s"}. Backing off
    1s, 2s against a per-minute quota is guaranteed to fail again, so
    when the server states a number, it wins over our guess.
    """
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None

    entries = details.get("error", {}).get("details")
    if not isinstance(entries, list):
        return None

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if "RetryInfo" not in str(entry.get("@type", "")):
            continue
        match = re.fullmatch(
            r"\s*([0-9]*\.?[0-9]+)s\s*", str(entry.get("retryDelay", ""))
        )
        if match:
            return float(match.group(1))
    return None


def delay_before_retry(
    error: Exception, attempt: int, base_delay: float
) -> float:
    """How long to wait: the server's hint if it gave one, else backoff."""
    backoff = base_delay * (2**attempt)
    hinted = retry_after_seconds(error)
    if hinted is None:
        return backoff
    # A tiny hint should not shorten a longer backoff we already planned.
    return min(max(hinted, backoff), MAX_RETRY_DELAY_SECONDS)


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(), retrying a retryable error after a considered delay.

    `sleep` is injectable so tests can assert on timing without a suite
    that actually takes minutes to run.
    """
    last_error: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as error:
            if not is_retryable(error):
                raise
            last_error = error
            if attempt < max_attempts - 1:
                wait = delay_before_retry(error, attempt, base_delay)
                if retry_after_seconds(error) is not None:
                    logger.warning(
                        "Rate limited; the server asked for %.0fs. Waiting "
                        "%.0fs before retry %d of %d.",
                        retry_after_seconds(error),
                        wait,
                        attempt + 2,
                        max_attempts,
                    )
                sleep(wait)

    assert last_error is not None
    raise last_error
