import time
from typing import Callable, TypeVar

import httpx2
from google.genai import errors


T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BASE_DELAY_SECONDS = 1.0

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


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(), retrying on a retryable error with exponential backoff.

    `sleep` is injectable so tests can assert on backoff timing without a
    real test suite that takes seconds to run.
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
                sleep(base_delay * (2**attempt))

    assert last_error is not None
    raise last_error
