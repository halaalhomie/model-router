import unittest

import httpx2
from google.genai import errors

from app.retry import (
    MAX_RETRY_DELAY_SECONDS,
    call_with_retry,
    is_retryable,
    retry_after_seconds,
)


def rate_limit_error(delay: float | str) -> errors.ClientError:
    """A 429 carrying the google.rpc.RetryInfo hint Gemini really sends.

    `delay` is passed through verbatim when it is a string, so tests can
    reproduce the exact wire format observed from the API ("58s", no
    decimal point) rather than only the float form.
    """
    retry_delay = delay if isinstance(delay, str) else f"{delay}s"
    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help"},
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": retry_delay,
                    },
                ],
            }
        },
    )


def server_error(code: int = 504) -> errors.ServerError:
    return errors.ServerError(
        code, {"error": {"message": "deadline exceeded"}}
    )


def client_error(code: int) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": "error"}})


class RetryAfterTests(unittest.TestCase):
    def test_reads_an_integer_hint_as_the_api_really_sends_it(self) -> None:
        """The observed wire format is "58s", not "58.0s"."""
        hinted = retry_after_seconds(rate_limit_error("58s"))

        self.assertAlmostEqual(hinted, 58.0)

    def test_reads_a_fractional_hint(self) -> None:
        hinted = retry_after_seconds(rate_limit_error("56.083075953s"))

        self.assertAlmostEqual(hinted, 56.083075953)

    def test_returns_none_when_there_is_no_hint(self) -> None:
        plain = errors.ClientError(429, {"error": {"code": 429}})

        self.assertIsNone(retry_after_seconds(plain))

    def test_returns_none_for_an_error_without_details(self) -> None:
        self.assertIsNone(retry_after_seconds(ValueError("nope")))

    def test_ignores_a_malformed_delay(self) -> None:
        bad = errors.ClientError(
            429,
            {
                "error": {
                    "details": [
                        {
                            "@type": ".../google.rpc.RetryInfo",
                            "retryDelay": "soon",
                        }
                    ]
                }
            },
        )

        self.assertIsNone(retry_after_seconds(bad))


class IsRetryableTests(unittest.TestCase):
    def test_server_errors_are_retryable(self) -> None:
        self.assertTrue(is_retryable(server_error(500)))
        self.assertTrue(is_retryable(server_error(504)))

    def test_rate_limit_is_retryable(self) -> None:
        self.assertTrue(is_retryable(client_error(429)))

    def test_bad_request_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(client_error(400)))

    def test_not_found_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(client_error(404)))

    def test_network_timeout_is_retryable(self) -> None:
        self.assertTrue(is_retryable(httpx2.ConnectTimeout("boom")))

    def test_an_unrelated_exception_is_not_retryable(self) -> None:
        self.assertFalse(is_retryable(ValueError("not a provider error")))


class CallWithRetryTests(unittest.TestCase):
    def test_returns_the_result_on_first_success(self) -> None:
        result = call_with_retry(lambda: "ok", sleep=lambda _: None)

        self.assertEqual(result, "ok")

    def test_retries_a_retryable_error_and_then_succeeds(self) -> None:
        attempts = [server_error(), server_error(), "ok"]

        def flaky() -> str:
            outcome = attempts.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        result = call_with_retry(flaky, sleep=lambda _: None)

        self.assertEqual(result, "ok")
        self.assertEqual(attempts, [])

    def test_does_not_retry_a_non_retryable_error(self) -> None:
        calls = 0

        def always_bad_request() -> str:
            nonlocal calls
            calls += 1
            raise client_error(400)

        with self.assertRaises(errors.ClientError):
            call_with_retry(always_bad_request, sleep=lambda _: None)

        self.assertEqual(calls, 1)

    def test_raises_the_last_error_once_attempts_are_exhausted(self) -> None:
        calls = 0

        def always_rate_limited() -> str:
            nonlocal calls
            calls += 1
            raise client_error(429)

        with self.assertRaises(errors.ClientError) as caught:
            call_with_retry(
                always_rate_limited, max_attempts=3, sleep=lambda _: None
            )

        self.assertEqual(calls, 3)
        self.assertEqual(caught.exception.code, 429)

    def test_honours_the_servers_retry_hint_over_our_backoff(self) -> None:
        """A per-minute quota tells us exactly how long to wait. Backing
        off 1s against a 56s hint just fails again."""
        delays: list[float] = []

        def rate_limited() -> str:
            raise rate_limit_error(56.0)

        with self.assertRaises(errors.ClientError):
            call_with_retry(
                rate_limited, max_attempts=2, sleep=delays.append
            )

        self.assertEqual(delays, [56.0])

    def test_caps_an_unreasonably_long_hint(self) -> None:
        """A hint far beyond the request's lifetime means the quota will
        not clear in time; failing fast beats hanging on."""
        delays: list[float] = []

        def rate_limited() -> str:
            raise rate_limit_error(3600.0)

        with self.assertRaises(errors.ClientError):
            call_with_retry(
                rate_limited, max_attempts=2, sleep=delays.append
            )

        self.assertEqual(delays, [MAX_RETRY_DELAY_SECONDS])

    def test_a_tiny_hint_does_not_shorten_the_backoff(self) -> None:
        delays: list[float] = []

        def rate_limited() -> str:
            raise rate_limit_error(0.1)

        with self.assertRaises(errors.ClientError):
            call_with_retry(
                rate_limited, max_attempts=2, base_delay=5.0,
                sleep=delays.append,
            )

        self.assertEqual(delays, [5.0])

    def test_backs_off_exponentially_between_attempts(self) -> None:
        delays: list[float] = []

        def always_retryable() -> str:
            raise server_error()

        with self.assertRaises(errors.ServerError):
            call_with_retry(
                always_retryable,
                max_attempts=3,
                base_delay=1.0,
                sleep=delays.append,
            )

        self.assertEqual(delays, [1.0, 2.0])
