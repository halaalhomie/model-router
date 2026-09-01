import unittest

import httpx2
from google.genai import errors

from app.retry import call_with_retry, is_retryable


def server_error(code: int = 504) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"message": "deadline exceeded"}})


def client_error(code: int) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": "error"}})


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
