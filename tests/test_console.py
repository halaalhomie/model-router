import unittest

from app.console import use_utf8


class FakeStream:
    def __init__(self) -> None:
        self.encoding: str | None = None
        self.errors: str | None = None

    def reconfigure(self, *, encoding: str, errors: str) -> None:
        self.encoding = encoding
        self.errors = errors


class UseUtf8Tests(unittest.TestCase):
    def test_switches_the_stream_to_utf8(self) -> None:
        stream = FakeStream()

        use_utf8(stream)

        self.assertEqual(stream.encoding, "utf-8")
        self.assertEqual(stream.errors, "replace")

    def test_ignores_a_stream_that_cannot_be_reconfigured(self) -> None:
        use_utf8(object())  # must not raise
