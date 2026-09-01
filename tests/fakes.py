"""Fake Gemini client objects so tests never make a network call."""


class FakeUsage:
    def __init__(
        self,
        prompt_token_count: int = 0,
        candidates_token_count: int = 0,
        total_token_count: int = 0,
        thoughts_token_count: int | None = None,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.thoughts_token_count = thoughts_token_count


class FakeResponse:
    def __init__(self, text: str, usage: FakeUsage | None = None) -> None:
        self.text = text
        if usage is not None:
            self.usage_metadata = usage


class ScriptedModels:
    """Return queued responses in order and record every request."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(
                "generate_content called more times than queued."
            )
        return self.responses.pop(0)


class ScriptedClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.models = ScriptedModels(responses)
