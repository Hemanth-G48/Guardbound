"""Deterministic mock backend for tests and offline runs."""
from __future__ import annotations

from .base import ChatLLM, DEFAULT_TEMPERATURE, Message


class MockChatLLM(ChatLLM):
    """Returns scripted responses in order; echoes when the script runs out.

    Never touches the network, so unit tests run everywhere.
    """

    name = "mock"

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.calls: list[tuple[list[Message], float]] = []  # recorded for assertions

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None,
                 json_format: bool = False) -> str | dict:
        # Snapshot the request: callers may mutate ``messages`` after generate()
        # returns (e.g. the runner appends the assistant reply), and a call
        # recorder must capture the request exactly as it was passed in.
        self.calls.append(([dict(m) for m in messages], temperature, json_format))
        if self.responses:
            raw = self.responses.pop(0)
        else:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            raw = f"[mock reply to]: {last_user}"
        if json_format:
            import json
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw
        return raw
