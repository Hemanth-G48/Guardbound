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
                 max_turns_context: int | None = None) -> str:
        self.calls.append((messages, temperature))
        if self.responses:
            return self.responses.pop(0)
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[mock reply to]: {last_user}"
