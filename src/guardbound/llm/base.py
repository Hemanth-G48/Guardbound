"""Chat-LLM abstraction used by every phase that talks to a target model.

All clients default to ``temperature=0.7`` — the paper-wide generation setting
(Sec. 5.1). Concrete backends import their SDK lazily so this package stays
importable without API keys or optional dependencies installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}

DEFAULT_TEMPERATURE = 0.7  # paper-wide setting


class ChatLLM(ABC):
    """Minimal chat interface shared by API, local, and mock backends."""

    name: str = "chat-llm"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
    ) -> str:
        """Return the assistant reply for a list of chat messages.

        Parameters
        ----------
        messages:
            Chat history as a list of role/content dicts.
        temperature:
            Sampling temperature (paper-wide default: 0.7).
        max_turns_context:
            Optional cap on how many recent turns to include in context.
            ``None`` means use all provided messages.
        """

    # -- convenience --------------------------------------------------------- #

    def chat(self, user_text: str, system: str | None = None,
             temperature: float = DEFAULT_TEMPERATURE,
             max_turns_context: int | None = None) -> str:
        messages: list[Message] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return self.generate(messages, temperature=temperature,
                             max_turns_context=max_turns_context)
