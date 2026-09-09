"""Anthropic API backend (Claude-3.5-Sonnet).

Model identifier pinned via config (``llm.anthropic``). Requires the
``anthropic`` package (lazy import).
"""
from __future__ import annotations

import os

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

logger = get_logger(__name__)


class AnthropicChatLLM(ChatLLM):
    def __init__(self, model_id: str,
                 api_key_env: str = "ANTHROPIC_API_KEY",
                 max_tokens: int = 1024):
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self._client = None  # lazy

    name = "anthropic"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicChatLLM. "
                    "Install it with: pip install anthropic"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            kwargs = {"api_key": api_key} if api_key else {}
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system_text: str | None = None
        rest: list[Message] = []
        for m in messages:
            if m["role"] == "system" and system_text is None:
                system_text = m["content"]
            else:
                rest.append(m)
        return system_text, rest

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None,
                 json_format: bool = False) -> str | dict:
        # Optionally trim to last N user turns
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            messages = system_msgs + trimmed

        client = self._get_client()
        system_text, rest = self._split_system(messages)
        kwargs: dict = dict(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=temperature,
            system=system_text or "",
            messages=[
                {"role": m["role"], "content": m["content"]} for m in rest
            ],
        )
        if json_format:
            kwargs["betas"] = ["json-mode"]
        response = client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content)
        logger.debug("Anthropic %s replied (%d chars)", self.model_id, len(text))
        if json_format and text:
            import json
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                logger.warning("Anthropic %s returned non-JSON response when json_format=True: %s",
                               self.model_id, text[:200])
                return text
        return text
