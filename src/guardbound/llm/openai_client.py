"""OpenAI API backend (GPT-3.5-turbo / GPT-4o / o1).

Model identifiers are pinned to the paper's versions via config
(``llm.openai``). Requires the ``openai`` package (lazy import).
"""
from __future__ import annotations

import os

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

logger = get_logger(__name__)


class OpenAIChatLLM(ChatLLM):
    def __init__(self, model_id: str, api_key_env: str = "OPENAI_API_KEY"):
        self.model_id = model_id
        self.api_key_env = api_key_env
        self._client = None  # lazy

    name = "openai"

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The 'openai' package is required for OpenAIChatLLM. "
                    "Install it with: pip install openai"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            self._client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
        return self._client

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None) -> str:
        # Optionally trim to last N user turns
        if max_turns_context is not None and max_turns_context > 0:
            # Keep system message(s) at front, trim user/assistant pairs from tail
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            # Each "turn" is a user+assistant pair; keep last max_turns_context pairs
            trimmed = non_system[-(max_turns_context * 2):]
            messages = system_msgs + trimmed

        client = self._get_client()
        kwargs: dict = dict(model=self.model_id, messages=messages)
        # o1-family reasoning models reject custom temperatures on some tiers.
        if not self.model_id.startswith("o1") and not self.model_id.startswith("o3"):
            kwargs["temperature"] = temperature
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        logger.debug("OpenAI %s replied (%d chars)", self.model_id,
                     len(content or ""))
        return content or ""
