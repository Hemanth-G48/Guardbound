"""Ollama-backed ChatLLM for local model serving.

Provides an ``OllamaChatLLM`` that wraps the Ollama API
in the same ``ChatLLM`` interface used by OpenAI/Anthropic backends.

Requires Ollama server running locally (default: http://localhost:11434).
"""
from __future__ import annotations

import os
from typing import Any

from ..logging_utils import get_logger
from .base import ChatLLM, Message, DEFAULT_TEMPERATURE

logger = get_logger(__name__)

try:
    import httpx
    _httpx_available = True
except ImportError:
    _httpx_available = False


class OllamaChatLLM(ChatLLM):
    """Ollama API-backed ChatLLM.

    Requires a running ``ollama`` server.  Set ``OLLAMA_BASE_URL`` to
    override the default ``http://localhost:11434``.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3.6",
        base_url: str | None = None,
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL")
                         or "http://localhost:11434")
        self.timeout = timeout
        self._client = None  # lazy
        self._async_client = None  # lazy async client

    # ------------------------------------------------------------------ #
    # Internal HTTP client (lazy so the package is importable without httpx)
    # ------------------------------------------------------------------ #

    def _get_client(self):
        if not _httpx_available:
            raise ImportError(
                "OllamaChatLLM requires the httpx package. "
                "Install it: pip install httpx"
            )
        if self._client is None:
            self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
        return self._client

    async def _get_async_client(self):
        if not _httpx_available:
            raise ImportError(
                "OllamaChatLLM requires the httpx package. "
                "Install it: pip install httpx"
            )
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self._async_client

    # ------------------------------------------------------------------ #
    # ChatLLM interface
    # ------------------------------------------------------------------ #

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
    ) -> str | dict:
        client = self._get_client()

        # Optionally trim to last N user/assistant pairs (keep system)
        msgs = list(messages)
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in msgs if m["role"] == "system"]
            non_system = [m for m in msgs if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            msgs = system_msgs + trimmed

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "stream": False,
        }
        if json_format:
            payload["format"] = "json"

        logger.info("Calling Ollama %s...", self.model)
        response = client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        reply = data.get("message", {}).get("content", "")
        logger.debug("Ollama %s replied (%d chars)", self.model, len(reply))
        if json_format and reply:
            import json
            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                return reply
        return reply

    async def generate_async(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
    ) -> str | dict:
        client = await self._get_async_client()

        # Optionally trim to last N user/assistant pairs (keep system)
        msgs = list(messages)
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in msgs if m["role"] == "system"]
            non_system = [m for m in msgs if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            msgs = system_msgs + trimmed

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "stream": False,
        }
        if json_format:
            payload["format"] = "json"

        logger.info("Calling Ollama %s (async)...", self.model)
        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        reply = data.get("message", {}).get("content", "")
        logger.debug("Ollama %s replied (%d chars)", self.model, len(reply))
        if json_format and reply:
            import json
            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                return reply
        return reply
