"""HuggingFace local backend for open-source targets (Llama-3-8b-instruct, Phi-4).

Requires ``transformers`` (+ torch). Model ids come from config
(``llm.local``). Lazy import keeps the package importable without GPU deps.
"""
from __future__ import annotations

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

logger = get_logger(__name__)


class HFLocalChatLLM(ChatLLM):
    """Runs a local instruct model with a transformers text-generation pipeline."""

    def __init__(self, model_id: str, device_map: str = "auto",
                 max_new_tokens: int = 512):
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._pipeline = None  # lazy

    name = f"hf-local"

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "The 'transformers' package is required for HFLocalChatLLM. "
                    "Install it with: pip install transformers torch accelerate"
                ) from exc
            logger.info("Loading %s on %s ...", self.model_id, self.device_map)
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device_map,
            )
        return self._pipeline

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None) -> str:
        # Optionally trim to last N user turns
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            messages = system_msgs + trimmed

        pipe = self._get_pipeline()
        output = pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0]["generated_text"]
        reply = generated[-1]["content"]  # last message is the assistant turn
        logger.debug("HF %s replied (%d chars)", self.model_id, len(reply))
        return reply
