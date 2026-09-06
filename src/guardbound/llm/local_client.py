"""HuggingFace local backend for open-source targets (Llama-3-8b-instruct, Phi-4).

Requires ``transformers`` (+ torch). Model ids come from config
(``llm.local``). Lazy import keeps the package importable without GPU deps.
"""
from __future__ import annotations

from pathlib import Path

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
        self._pipeline = None
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            model_path = Path(self.model_id)
            if model_path.exists() and model_path.is_dir():
                actual_path = str(model_path.resolve())
            else:
                actual_path = self.model_id
            self._tokenizer = AutoTokenizer.from_pretrained(
                actual_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    name = f"hf-local"

    def _get_pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            except ImportError as exc:
                raise ImportError(
                    "The 'transformers' package is required for HFLocalChatLLM. "
                    "Install it with: pip install transformers torch accelerate"
                ) from exc
            
            model_path = Path(self.model_id)
            if model_path.exists() and model_path.is_dir():
                actual_path = str(model_path.resolve())
                logger.info("Loading local model from: %s", actual_path)
                self._pipeline = pipeline(
                    "text-generation",
                    model=actual_path,
                    device_map=self.device_map,
                    torch_dtype="auto",
                    trust_remote_code=True,
                )
            else:
                logger.info("Loading HuggingFace model: %s", self.model_id)
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map=self.device_map,
                )
        return self._pipeline

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None) -> str:
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            messages = system_msgs + trimmed

        tokenizer = self._get_tokenizer()
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        pipe = self._get_pipeline()
        output = pipe(
            text,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0]["generated_text"]
        reply = generated[len(text):] if generated.startswith(text) else generated
        logger.debug("HF %s replied (%d chars)", self.model_id, len(reply))
        return reply
