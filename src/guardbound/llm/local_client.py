"""HuggingFace local backend for open-source targets (Llama-3-8b-instruct, Phi-4).

Requires ``transformers`` (+ torch). Model ids come from config
(``llm.local``). Lazy import keeps the package importable without GPU deps.
"""
from __future__ import annotations

import threading
from pathlib import Path

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

logger = get_logger(__name__)

_pipeline_cache: dict = {}
_cache_lock = threading.Lock()


def _extract_json_block(text: str):
    """Return the first ``{...}`` JSON object embedded in ``text`` or None.

    Local instruct models frequently wrap JSON in prose or markdown fences.
    """
    import json

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start:i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    return None
    return None


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
            cache_key = f"{self.model_id}_{self.device_map}"
            with _cache_lock:
                if cache_key in _pipeline_cache:
                    self._pipeline = _pipeline_cache[cache_key]
                    logger.info("Reusing cached pipeline for: %s", self.model_id)
                    return self._pipeline
            
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
            
            with _cache_lock:
                _pipeline_cache[cache_key] = self._pipeline
        return self._pipeline

    def generate(self, messages: list[Message],
                 temperature: float = DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None,
                 json_format: bool = False) -> str | dict:
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
        if json_format and reply:
            import json
            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                # Best-effort: extract the first {...} JSON block. Local models
                # frequently wrap JSON in prose or markdown fences; the official
                # local generate path returns raw text in this situation, so
                # the attacks already tolerate a str fallback, but extracting
                # the block keeps the pipeline flowing on local Llama.
                extracted = _extract_json_block(reply)
                if extracted is not None:
                    return extracted
                return reply
        return reply
