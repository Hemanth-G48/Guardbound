"""HuggingFace local backend for open-source targets (Llama-3-8b-instruct, Phi-4).

Requires ``transformers`` (+ torch). Model ids come from config
(``llm.local``). Lazy import keeps the package importable without GPU deps.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import torch

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message


def _resolve_model_family(model_id: str) -> str:
    """Best-effort Qwen-family detection for chat-template override.

    Qwen3-family configs (``Qwen3ForCausalLM``, Qwen3.5, Qwen3-4B-Instruct-2507,
    etc.) typically declare ``enable_thinking=True`` by default in their
    ``generation_config.json``, which is picked up by the pipeline and turned into
    a thinking block by their chat template. We disable it at the template layer so
    the model returns usable text inside the 256-token budget.

    Detection is a cheap best-effort heuristic: any model_id whose resolved
    config.json ``model_type`` contains ``Qwen3`` is treated as Qwen3-family.
    """
    try:
        from transformers import AutoConfig

        model_path = Path(model_id)
        actual = (
            str(model_path.resolve())
            if model_path.exists() and model_path.is_dir()
            else model_id
        )
        cfg = AutoConfig.from_pretrained(actual, trust_remote_code=True)
        mt: str = getattr(cfg, "model_type", "")
        if "qwen3" in mt.lower() or "qwen3" in str(model_id).lower():
            return "qwen3"
    except Exception:  # noqa: BLE001 -- best-effort; never hard-fail generation
        pass
    return "other"


def _maybe_thinking_kwargs(model_id: str) -> dict[str, Any]:
    """Return ``{enable_thinking: False}`` for Qwen3-family models."""
    return {"enable_thinking": False} if _resolve_model_family(model_id) == "qwen3" else {}


logger = get_logger(__name__)

_pipeline_cache: dict[str, Any] = {}
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
                block = text[start : i + 1]
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    return None
    return None


class HFLocalChatLLM(ChatLLM):
    """Runs a local instruct model with a transformers text-generation pipeline.

    Provider-layer chat-template settings are applied at ``apply_chat_template``
    time, not at generation-param time. For Qwen3-family models this currently
    means ``enable_thinking=False`` so the model does not emit a long
    ``<think>...</think>`` block that burns the 256-token budget and truncates the
    JSON answer. This is a harness-level template override, not a change to model
    weights, generation params, or the attack/NBF prompts.
    """

    name = "hf-local"

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int = 512,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self.chat_template_kwargs.update(_maybe_thinking_kwargs(model_id))
        self._pipeline: Any = None
        self._tokenizer: Any = None

    @property
    def enable_thinking(self) -> bool | None:
        """Template-layer thinking flag for Qwen3-family models."""
        return self.chat_template_kwargs.get("enable_thinking", False)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_path = Path(self.model_id)
            actual_path = str(model_path.resolve()) if model_path.exists() and model_path.is_dir() else self.model_id
            self._tokenizer = AutoTokenizer.from_pretrained(
                actual_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def _build_gen_config(
        self,
        temperature: float,
        max_new_tokens: int,
        do_sample: bool,
    ):
        from transformers import GenerationConfig

        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=do_sample,
            pad_token_id=self._get_tokenizer().pad_token_id,
            eos_token_id=self._get_tokenizer().eos_token_id,
        )

    def _get_pipeline(self):
        if self._pipeline is None:
            cache_key = f"{self.model_id}_{self.device_map}"
            with _cache_lock:
                if cache_key in _pipeline_cache:
                    self._pipeline = _pipeline_cache[cache_key]
                    logger.info("Reusing cached pipeline for: %s", self.model_id)
                    return self._pipeline

            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    GenerationConfig,
                    pipeline,
                )
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
                    generation_config=self._build_gen_config(
                        temperature=DEFAULT_TEMPERATURE,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    ),
                )
            else:
                logger.info("Loading HuggingFace model: %s", self.model_id)
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.model_id,
                    device_map=self.device_map,
                    dtype=torch.bfloat16,
                    generation_config=self._build_gen_config(
                        temperature=DEFAULT_TEMPERATURE,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    ),
                )

            with _cache_lock:
                _pipeline_cache[cache_key] = self._pipeline
        return self._pipeline

    def __repr__(self) -> str:
        return (
            f"HFLocalChatLLM(model_id={self.model_id!r}, "
            f"device_map={self.device_map!r}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"chat_template_kwargs={self.chat_template_kwargs!r})"
        )

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
    ) -> str | dict:
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in messages if m["role"] == "system"]
            non_system = [m for m in messages if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2) :]
            messages = system_msgs + trimmed

        tokenizer = self._get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        pipe = self._get_pipeline()

        gen_config = self._build_gen_config(
            temperature=temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
        )
        output = pipe(text, generation_config=gen_config)
        generated = output[0]["generated_text"]
        reply = generated[len(text) :] if generated.startswith(text) else generated
        logger.debug("HF %s replied (%d chars)", self.model_id, len(reply))
        if json_format and reply:
            import json

            try:
                return json.loads(reply)
            except json.JSONDecodeError:
                extracted = _extract_json_block(reply)
                if extracted is not None:
                    return extracted
                return reply
        return reply


def build_llm(
    cfg: dict[str, Any],
    kwargs: dict[str, Any] | None = None,
    **gen_kwargs: Any,
) -> ChatLLM:
    """Build a local ``ChatLLM`` from a provider config dict.

    ``kwargs`` can override ``max_new_tokens``, ``device_map``, and
    ``chat_template_kwargs``. The Qwen3-family thinking override is applied on top
    of whatever kwargs the caller provides.
    """
    if not kwargs:
        kwargs = {}
    kwargs.setdefault("max_new_tokens", int(cfg.get("max_new_tokens", 256)))
    kwargs.setdefault("device_map", cfg.get("device_map", "auto"))
    kwargs.setdefault("chat_template_kwargs", {})
    kwargs["chat_template_kwargs"] = dict(kwargs["chat_template_kwargs"])
    kwargs["chat_template_kwargs"].update(_maybe_thinking_kwargs(cfg.get("model_id", "")))
    return HFLocalChatLLM(cfg["model_id"], **kwargs)
