"""Shared machinery for local *multimodal-processor* attacker backends.

Two backends share one shape:

    messages -> AutoProcessor.apply_chat_template(...) -> tensor inputs
             -> <native conditional-generation class>.generate(...) -> decoded text

``GLM4VChatLLM`` (``zai-org/GLM-4.6V-Flash``, ``Glm4vForConditionalGeneration``)
and ``Ornith15ChatLLM`` (``ornith-ai/Ornith-1.5-9B``,
``Qwen3_5ForConditionalGeneration``) differ only in which processor/model class
they resolve and in the label they log. Everything else — context trimming, the
generation parameters, the frozen structured-output machinery and the frozen
JSON contract — is identical and lives here once.

Why this base exists rather than ``pipeline("text-generation")``: both models are
image-text-to-text checkpoints whose ``config.architectures`` name a
*ConditionalGeneration* class. Driving them through a causal-LM pipeline either
warns and uses a mismatched wrapper (GLM) or resolves a different class entirely
(``Qwen3_5ForCausalLM`` for a ``Qwen3_5ForConditionalGeneration`` checkpoint).
Each subclass therefore resolves its own native class and may verify the loaded
model matches the architecture the checkpoint declares.

Deliberately NOT implemented here (matching the frozen experiment contract):
no JSON repair/sanitisation, no retry, no second generation, no fallback model,
no alternate prompt, and no ``<think>`` / reasoning-wrapper handling. Malformed
output reaches the caller's parser exactly as it would from any other backend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..logging_utils import get_logger
from .base import ChatLLM, DEFAULT_TEMPERATURE, Message

# Reuse the frozen JSON contract verbatim: the same structural extraction and the
# same `}` stopping criterion every other backend uses. Importing (not copying)
# keeps a single authoritative implementation.
from .local_client import (
    _JSONStoppingCriteria,
    _build_quantization_config,
    _extract_json_block,
    _termination_reason,
)

logger = get_logger(__name__)

_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class MultimodalChatLLM(ChatLLM):
    """Base for local backends that drive an ``AutoProcessor`` + native model.

    Subclasses must set ``name`` / ``label`` and implement:
      * ``_load_processor(model_id, revision)`` -> processor
      * ``_model_class()`` -> the class whose ``from_pretrained`` loads weights
    and may override ``_verify_model(model)`` to assert the loaded class.
    """

    name = "multimodal-local"
    label = "multimodal"

    def __init__(
        self,
        model_id: str,
        device_map: str = "cuda",
        max_new_tokens: int | None = 256,
        chat_template_kwargs: dict[str, Any] | None = None,
        structured_output_mode: str | None = None,
        revision: str | None = None,
        dtype: str = "bfloat16",
        # Phase 17 Stage 3: opt-in load-time quantisation, mirroring the text
        # backend. ``None`` (the default, and every existing caller) loads
        # exactly as before.
        quantization: str | None = None,
        # Phase 17 Stage 3: declared sampling parameters. ``None`` keeps the
        # previous behaviour (sampling derived from temperature, no top_p).
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        # ``None`` means no artificial output cap: generation is then bounded
        # only by the model's own context window, exactly as the text backend
        # behaves when an experiment leaves max_new_tokens unset.
        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
        self.structured_output_mode = structured_output_mode
        self.revision = revision
        self.dtype_name = dtype
        if quantization not in (None, "nf4"):
            raise ValueError(
                f"unsupported quantization {quantization!r}; supported: [None, 'nf4']"
            )
        self.quantization = quantization
        self.top_p = top_p
        self.do_sample = do_sample
        # Lifecycle handles. ``ModelManager._evict`` nulls exactly these names,
        # so they must stay in sync with its release conventions.
        self._pipeline: Any = None   # the loaded model
        self._tokenizer: Any = None  # the loaded processor
        # Per-call telemetry from the most recent generate() (same shape as the
        # text backend's, so both paths report the same quantities).
        self.last_generation_stats: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _load_processor(self, model_id: str, revision: str | None):
        raise NotImplementedError

    def _model_class(self):
        raise NotImplementedError

    def _verify_model(self, model) -> None:
        """Optional post-load architecture assertion (default: no-op)."""

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype_name not in _DTYPE_NAMES:
            raise ValueError(
                f"unsupported dtype {self.dtype_name!r}; "
                f"expected one of {sorted(_DTYPE_NAMES)}"
            )
        return _DTYPE_NAMES[self.dtype_name]

    def _resolve_path(self) -> str:
        """Local directory when the id is one, otherwise the repo id."""
        model_path = Path(self.model_id)
        if model_path.exists() and model_path.is_dir():
            return str(model_path.resolve())
        return self.model_id

    def _get_processor(self):
        if self._tokenizer is None:
            actual = self._resolve_path()
            logger.info("Loading %s processor: %s", self.label, actual)
            revision = self.revision if actual == self.model_id else None
            self._tokenizer = self._load_processor(actual, revision)
        return self._tokenizer

    def _get_pipeline(self):
        """Load the model. Gated by the caller (ModelManager)."""
        if self._pipeline is None:
            model_cls = self._model_class()
            actual = self._resolve_path()
            logger.info(
                "Loading %s model: %s (revision=%s, dtype=%s, device_map=%s)",
                self.label, actual, self.revision, self.dtype_name, self.device_map,
            )
            kwargs: dict[str, Any] = {
                "dtype": self._torch_dtype(),
                "device_map": self.device_map,
                "attn_implementation": "sdpa",
            }
            if self.quantization is not None:
                kwargs["quantization_config"] = _build_quantization_config(self.quantization)
            if actual == self.model_id:
                if self.revision is not None:
                    kwargs["revision"] = self.revision
                kwargs["trust_remote_code"] = True
            self._pipeline = model_cls.from_pretrained(actual, **kwargs)
            self._verify_model(self._pipeline)
            self._pipeline.eval()
        return self._pipeline

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _trim_context(self, messages: list[Message],
                      max_turns_context: int | None) -> list[Message]:
        """Same context-window rule as ``HFLocalChatLLM`` (system kept, N turns)."""
        if max_turns_context is None or max_turns_context <= 0:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        return system_msgs + non_system[-(max_turns_context * 2):]

    def _build_inputs(self, messages: list[Message]):
        processor = self._get_processor()
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **self.chat_template_kwargs,
        )
        model = self._get_pipeline()
        device = next(model.parameters()).device
        moved = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in dict(inputs).items()
        }
        return moved, processor, model

    def _context_generation_budget(self, model, input_len: int) -> int:
        """Largest number of tokens a prompt can still generate (context bound).

        Mirrors the text backend: with no declared output cap, the only bound is
        the model's own context window — for a multimodal wrapper the language
        sub-config carries it.
        """
        limits: list[int] = []
        config = getattr(model, "config", None)
        for candidate in (config, getattr(config, "text_config", None)):
            value = getattr(candidate, "max_position_embeddings", None)
            if isinstance(value, int) and value > 0:
                limits.append(value)
        window = min(limits) if limits else 4096
        return max(1, window - input_len)

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ) -> str | dict:
        import time as _time

        started = _time.time()
        messages = self._trim_context(list(messages), max_turns_context)
        inputs, processor, model = self._build_inputs(messages)
        tokenizer = getattr(processor, "tokenizer", processor)
        input_len = int(inputs["input_ids"].shape[1])

        budget = self.max_new_tokens
        if budget is None:
            budget = self._context_generation_budget(model, input_len)
        sampling = self.do_sample if self.do_sample is not None else temperature > 0

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": budget,
            "do_sample": sampling,
            "use_cache": True,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p

        # Structured output: reuse the frozen Phase 14.1 mechanism unchanged.
        mode = (structured_output_mode if structured_output_mode is not None
                else self.structured_output_mode)
        if mode == "constrained_json" and json_format:
            from .json_constrained_decoder import build_json_logits_processor

            logits_processor = build_json_logits_processor(tokenizer)
            if logits_processor:
                gen_kwargs["logits_processor"] = logits_processor
            stopping = _JSONStoppingCriteria.build(tokenizer, True)
            if stopping:
                gen_kwargs["stopping_criteria"] = stopping
        else:
            stopping = _JSONStoppingCriteria.build(tokenizer, json_format)
            if stopping:
                gen_kwargs["stopping_criteria"] = stopping

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0][input_len:]
        reply = processor.batch_decode([new_ids], skip_special_tokens=True)[0]
        logger.debug("%s replied (%d chars)", self.label, len(reply))

        # Per-call telemetry, the same quantities the text backend records, so
        # the two paths can be compared line by line.
        last_token = int(new_ids[-1]) if new_ids.numel() else None
        eos_ids = gen_kwargs.get("eos_token_id")
        self.last_generation_stats = {
            "model_id": self.model_id,
            "backend": type(self).__name__,
            "quantization": self.quantization,
            "raw_reply": reply,
            "prompt_tokens": input_len,
            "generated_tokens": int(new_ids.shape[-1]),
            "context_tokens_total": int(input_len + new_ids.shape[-1]),
            "max_new_tokens_effective": int(budget),
            "max_new_tokens_declared": self.max_new_tokens,
            "hit_output_budget": int(new_ids.shape[-1]) >= int(budget),
            "last_token_id": last_token,
            "eos_token_ids": (
                [eos_ids] if isinstance(eos_ids, int)
                else list(eos_ids or [])
            ),
            "termination": _termination_reason(
                last_token, eos_ids, int(new_ids.shape[-1]), int(budget)
            ),
            "elapsed_s": round(_time.time() - started, 3),
            "json_format": json_format,
            "temperature": temperature,
            "do_sample": bool(sampling),
            "top_p": self.top_p,
        }
        if torch.cuda.is_available():
            self.last_generation_stats["vram_allocated_gib"] = round(
                torch.cuda.memory_allocated() / 1024 ** 3, 3
            )

        # JSON handling: identical to HFLocalChatLLM. Parse, else the frozen
        # structural extraction, else return the raw string so the caller's
        # parser raises exactly as it would for any other backend.
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

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_id={self.model_id!r}, "
            f"revision={self.revision!r}, dtype={self.dtype_name!r}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"structured_output_mode={self.structured_output_mode!r})"
        )
