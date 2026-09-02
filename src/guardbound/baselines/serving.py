"""HuggingFace-backed model serving for Phase 8 baselines.

Provides a single ``HFChatLLM`` that supports:

    - base model
    - base model + LoRA SFT/DPO/KTO adapter

All loading is lazy.  The module imports do NOT trigger torch /
transformers / peft imports — those happen only inside
``HFChatLLM.__init__`` and can be skipped entirely with
``HFMOCK=1`` or by using the ``--mock`` CLI flag.

The ``build_hf_chatllm`` factory returns either a real
``HFChatLLM`` (when ``--no-mock``) or a ``MockChatLLM`` placeholder
that satisfies the ``ChatLLM`` interface (so the rest of the
pipeline can be smoke-tested without GPU or model downloads).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..llm.base import ChatLLM
from ..llm.mock import MockChatLLM
from ..logging_utils import get_logger

logger = get_logger(__name__)


def is_mock_mode() -> bool:
    """True if HF serving should fall back to MockChatLLM.

    Triggered by ``--mock``, ``HFMOCK=1``, or absence of ``torch``.
    """
    if os.environ.get("HFMOCK", "").lower() in {"1", "true", "yes"}:
        return True
    if os.environ.get("NBF_MOCK", "").lower() in {"1", "true", "yes"}:
        return True
    try:
        import torch  # noqa: F401
    except ImportError:
        return True
    return False


class HFChatLLM(ChatLLM):
    """HuggingFace-backed ChatLLM with optional LoRA adapter.

    The class is lazy about heavy imports.  Until ``generate`` is
    actually called the underlying ``transformers`` pipeline is not
    built.  This keeps the rest of the codebase importable without
    GPU dependencies.
    """

    name = "hf-adapter"

    def __init__(
        self,
        base_model: str,
        adapter_path: str | Path | None = None,
        *,
        device_map: str | Any = "auto",
        max_new_tokens: int = 512,
        torch_dtype: str = "auto",
    ):
        self.base_model = base_model
        self.adapter_path = str(adapter_path) if adapter_path is not None else None
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self._pipeline = None  # lazy

    def _load(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "HFChatLLM requires the transformers + torch packages. "
                "Install with: pip install transformers torch accelerate"
            ) from exc
        kwargs: dict[str, Any] = dict(device_map=self.device_map)
        if self.torch_dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
        logger.info("Loading base model %s ...", self.base_model)
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.adapter_path is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "Loading a LoRA adapter requires the peft package. "
                    "Install with: pip install peft"
                ) from exc
            logger.info("Applying LoRA adapter from %s", self.adapter_path)
            model = PeftModel.from_pretrained(model, self.adapter_path)
        # Build a text-generation pipeline on demand in generate().
        self._pipeline = (model, tokenizer)

    def generate(self, messages, temperature=0.7, max_turns_context=None) -> str:
        self._load()
        import torch
        model, tokenizer = self._pipeline
        # Truncate to last N user/assistant pairs (keep system)
        msgs = list(messages)
        if max_turns_context is not None and max_turns_context > 0:
            system_msgs = [m for m in msgs if m["role"] == "system"]
            non_system = [m for m in msgs if m["role"] != "system"]
            trimmed = non_system[-(max_turns_context * 2):]
            msgs = system_msgs + trimmed
        # transformers' pipeline accepts chat messages directly.
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=self.device_map,
        )
        out = pipe(
            msgs, max_new_tokens=self.max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
        )
        generated = out[0]["generated_text"]
        reply = generated[-1]["content"] if generated and isinstance(generated[-1], dict) else ""
        return reply or ""


def build_hf_chatllm(
    base_model: str,
    adapter_path: str | Path | None = None,
    *,
    mock: bool = False,
    **kwargs: Any,
) -> ChatLLM:
    """Build a real HFChatLLM or a MockChatLLM depending on flags.

    Use ``--mock`` (or set ``NBF_MOCK=1``) to bypass real model
    loading; this is the path used in unit tests and CI.
    """
    if mock or is_mock_mode():
        # The mock returns scripted responses in order, falling back
        # to echoing the last user message.  Good enough for schema
        # and pipeline smoke tests.
        return MockChatLLM(responses=[])
    return HFChatLLM(base_model=base_model, adapter_path=adapter_path, **kwargs)
