"""Defense-variant registry (Phase 8).

Every variant is a ``DefenseWrapper`` that turns a target ``ChatLLM``
into a defended ``ChatLLM``.  Phase 6 attacks and Phase 7 metrics are
unaware of which variant is in use; the registry is the only
component that needs to know the variant details.

Variants
--------
- ``original``      — pass-through.
- ``system_prompt`` — Llama-2 safety system prompt (see
  ``prompts/llama2_safety_system.txt``).
- ``lora_sft``      — base model + LoRA SFT adapter (HF backend).
- ``lora_dpo``      — base model + LoRA DPO adapter.
- ``lora_kto``      — base model + LoRA KTO adapter.
- ``guardbound``  — delegates to Phase 5 ``SteeredLLMChat``.

The ``build_defense(variant, llm, ...)`` factory returns a
``ChatLLM``-compatible object for the requested variant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..llm.base import ChatLLM
from ..logging_utils import get_logger
from .base import DefenseWrapper
from .original import OriginalDefense
from .serving import build_hf_chatllm
from .system_prompt import SystemPromptDefense

logger = get_logger(__name__)


# Public list of registered variant names.
BASELINE_VARIANTS: list[str] = [
    "original",
    "system_prompt",
    "lora_sft",
    "lora_dpo",
    "lora_kto",
    "guardbound",
]

# Default evaluation matrix (paper Sec. 5.1 / B.1 — defense baselines).
DEFAULT_BASELINE_VARIANTS: list[str] = [
    "original",
    "system_prompt",
    "lora_sft",
    "lora_dpo",
    "lora_kto",
    "guardbound",
]


def list_variants() -> list[str]:
    """Return the names of all registered defense variants."""
    return list(BASELINE_VARIANTS)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_defense(
    variant: str,
    llm: ChatLLM,
    *,
    checkpoint: str | Path | None = None,
    base_model: str | Path | None = None,
    eta: float | None = None,
    barrier: Any = None,
    embed_fn: Any = None,
    system_prompt_path: str | Path | None = None,
    mock: bool = False,
    **kwargs: Any,
) -> ChatLLM:
    """Build a defended ``ChatLLM`` for ``variant``.

    Parameters
    ----------
    variant
        One of ``BASELINE_VARIANTS``.
    llm
        The base target LLM.  Used for ``original`` and
        ``system_prompt``; for ``lora_*`` the base model is taken
        from ``base_model`` (or ``llm.model_id``) and the LoRA
        adapter from ``checkpoint``.
    checkpoint
        Path to a LoRA adapter directory (``lora_sft`` / ``lora_dpo``
        / ``lora_kto``) OR a path to an NBF predictor
        (``guardbound``).
    base_model
        Required for ``lora_*`` when the supplied ``llm`` does not
        expose ``model_id`` (e.g. ``MockChatLLM``).
    eta
        Required for ``guardbound``; defaults to 0.0.
    barrier
        Required for ``guardbound``; a ``NeuralBarrierFunction``.
    embed_fn
        Required for ``guardbound``; text -> tensor[1, 768].
    system_prompt_path
        Optional override of the default Llama-2 safety prompt file.
    mock
        If True (or the env var ``NBF_MOCK=1`` is set), LoRA loading
        is bypassed and a ``MockChatLLM`` is returned.
    """
    variant = variant.lower()
    if variant == "original":
        return OriginalDefense().wrap(llm)

    if variant == "system_prompt":
        return SystemPromptDefense(
            prompt_path=system_prompt_path,
        ).wrap(llm)

    if variant in {"lora_sft", "lora_dpo", "lora_kto"}:
        # Resolve base model and adapter.
        base_model_id = base_model or getattr(llm, "model_id", None)
        if not base_model_id:
            raise ValueError(
                f"Variant '{variant}' requires a base_model.  Pass "
                f"base_model=... or use a ChatLLM with .model_id."
            )
        if checkpoint is None:
            raise ValueError(
                f"Variant '{variant}' requires a LoRA adapter path "
                f"(--checkpoint)."
            )
        return build_hf_chatllm(
            base_model=str(base_model_id),
            adapter_path=checkpoint,
            mock=mock,
            **{k: v for k, v in kwargs.items()
               if k in {"device_map", "max_new_tokens", "torch_dtype"}},
        )

    if variant == "guardbound":
        if barrier is None or embed_fn is None:
            raise ValueError(
                "Variant 'guardbound' requires `barrier` and `embed_fn`."
            )
        from ..defense.steered_chat import SteeredLLMChat
        return SteeredLLMChat(
            target_llm=llm,
            barrier=barrier,
            eta=eta if eta is not None else 0.0,
            max_turns=8,
            temperature=0.7,
        )

    raise ValueError(
        f"Unknown defense variant '{variant}'.  "
        f"Available: {BASELINE_VARIANTS}"
    )


# --------------------------------------------------------------------------- #
# Result tag
# --------------------------------------------------------------------------- #

def results_tag_for(variant: str, checkpoint: str | Path | None = None) -> str:
    """Stable tag for a variant, optionally with a checkpoint suffix."""
    tag = variant
    if checkpoint is not None:
        chk = Path(checkpoint).name or "ckpt"
        tag = f"{variant}__{chk}"
    return tag
