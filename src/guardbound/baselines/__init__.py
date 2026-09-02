"""Phase 8 — Defense baselines.

Registry of defense variants that all expose a ``ChatLLM``-compatible
target.  Phase 7 and Phase 9 evaluate every variant through the same
attack+metric pipeline; the only difference is the wrapper here.

Variants
--------
- ``original``      — pass-through; no behavior change.
- ``system_prompt`` — prepends the Llama-2-Chat safety system prompt.
- ``lora_sft``      — base model + LoRA SFT adapter (HF backend).
- ``lora_dpo``      — base model + LoRA DPO adapter (HF backend).
- ``lora_kto``      — base model + LoRA KTO adapter (HF backend).
- ``guardbound``  — delegates to Phase 5 ``SteeredLLMChat``.
"""
from .base import DefenseWrapper
from .registry import (
    BASELINE_VARIANTS,
    DEFAULT_BASELINE_VARIANTS,
    build_defense,
    list_variants,
)

__all__ = [
    "DefenseWrapper",
    "BASELINE_VARIANTS",
    "DEFAULT_BASELINE_VARIANTS",
    "build_defense",
    "list_variants",
]
