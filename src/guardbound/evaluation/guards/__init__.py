"""Guardrail package: PromptGuard ABC and baseline backends.

Backends:
    - OpenAIModerationGuard (OpenAI Moderation API)
    - ShieldGemmaGuard (google/shieldgemma-2b)
    - LlamaGuardGuard (meta-llama/LlamaGuard-7b)

All backends are inference-only.  Tests use mocks; real evaluation
requires the relevant credentials and/or model weights.  The guards
NEVER silently substitute another model — see each module for the
exact error path.
"""
from .base import PromptGuard
from .openai_moderation import OpenAIModerationGuard
from .shieldgemma import ShieldGemmaGuard
from .llama_guard import LlamaGuardGuard

__all__ = [
    "PromptGuard",
    "OpenAIModerationGuard",
    "ShieldGemmaGuard",
    "LlamaGuardGuard",
]
