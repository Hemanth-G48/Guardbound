"""``original`` baseline: pass-through defense.

The undefended target LLM.  No safety system prompt, no fine-tuning,
no NBF, no additional filtering.  This is the baseline every other
defense variant is compared against.
"""
from __future__ import annotations

from ..llm.base import ChatLLM
from .base import DefenseWrapper


class OriginalDefense(DefenseWrapper):
    """Identity wrapper for the undefended target LLM."""

    name = "original"

    def wrap(self, llm: ChatLLM) -> ChatLLM:
        return llm
