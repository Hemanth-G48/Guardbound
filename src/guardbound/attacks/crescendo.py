"""Crescendo attack — STUB.

Paper reference: Russinovich, Salem, Eykholt, "Great, Now Write an Essay for
Me": Attacking LLMs with Multi-Turn Prompts, 2024. arXiv:2404.06773.

Official implementation: https://github.com/IBM/ai-anchoring
(Note: the official repository is Azure-Cognitive-Services-specific.)

NBF paper status:
    The NBF paper does NOT restate the Crescendo algorithm. It refers
    readers to the original paper. The exact generation mechanism,
    escalation cadence, and refusal-handling rules are therefore NOT
    SPECIFIED in the NBF paper.

This file ships a strict-paper stub: it implements only the
``MultiTurnAttack`` interface and raises ``NotImplementedError`` on
``next_query``. To make Phase 6 evaluation work, paste the Crescendo
generator from the original Russinovich et al. 2024 implementation (or a
faithful reimplementation of its mechanism) into ``next_query`` below.
"""
from __future__ import annotations

from ..schemas import Turn
from .base import MultiTurnAttack


class CrescendoAttack(MultiTurnAttack):
    """Crescendo: gradual escalation (Russinovich et al. 2024) — STUB."""

    name = "crescendo"

    def next_query(self, goal: str, history: list[Turn]) -> str:
        raise NotImplementedError(
            "CrescendoAttack.next_query is a paper-strict stub. "
            "Paste the official Russinovich et al. 2024 generator here. "
            "The NBF paper does not restate the Crescendo algorithm — see "
            "https://github.com/IBM/ai-anchoring for the original code."
        )

    def is_finished(self, history: list[Turn], max_turns: int = 8) -> bool:
        return len(history) >= max_turns
