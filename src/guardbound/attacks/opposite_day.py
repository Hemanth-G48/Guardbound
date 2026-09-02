"""Opposite-day attack — STUB.

Paper reference: Li et al., "A Wolf in Sheep's Clothing: Generalized Nested
Jailbreak Prompts can Fool Large Language Models Easily", 2024. arXiv:2311.08268.

NBF paper status:
    The NBF paper does NOT restate the Opposite-day (nested-attack)
    generation mechanism. It refers readers to the original Li et al. 2024b
    paper.

The exact rephrasing surface, nesting depth, and per-turn framing logic
are therefore NOT SPECIFIED in the NBF paper.

This file ships a strict-paper stub: it implements only the
``MultiTurnAttack`` interface and raises ``NotImplementedError`` on
``next_query``. Paste the nested-attack generator from Li et al. 2024b
into ``next_query`` below.
"""
from __future__ import annotations

from ..schemas import Turn
from .base import MultiTurnAttack


class OppositeDayAttack(MultiTurnAttack):
    """Opposite-day: opposite-perspective reframing (Li et al. 2024b) — STUB."""

    name = "opposite_day"

    def next_query(self, goal: str, history: list[Turn]) -> str:
        raise NotImplementedError(
            "OppositeDayAttack.next_query is a paper-strict stub. "
            "Paste the official Li et al. 2024b (Nested Attack) generator "
            "here. The NBF paper does not restate the opposite-day "
            "algorithm — see arXiv:2311.08268 for the original paper."
        )

    def is_finished(self, history: list[Turn], max_turns: int = 8) -> bool:
        return len(history) >= max_turns
