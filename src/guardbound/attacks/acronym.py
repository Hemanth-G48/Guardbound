"""Acronym attack — STUB (TRAINING ONLY per NBF paper).

Paper reference: Li et al., "A Wolf in Sheep's Clothing" (Opposite-day
paper family, Acronym variant). arXiv:2311.08268.

NBF paper status:
    The NBF paper uses Acronym only for training-data diversity
    (Sec. 5.1: training conversations against GPT-3.5-turbo) and
    EXPLICITLY EXCLUDES it from the standard evaluation suite (its
    single-turn success rate is too high and would contaminate eval).

The acronym spelling strategy, per-letter prompt wording, and turn
progression are NOT SPECIFIED in the NBF paper.

This file ships a strict-paper stub: it implements only the
``MultiTurnAttack`` interface and raises ``NotImplementedError`` on
``next_query``. Paste the acronym-style generator from Li et al. 2024b
into ``next_query`` below.

EXCLUSION:
    Acronym is NOT in ``DEFAULT_EVALUATION_ATTACKS`` in ``registry.py``.
"""
from __future__ import annotations

from ..schemas import Turn
from .base import MultiTurnAttack


class AcronymAttack(MultiTurnAttack):
    """Acronym: abbreviation / obfuscation (TRAINING ONLY) — STUB."""

    name = "acronym"

    def next_query(self, goal: str, history: list[Turn]) -> str:
        raise NotImplementedError(
            "AcronymAttack.next_query is a paper-strict stub. "
            "Paste the official Li et al. 2024b (Acronym variant) "
            "generator here. The NBF paper excludes Acronym from "
            "evaluation; it is used only for training-data diversity."
        )

    def is_finished(self, history: list[Turn], max_turns: int = 8) -> bool:
        return len(history) >= max_turns
