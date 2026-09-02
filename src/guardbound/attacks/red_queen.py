"""RedQueen attack — STUB (unseen-attack generalization, Fig. 6).

Paper reference: Jiang et al., "RedQueen: A Generalized Attack on Generate
Your Own Prompts", 2024. arXiv:2401.16158.

NBF paper status:
    The NBF paper uses RedQueen at 1/3/4/5-turn settings for the
    unseen-attack generalization experiment (Fig. 6). The NBF paper does
    NOT restate RedQueen's generation algorithm; it refers readers to
    the original Jiang et al. 2024 paper.

The exact rephrasing mechanism, refusal-detection rule, and turn
progression are NOT SPECIFIED in the NBF paper.

This file ships a strict-paper stub: it implements only the
``MultiTurnAttack`` interface and raises ``NotImplementedError`` on
``next_query``. Paste the RedQueen generator from Jiang et al. 2024 into
``next_query`` below.

NOTE:
    RedQueen is NOT in ``DEFAULT_EVALUATION_ATTACKS`` in ``registry.py``.
    Per the NBF paper, RedQueen is the unseen-attack probe only (Fig. 6);
    it is invoked explicitly with ``--max-turns 1|3|4|5`` for the
    generalization experiment.
"""
from __future__ import annotations

from ..schemas import Turn
from .base import MultiTurnAttack


class RedQueenAttack(MultiTurnAttack):
    """RedQueen: feedback-driven rephraser (Jiang et al. 2024) — STUB."""

    name = "red_queen"

    def next_query(self, goal: str, history: list[Turn]) -> str:
        raise NotImplementedError(
            "RedQueenAttack.next_query is a paper-strict stub. "
            "Paste the official Jiang et al. 2024 generator here. "
            "The NBF paper uses RedQueen only for the unseen-attack "
            "generalization experiment (Fig. 6) at 1/3/4/5 turns."
        )

    def is_finished(self, history: list[Turn], max_turns: int = 8) -> bool:
        return len(history) >= max_turns
