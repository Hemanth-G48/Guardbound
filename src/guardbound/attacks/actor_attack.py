"""ActorAttack — STUB.

Paper reference: Ren et al., "An LLM can Fool Itself: A Prompt-based
Counterfactual Framework for Jailbreak Instructions", 2024. arXiv:2402.03007.

Official dataset and reference code:
    https://github.com/wjlan/LLM-Attacks-Two-Hop

NBF paper status:
    The NBF paper uses ActorAttack's released 2,327-conversation dataset
    for training but does NOT restate ActorAttack's generation algorithm.
    It refers readers to the original Ren et al. 2024 paper.

The exact decomposition strategy, clue-elimination logic, and turn
progression are therefore NOT SPECIFIED in the NBF paper.

This file ships a strict-paper stub: it implements only the
``MultiTurnAttack`` interface and raises ``NotImplementedError`` on
``next_query``. Paste the ActorAttack generator from Ren et al. 2024 (or
the released ``LLM-Attacks-Two-Hop`` repo) into ``next_query`` below.
"""
from __future__ import annotations

from ..schemas import Turn
from .base import MultiTurnAttack


class ActorAttack(MultiTurnAttack):
    """ActorAttack: actor-scenario decomposition (Ren et al. 2024) — STUB."""

    name = "actor_attack"

    def next_query(self, goal: str, history: list[Turn]) -> str:
        raise NotImplementedError(
            "ActorAttack.next_query is a paper-strict stub. "
            "Paste the official Ren et al. 2024 generator here. "
            "The NBF paper does not restate the ActorAttack algorithm — "
            "see https://github.com/wjlan/LLM-Attacks-Two-Hop for the "
            "original code/dataset."
        )

    def is_finished(self, history: list[Turn], max_turns: int = 8) -> bool:
        return len(history) >= max_turns
