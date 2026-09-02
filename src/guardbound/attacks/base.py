"""Common attack interface: MultiTurnAttack ABC.

Every attack adapter implements next_query() and is_finished().
The runner orchestrates the multi-turn conversation loop.

This module intentionally does NOT provide a backward-compatible
``generate(goal, target_llm, max_turns, temperature) -> Conversation``
loop.  Phase 2's ``data/attack_runner.py`` defines its own
``MultiTurnAttack`` ABC with a ``generate()`` method; the Phase 2 tests
expect that legacy interface.  The two attack interfaces coexist: Phase
6 attacks implement ``next_query`` only and are driven by
``attacks.runner.run_attack``; Phase 2 attacks implement ``generate``
and are driven by ``data.attack_runner.AttackRunner``.  Migrating the
Phase 2 tests to the new interface is tracked separately.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import Turn


class MultiTurnAttack(ABC):
    """Abstract base class for multi-turn jailbreak attacks (Phase 6)."""

    name: str = "base"

    @abstractmethod
    def next_query(
        self,
        goal: str,
        history: list[Turn],
    ) -> str:
        """Generate the next attacker query.

        Parameters
        ----------
        goal : str
            The harmful behavior goal.
        history : list[Turn]
            Previous turns in the conversation.

        Returns
        -------
        str
            The next query. Must be non-empty.
        """
        ...

    def is_finished(
        self,
        history: list[Turn],
        max_turns: int = 8,
    ) -> bool:
        """Check whether the attack should terminate.

        Default: terminate when max_turns is reached.
        """
        return len(history) >= max_turns
