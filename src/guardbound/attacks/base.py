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
from typing import Optional

from ..llm.base import ChatLLM
from ..schemas import Turn


class MultiTurnAttack(ABC):
    """Abstract base class for multi-turn jailbreak attacks (Phase 6)."""

    name: str = "base"

    # Harness telemetry for NBF candidate filtering (official semantics are
    # unchanged; the runner appends one entry per scored candidate).
    # Official --safety_filtering: each candidate query is scored by the
    # barrier; candidates scoring >= threshold are filtered and regenerated,
    # never reaching the target. Recorded for JSONL reproducibility.
    nbf_candidate_scores: list[float] = []
    nbf_candidate_verdicts: list[bool] = []  # True = accepted, False = filtered

    def set_evaluator_llm(self, llm: ChatLLM) -> None:
        """Route all evaluator/judge calls to ``llm`` (Phase 8).

        Officially the attacker model doubles as the evaluator (one LLM
        generates and judges). The three-model configuration instead assigns
        evaluation to a separate physical model while keeping every evaluator
        subrole (rubric / refusal / disclaimer / ActorAttack step judge)
        logically distinct — same prompts, same schemas, same thresholds,
        different backend.

        Default behavior: a no-op. Attacks that support a separate evaluator
        override this (see ``_EvaluatorMixin``) — if they never receive one,
        the official single-LLM behavior is preserved unchanged.
        """
        # Default: no separate evaluator — evaluation stays on the attacker LLM.

    def get_evaluator_llm(self) -> Optional[ChatLLM]:
        """The evaluator backend in use, or None for the official default.

        ``None`` means evaluator calls route to the attacker LLM (official
        single-model behavior).
        """
        return None


class _EvaluatorMixin:
    """Mixin for attacks that can route judging to a separate evaluator LLM.
    Mixins hold no state themselves; concrete attacks initialize
    ``self._evaluator_llm = None`` and call ``set_evaluator_llm`` to override
    the official attacker-as-evaluator default.
    """

    _evaluator_llm: Optional[ChatLLM]

    def set_evaluator_llm(self, llm: ChatLLM) -> None:
        self._evaluator_llm = llm

    def get_evaluator_llm(self) -> Optional[ChatLLM]:
        return self._evaluator_llm

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
