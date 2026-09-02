"""Generic attack runner: executes attacks in bare LLM or NBF-steered mode.

Handles the multi-turn loop, conversation recording, filtering, and resume.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

from ..llm.base import ChatLLM, DEFAULT_TEMPERATURE, Message
from ..logging_utils import get_logger
from ..schemas import Conversation, Turn
from .base import MultiTurnAttack

logger = get_logger(__name__)


def run_attack(
    attack: MultiTurnAttack,
    goal: str,
    target_llm: ChatLLM,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    barrier=None,
    eta: float = 0.0,
    max_turns: int = 8,
    temperature: float = DEFAULT_TEMPERATURE,
    target_llm_name: str = "",
    attack_method: str = "",
) -> Conversation:
    """Run a multi-turn attack conversation.

    Parameters
    ----------
    attack : MultiTurnAttack
        The attack strategy.
    goal : str
        The harmful behavior goal.
    target_llm : ChatLLM
        Target language model (used in bare mode).
    embed_fn : callable, optional
        Embedding function for NBF mode.
    barrier : NeuralBarrierFunction, optional
        NBF barrier for steered mode.
    eta : float
        Steering threshold.
    max_turns : int
        Maximum turns.
    temperature : float
        Generation temperature.
    target_llm_name : str
        Model identifier for logging.
    attack_method : str
        Attack method name.

    Returns
    -------
    Conversation
    """
    from ..defense.steered_chat import SteeredLLMChat

    history: list[Turn] = []
    messages: list[Message] = []
    use_steered = barrier is not None and embed_fn is not None

    if use_steered:
        steered = SteeredLLMChat(
            target_llm=target_llm,
            barrier=barrier,
            eta=eta,
            max_turns=max_turns,
            temperature=temperature,
        )

        # If the attack is adaptive, automatically wire it to the steered
        # chat's current state.  This is a small backward-compatible helper:
        # the adaptive attack never modifies the steered chat's state.
        if hasattr(attack, "state_fn") and getattr(attack, "state_fn", None) is None:
            attack.state_fn = steered.get_state
        if hasattr(attack, "barrier") and getattr(attack, "barrier", None) is None:
            attack.barrier = barrier
        if hasattr(attack, "embed_fn") and getattr(attack, "embed_fn", None) is None:
            attack.embed_fn = embed_fn

    for turn_idx in range(max_turns):
        if attack.is_finished(history, max_turns):
            break

        # Generate query
        query = attack.next_query(goal, history)
        if not query or not query.strip():
            logger.warning("Attack produced empty query at turn %d, stopping", turn_idx)
            break

        # Process through defense or directly
        if use_steered:
            # Refresh the adaptive attack's state view at every turn
            if hasattr(attack, "state_fn") and attack.state_fn is steered.get_state:
                pass  # already wired; state_fn reads the live steered state
            result = steered.chat(query, embed_fn, temperature=temperature)
            response = result.response
            was_filtered = result.filtered
        else:
            messages.append({"role": "user", "content": query})
            response = target_llm.generate(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            was_filtered = False

        # Record turn
        turn = Turn(
            query=query,
            response=response,
            was_filtered=was_filtered,
        )
        history.append(turn)

    return Conversation(
        goal=goal,
        attack_method=attack_method or attack.label if hasattr(attack, "label") else (attack_method or attack.name),
        target_llm=target_llm_name,
        turns=history,
        max_turns=max_turns,
    )


def run_attack_batch(
    attack: MultiTurnAttack,
    goals: list[str],
    target_llm: ChatLLM,
    output_path: Path,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    barrier=None,
    eta: float = 0.0,
    max_turns: int = 8,
    temperature: float = DEFAULT_TEMPERATURE,
    target_llm_name: str = "",
    attack_method: str = "",
    resume: bool = True,
    dry_run: bool = False,
) -> list[Conversation]:
    """Run attacks on multiple goals with resume support.

    Parameters
    ----------
    attack : MultiTurnAttack
    goals : list[str]
    target_llm : ChatLLM
    output_path : Path
        Output JSONL path.
    resume : bool
        Skip already-completed goals.
    dry_run : bool
        If True, don't execute attacks.

    Returns
    -------
    list[Conversation]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing for resume
    existing_goals: set[str] = set()
    if resume and output_path.exists():
        try:
            from ..schemas import load_conversations_jsonl
            for c in load_conversations_jsonl(output_path):
                existing_goals.add(c.goal)
            logger.info("Resume: %d existing conversations loaded", len(existing_goals))
        except Exception as exc:
            logger.warning("Could not load existing file for resume: %s", exc)

    conversations = []
    for i, goal in enumerate(goals):
        # Resume: skip completed
        if goal in existing_goals:
            logger.info("Skip %d/%d (already done): %s", i + 1, len(goals), goal[:60])
            # Load existing conversation
            from ..schemas import load_conversations_jsonl
            if output_path.exists():
                for c in load_conversations_jsonl(output_path):
                    if c.goal == goal:
                        conversations.append(c)
                        break
            continue

        if dry_run:
            logger.info("[DRY RUN] %d/%d: %s", i + 1, len(goals), goal[:60])
            conv = Conversation(
                goal=goal, attack_method=attack_method or attack.name,
                target_llm=target_llm_name, max_turns=max_turns,
            )
            conversations.append(conv)
            continue

        logger.info("Attack %d/%d: %s", i + 1, len(goals), goal[:60])
        try:
            conv = run_attack(
                attack=attack, goal=goal, target_llm=target_llm,
                embed_fn=embed_fn, barrier=barrier, eta=eta,
                max_turns=max_turns, temperature=temperature,
                target_llm_name=target_llm_name,
                attack_method=attack_method or attack.name,
            )
            conversations.append(conv)

            # Append to output (atomic-ish)
            _append_jsonl(output_path, conv)
            logger.info("  -> %d turns, %d filtered",
                        len(conv.turns),
                        sum(1 for t in conv.turns if t.was_filtered))

        except Exception as exc:
            logger.error("  -> FAILED: %s", exc)
            conv = Conversation(
                goal=goal, attack_method=attack_method or attack.name,
                target_llm=target_llm_name, max_turns=max_turns,
            )
            conversations.append(conv)

    return conversations


def _append_jsonl(path: Path, conv: Conversation) -> None:
    """Append a single conversation to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(conv.to_json() + "\n")
