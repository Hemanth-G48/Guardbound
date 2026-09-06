"""Generic attack runner: executes attacks in bare LLM or NBF-steered mode.

Handles the multi-turn loop, conversation recording, filtering, and resume.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Optional

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
    allow_regeneration: bool = False,
    regeneration_max: int = 3,
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
    allow_regeneration : bool
        If True, regenerate turns when model refuses but NBF doesn't detect (paper spec).
    regeneration_max : int
        Maximum number of regeneration attempts per turn.

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
            allow_regeneration=allow_regeneration,
            regeneration_max=regeneration_max,
        )

        if hasattr(attack, "state_fn") and getattr(attack, "state_fn", None) is None:
            attack.state_fn = steered.get_state
        if hasattr(attack, "barrier") and getattr(attack, "barrier", None) is None:
            attack.barrier = barrier
        if hasattr(attack, "embed_fn") and getattr(attack, "embed_fn", None) is None:
            attack.embed_fn = embed_fn

    for turn_idx in range(max_turns):
        if attack.is_finished(history, max_turns):
            break

        query = attack.next_query(goal, history)
        if not query or not query.strip():
            logger.warning("Attack produced empty query at turn %d, stopping", turn_idx)
            break

        if use_steered:
            if hasattr(attack, "state_fn") and attack.state_fn is steered.get_state:
                pass
            result = steered.chat(query, embed_fn, temperature=temperature)

            if result.regenerate:
                logger.info("Turn %d: model refused but NBF didn't detect, regenerating (attempt %d/%d)",
                           turn_idx + 1, steered._regeneration_count, regeneration_max)
                continue

            response = result.response
            was_filtered = result.filtered
        else:
            messages.append({"role": "user", "content": query})
            response = target_llm.generate(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            was_filtered = False

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


async def run_attack_async(
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
    allow_regeneration: bool = False,
    regeneration_max: int = 3,
) -> Conversation:
    """Async version of run_attack for parallel execution."""
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
            allow_regeneration=allow_regeneration,
            regeneration_max=regeneration_max,
        )

        if hasattr(attack, "state_fn") and getattr(attack, "state_fn", None) is None:
            attack.state_fn = steered.get_state
        if hasattr(attack, "barrier") and getattr(attack, "barrier", None) is None:
            attack.barrier = barrier
        if hasattr(attack, "embed_fn") and getattr(attack, "embed_fn", None) is None:
            attack.embed_fn = embed_fn

    for turn_idx in range(max_turns):
        if attack.is_finished(history, max_turns):
            break

        query = attack.next_query(goal, history)
        if not query or not query.strip():
            logger.warning("Attack produced empty query at turn %d, stopping", turn_idx)
            break

        if use_steered:
            if hasattr(attack, "state_fn") and attack.state_fn is steered.get_state:
                pass
            result = steered.chat(query, embed_fn, temperature=temperature)

            if result.regenerate:
                logger.info("Turn %d: model refused but NBF didn't detect, regenerating (attempt %d/%d)",
                           turn_idx + 1, steered._regeneration_count, regeneration_max)
                continue

            response = result.response
            was_filtered = result.filtered
        else:
            messages.append({"role": "user", "content": query})
            if hasattr(target_llm, "generate_async"):
                response = await target_llm.generate_async(messages, temperature=temperature)
            else:
                response = target_llm.generate(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            was_filtered = False

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


async def _run_single_attack_async(
    attack: MultiTurnAttack,
    goal: str,
    target_llm: ChatLLM,
    embed_fn: Callable[[str], torch.Tensor] | None,
    barrier,
    eta: float,
    max_turns: int,
    temperature: float,
    target_llm_name: str,
    attack_method: str,
    output_path: Path,
) -> tuple[Conversation, bool]:
    """Run a single attack and append result to file. Returns (conversation, success)."""
    try:
        conv = await run_attack_async(
            attack=attack, goal=goal, target_llm=target_llm,
            embed_fn=embed_fn, barrier=barrier, eta=eta,
            max_turns=max_turns, temperature=temperature,
            target_llm_name=target_llm_name,
            attack_method=attack_method or attack.name,
        )
        _append_jsonl(output_path, conv)
        return (conv, True)
    except Exception as exc:
        logger.error("  -> FAILED: %s", exc)
        conv = Conversation(
            goal=goal, attack_method=attack_method or attack.name,
            target_llm=target_llm_name, max_turns=max_turns,
        )
        return (conv, False)


async def run_attack_batch_parallel(
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
    max_concurrent: int = 4,
) -> list[Conversation]:
    """Run attacks on multiple goals concurrently.

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
    max_concurrent : int
        Maximum concurrent attacks (default: 4).

    Returns
    -------
    list[Conversation]
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_goals: set[str] = set()
    if resume and output_path.exists():
        try:
            from ..schemas import load_conversations_jsonl
            for c in load_conversations_jsonl(output_path):
                existing_goals.add(c.goal)
            logger.info("Resume: %d existing conversations loaded", len(existing_goals))
        except Exception as exc:
            logger.warning("Could not load existing file for resume: %s", exc)

    pending_goals = []
    for i, goal in enumerate(goals):
        if goal in existing_goals:
            logger.info("Skip %d/%d (already done): %s", i + 1, len(goals), goal[:60])
            from ..schemas import load_conversations_jsonl
            if output_path.exists():
                for c in load_conversations_jsonl(output_path):
                    if c.goal == goal:
                        pending_goals.append((goal, c))
                        break
        else:
            pending_goals.append((goal, None))

    if dry_run:
        conversations = []
        for i, (goal, _) in enumerate(pending_goals):
            logger.info("[DRY RUN] %d/%d: %s", i + 1, len(pending_goals), goal[:60])
            conv = Conversation(
                goal=goal, attack_method=attack_method or attack.name,
                target_llm=target_llm_name, max_turns=max_turns,
            )
            conversations.append(conv)
        return conversations

    # Filter to goals that need running
    goals_to_run = [(goal, idx) for idx, (goal, conv) in enumerate(pending_goals) if conv is None]
    if not goals_to_run:
        return [conv for _, conv in pending_goals]

    # Process in batches
    completed = {idx: conv for idx, (_, conv) in enumerate(pending_goals) if conv is not None}
    total = len(goals)
    start_time = time.time()

    for batch_start in range(0, len(goals_to_run), max_concurrent):
        batch = goals_to_run[batch_start:batch_start + max_concurrent]
        tasks = [
            _run_single_attack_async(
                attack=attack, goal=goal_str, target_llm=target_llm,
                embed_fn=embed_fn, barrier=barrier, eta=eta,
                max_turns=max_turns, temperature=temperature,
                target_llm_name=target_llm_name,
                attack_method=attack_method or attack.name,
                output_path=output_path,
            )
            for goal_str, idx in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (goal, idx), result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error("Goal %d/%d FAILED: %s", idx + 1, total, goal[:60])
                conv = Conversation(
                    goal=goal, attack_method=attack_method or attack.name,
                    target_llm=target_llm_name, max_turns=max_turns,
                )
                completed[idx] = conv
            else:
                conv, success = result
                completed[idx] = conv
                elapsed = time.time() - start_time
                n_filtered = sum(1 for t in conv.turns if t.was_filtered)
                logger.info("  [%d/%d] %s -> %d turns, %d filtered (%.1fs)",
                           idx + 1, total, goal[:50], len(conv.turns), n_filtered, elapsed)

    # Sort by original index
    conversations = [completed[i] for i in sorted(completed.keys())]
    return conversations


def run_attack_with_backtracking(
    attack,
    goal: str,
    target_llm: ChatLLM,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    barrier=None,
    eta: float = 0.0,
    max_turns: int = 8,
    temperature: float = DEFAULT_TEMPERATURE,
    target_llm_name: str = "",
    attack_method: str = "",
    allow_regeneration: bool = True,
    regeneration_max: int = 3,
) -> Conversation:
    """Run attack with backtracking support for paper implementations.

    This runner supports CrescendoAttackPaper, OppositeDayAttack, ActorAttack,
    AcronymAttack and similar attacks that require dynamic question generation
    and backtracking on refusal.

    Includes safety filtering via SSM/NBF when barrier and embed_fn are provided.
    Implements paper Section B.1: If model refuses but NBF doesn't detect → regenerate turn.
    """
    from ..defense.steered_chat import SteeredLLMChat
    from .crescendo_paper import CrescendoAttackPaper
    from .opposite_day import OppositeDayAttack
    from .actor_attack import ActorAttack
    from .acronym import AcronymAttack

    supported_types = (CrescendoAttackPaper, OppositeDayAttack, ActorAttack, AcronymAttack)
    if not isinstance(attack, supported_types):
        raise ValueError(
            f"run_attack_with_backtracking requires one of: "
            f"CrescendoAttackPaper, OppositeDayAttack, ActorAttack, AcronymAttack"
        )

    history: list[Turn] = []
    messages: list[Message] = []
    use_steered = barrier is not None and embed_fn is not None
    use_safety_filter = barrier is not None and embed_fn is not None
    safety_threshold = -eta if eta > 0 else 0.0

    attack.reset()

    if hasattr(attack, 'set_safety_filter') and barrier is not None and embed_fn is not None:
        attack.set_safety_filter(embed_fn, barrier, safety_threshold)

    if use_steered:
        steered = SteeredLLMChat(
            target_llm=target_llm,
            barrier=barrier,
            eta=eta,
            max_turns=max_turns,
            temperature=temperature,
            allow_regeneration=allow_regeneration,
            regeneration_max=regeneration_max,
        )

    turn_num = 0
    last_response = ""
    regeneration_count = 0

    def calculate_score_from_dialog(dialog_hist, query):
        """Calculate SSM/NBF score for a query given dialog history.

        Implements the author's calculate_score_from_dialog function.
        Returns score where lower = more unsafe (needs filtering).
        """
        if embed_fn is None or barrier is None:
            return float('inf')

        import torch
        state_dim = barrier.predictor.state_dim
        x_t = torch.zeros(1, state_dim, device=next(barrier.parameters()).device)

        for dialog in dialog_hist:
            if dialog['role'] == 'user':
                x_t_prev = x_t.clone()
                u_t = embed_fn(dialog['content']).unsqueeze(0)
                u_t = u_t.to(x_t.device)

                with torch.no_grad():
                    nbf_output = barrier.predictor(x_t_prev, u_t)
                    probs = torch.softmax(nbf_output, dim=-1)
                    last_class_prob = probs[:, -1]
                    max_other_class_prob = torch.max(probs[:, :-1], dim=1).values
                    x_t = barrier.dynamics(x_t_prev, u_t)

        u_t = embed_fn(query).unsqueeze(0)
        u_t = u_t.to(x_t.device)

        with torch.no_grad():
            nbf_output = barrier.predictor(x_t, u_t)
            probs = torch.softmax(nbf_output, dim=-1)
            last_class_prob = probs[:, -1]
            max_other_class_prob = torch.max(probs[:, :-1], dim=1).values
            nbf_score = last_class_prob - max_other_class_prob

        return nbf_score[0].item()

    while turn_num < max_turns:
        turn_num += 1

        if hasattr(attack, 'generate_question_for_turn'):
            query, _ = attack.generate_question_for_turn(
                goal=goal,
                turn_num=turn_num,
                last_response=last_response,
            )
        else:
            query = attack.next_query(goal, history)

        if not query or not query.strip():
            logger.warning("Attack produced empty query at turn %d, stopping", turn_num)
            break

        num_filtering = 0

        if use_safety_filter and isinstance(attack, (CrescendoAttackPaper, OppositeDayAttack, AcronymAttack)):
            minimal_ssm_nbf_score = float('inf')
            min_score_query = None
            unsafe_cutoff = False

            for index_ in range(3):
                ssm_nbf_score = calculate_score_from_dialog(messages, query)
                if minimal_ssm_nbf_score > ssm_nbf_score:
                    minimal_ssm_nbf_score = ssm_nbf_score
                    min_score_query = query
                if ssm_nbf_score < safety_threshold:
                    break
                else:
                    logger.info("Trial %d: unsafe prompt filtered: %s, safety index: %s",
                               num_filtering, query[:50], ssm_nbf_score - safety_threshold)
                    unsafe_cutoff = True
                    break

            query = min_score_query if min_score_query else query
            if unsafe_cutoff:
                num_filtering += 1
                continue

        if use_steered:
            result = steered.chat(query, embed_fn, temperature=temperature)
            response = result.response
            was_filtered = result.filtered
        else:
            messages.append({"role": "user", "content": query})
            response = target_llm.generate(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            was_filtered = False

        score = attack.evaluate_response(query, response, goal)
        was_refusal = attack.check_refusal(query, response)

        if hasattr(attack, 'record_turn'):
            attack.record_turn(query, response, score)

        turn = Turn(
            query=query,
            response=response,
            was_filtered=was_filtered,
        )
        history.append(turn)
        last_response = response

        if score == 5:
            logger.info("Goal achieved at turn %d", turn_num)
            break

        if was_refusal and attack.should_backtrack():
            attack.increment_refusal()
            logger.info("Refusal detected at turn %d, backtracking (attempt %d)",
                       turn_num, attack.get_refusal_count())
            if hasattr(attack, 'generate_question_for_turn'):
                messages.append({"role": "user", "content": query})
                messages.append({"role": "assistant", "content": response})
            continue

        if allow_regeneration and was_refusal and not was_filtered:
            regeneration_count += 1
            logger.info("Turn %d: model refused but NBF didn't detect, regenerating (attempt %d/%d)",
                       turn_num, regeneration_count, regeneration_max)
            if regeneration_count >= regeneration_max:
                logger.info("Max regeneration attempts reached, continuing without regeneration")
                regeneration_count = 0
            else:
                if hasattr(attack, 'generate_question_for_turn'):
                    messages.append({"role": "user", "content": query})
                    messages.append({"role": "assistant", "content": response})
                turn_num -= 1
                continue

    return Conversation(
        goal=goal,
        attack_method=attack_method or attack.label if hasattr(attack, "label") else (attack_method or attack.name),
        target_llm=target_llm_name,
        turns=history,
        max_turns=max_turns,
    )


async def run_attack_with_backtracking_async(
    attack,
    goal: str,
    target_llm: ChatLLM,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    barrier=None,
    eta: float = 0.0,
    max_turns: int = 8,
    temperature: float = DEFAULT_TEMPERATURE,
    target_llm_name: str = "",
    attack_method: str = "",
    allow_regeneration: bool = True,
    regeneration_max: int = 3,
) -> Conversation:
    """Async version of run_attack_with_backtracking with safety filtering.

    Implements paper Section B.1: If model refuses but NBF doesn't detect → regenerate turn.
    """
    from ..defense.steered_chat import SteeredLLMChat
    from .crescendo_paper import CrescendoAttackPaper
    from .opposite_day import OppositeDayAttack
    from .actor_attack import ActorAttack
    from .acronym import AcronymAttack

    supported_types = (CrescendoAttackPaper, OppositeDayAttack, ActorAttack, AcronymAttack)
    if not isinstance(attack, supported_types):
        raise ValueError(
            f"run_attack_with_backtracking_async requires one of: "
            f"CrescendoAttackPaper, OppositeDayAttack, ActorAttack, AcronymAttack"
        )

    history: list[Turn] = []
    messages: list[Message] = []
    use_steered = barrier is not None and embed_fn is not None
    use_safety_filter = barrier is not None and embed_fn is not None
    safety_threshold = -eta if eta > 0 else 0.0

    attack.reset()

    if hasattr(attack, 'set_safety_filter') and barrier is not None and embed_fn is not None:
        attack.set_safety_filter(embed_fn, barrier, safety_threshold)

    if use_steered:
        steered = SteeredLLMChat(
            target_llm=target_llm,
            barrier=barrier,
            eta=eta,
            max_turns=max_turns,
            temperature=temperature,
            allow_regeneration=allow_regeneration,
            regeneration_max=regeneration_max,
        )

    turn_num = 0
    last_response = ""
    regeneration_count = 0

    def calculate_score_from_dialog(dialog_hist, query):
        """Calculate SSM/NBF score for a query given dialog history."""
        if embed_fn is None or barrier is None:
            return float('inf')

        import torch
        state_dim = barrier.predictor.state_dim
        x_t = torch.zeros(1, state_dim, device=next(barrier.parameters()).device)

        for dialog in dialog_hist:
            if dialog['role'] == 'user':
                x_t_prev = x_t.clone()
                u_t = embed_fn(dialog['content']).unsqueeze(0)
                u_t = u_t.to(x_t.device)

                with torch.no_grad():
                    nbf_output = barrier.predictor(x_t_prev, u_t)
                    probs = torch.softmax(nbf_output, dim=-1)
                    last_class_prob = probs[:, -1]
                    max_other_class_prob = torch.max(probs[:, :-1], dim=1).values
                    x_t = barrier.dynamics(x_t_prev, u_t)

        u_t = embed_fn(query).unsqueeze(0)
        u_t = u_t.to(x_t.device)

        with torch.no_grad():
            nbf_output = barrier.predictor(x_t, u_t)
            probs = torch.softmax(nbf_output, dim=-1)
            last_class_prob = probs[:, -1]
            max_other_class_prob = torch.max(probs[:, :-1], dim=1).values
            nbf_score = last_class_prob - max_other_class_prob

        return nbf_score[0].item()

    while turn_num < max_turns:
        turn_num += 1

        if hasattr(attack, 'generate_question_for_turn'):
            query, _ = attack.generate_question_for_turn(
                goal=goal,
                turn_num=turn_num,
                last_response=last_response,
            )
        else:
            query = attack.next_query(goal, history)

        if not query or not query.strip():
            logger.warning("Attack produced empty query at turn %d, stopping", turn_num)
            break

        num_filtering = 0

        if use_safety_filter and isinstance(attack, (CrescendoAttackPaper, OppositeDayAttack, AcronymAttack)):
            minimal_ssm_nbf_score = float('inf')
            min_score_query = None
            unsafe_cutoff = False

            for index_ in range(3):
                ssm_nbf_score = calculate_score_from_dialog(messages, query)
                if minimal_ssm_nbf_score > ssm_nbf_score:
                    minimal_ssm_nbf_score = ssm_nbf_score
                    min_score_query = query
                if ssm_nbf_score < safety_threshold:
                    break
                else:
                    logger.info("Trial %d: unsafe prompt filtered: %s, safety index: %s",
                               num_filtering, query[:50], ssm_nbf_score - safety_threshold)
                    unsafe_cutoff = True
                    break

            query = min_score_query if min_score_query else query
            if unsafe_cutoff:
                num_filtering += 1
                continue

        if use_steered:
            result = steered.chat(query, embed_fn, temperature=temperature)
            response = result.response
            was_filtered = result.filtered
        else:
            messages.append({"role": "user", "content": query})
            if hasattr(target_llm, "generate_async"):
                response = await target_llm.generate_async(messages, temperature=temperature)
            else:
                response = target_llm.generate(messages, temperature=temperature)
            messages.append({"role": "assistant", "content": response})
            was_filtered = False

        score = attack.evaluate_response(query, response, goal)
        was_refusal = attack.check_refusal(query, response)

        if hasattr(attack, 'record_turn'):
            attack.record_turn(query, response, score)

        turn = Turn(
            query=query,
            response=response,
            was_filtered=was_filtered,
        )
        history.append(turn)
        last_response = response

        if score == 5:
            logger.info("Goal achieved at turn %d", turn_num)
            break

        if was_refusal and attack.should_backtrack():
            attack.increment_refusal()
            logger.info("Refusal detected at turn %d, backtracking (attempt %d)",
                       turn_num, attack.get_refusal_count())
            if hasattr(attack, 'generate_question_for_turn'):
                messages.append({"role": "user", "content": query})
                messages.append({"role": "assistant", "content": response})
            continue

        if allow_regeneration and was_refusal and not was_filtered:
            regeneration_count += 1
            logger.info("Turn %d: model refused but NBF didn't detect, regenerating (attempt %d/%d)",
                       turn_num, regeneration_count, regeneration_max)
            if regeneration_count >= regeneration_max:
                logger.info("Max regeneration attempts reached, continuing without regeneration")
                regeneration_count = 0
            else:
                if hasattr(attack, 'generate_question_for_turn'):
                    messages.append({"role": "user", "content": query})
                    messages.append({"role": "assistant", "content": response})
                turn_num -= 1
                continue

    return Conversation(
        goal=goal,
        attack_method=attack_method or attack.label if hasattr(attack, "label") else (attack_method or attack.name),
        target_llm=target_llm_name,
        turns=history,
        max_turns=max_turns,
    )
