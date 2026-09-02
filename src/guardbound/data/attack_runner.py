"""Common interface and runner for multi-turn jailbreak attacks.

The ``MultiTurnAttack`` ABC defines the contract every attack adapter must
implement.  ``AttackRunner`` handles target-model communication, persistence,
caching, checkpointing, retry logic, and resume — so individual adapters only
need to implement attack-specific prompt construction and turn progression.

Phase 6 update: New attacks use ``attacks.base.MultiTurnAttack`` with
``next_query``/``is_finished`` pattern. This module retains backward
compatibility with Phase 2's ``generate()``-based interface.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.base import ChatLLM, DEFAULT_TEMPERATURE
from ..logging_utils import get_logger
from ..schemas import Conversation, Turn

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Common attack interface (Phase 2 style — kept for backward compat)
# --------------------------------------------------------------------------- #

class MultiTurnAttack(ABC):
    """Every attack adapter must implement this interface.

    This is the Phase 2 ``generate()``-based interface retained for backward
    compatibility with Phase 2 scripts and tests.
    """

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        goal: str,
        target_llm: ChatLLM,
        max_turns: int = 8,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Conversation:
        """Run a multi-turn attack conversation.

        Parameters
        ----------
        goal:
            The harmful behavior the attacker is trying to elicit.
        target_llm:
            The target language model to attack.
        max_turns:
            Maximum number of user/assistant turn pairs.
        temperature:
            Generation temperature.

        Returns
        -------
        A complete Conversation with all turns recorded.
        """
        ...


# --------------------------------------------------------------------------- #
# Cache key computation
# --------------------------------------------------------------------------- #

def _cache_key(
    attack_method: str,
    goal: str,
    target_model: str,
    temperature: float,
    max_turns: int,
    extra: str = "",
) -> str:
    """Deterministic SHA-256 cache key from all generation inputs."""
    payload = json.dumps({
        "attack": attack_method,
        "goal": goal,
        "model": target_model,
        "temperature": temperature,
        "max_turns": max_turns,
        "extra": extra,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Attack runner
# --------------------------------------------------------------------------- #

@dataclass
class GenerationResult:
    """Outcome of generating a single conversation."""
    conversation: Conversation
    from_cache: bool = False
    error: str | None = None
    retries: int = 0


class AttackRunner:
    """Orchestrates conversation generation with caching and resume.

    The runner handles:
    - target-model communication
    - per-conversation caching
    - incremental JSONL persistence
    - checkpoint tracking
    - resume after interruption
    - retry with backoff
    - logging
    """

    def __init__(
        self,
        attack: MultiTurnAttack,
        target_llm: ChatLLM,
        output_path: Path,
        cache_dir: Path,
        target_model: str = "gpt-3.5-turbo-0125",
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns: int = 8,
        retry_max: int = 3,
        retry_backoff_base: float = 2.0,
        resume: bool = True,
    ):
        self.attack = attack
        self.target_llm = target_llm
        self.output_path = Path(output_path)
        self.cache_dir = Path(cache_dir)
        self.target_model = target_model
        self.temperature = temperature
        self.max_turns = max_turns
        self.retry_max = retry_max
        self.retry_backoff_base = retry_backoff_base
        self.resume = resume

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing conversations for resume
        self._existing_goals: set[str] = set()
        if self.resume:
            self._load_existing()

    def _load_existing(self) -> None:
        """Load already-generated conversations to support resume."""
        if not self.output_path.exists():
            return
        try:
            from ..schemas import load_conversations_jsonl
            existing = load_conversations_jsonl(self.output_path)
            self._existing_goals = {c.goal for c in existing}
            logger.info(
                "Loaded %d existing conversations from %s (resume mode)",
                len(existing), self.output_path,
            )
        except Exception as exc:
            logger.warning("Could not load existing conversations for resume: %s", exc)

    def _get_cached(self, cache_key: str) -> Conversation | None:
        """Check if a conversation is already cached."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                return Conversation.from_dict(data)
            except Exception:
                return None
        return None

    def _save_cache(self, cache_key: str, conv: Conversation) -> None:
        """Save a conversation to the cache."""
        cache_path = self.cache_dir / f"{cache_key}.json"
        cache_path.write_text(conv.to_json())

    def _append_jsonl(self, conv: Conversation) -> None:
        """Atomically append a conversation to the output JSONL."""
        tmp_path = self.output_path.with_suffix(".tmp")
        # Write existing + new
        existing = []
        if self.output_path.exists():
            from ..schemas import load_conversations_jsonl
            existing = load_conversations_jsonl(self.output_path)
        existing.append(conv)
        from ..schemas import save_conversations_jsonl
        save_conversations_jsonl(existing, tmp_path)
        # Atomic rename
        tmp_path.replace(self.output_path)

    def generate_one(
        self,
        goal: str,
        dry_run: bool = False,
    ) -> GenerationResult:
        """Generate a single conversation, with caching and retry.

        Returns a GenerationResult with the conversation or error info.
        """
        # Skip if already generated (resume support)
        if goal in self._existing_goals:
            logger.info("Skipping already-generated goal: %s", goal[:60])
            # Find and return existing
            from ..schemas import load_conversations_jsonl
            if self.output_path.exists():
                for c in load_conversations_jsonl(self.output_path):
                    if c.goal == goal:
                        return GenerationResult(conversation=c, from_cache=True)
            # Fallback: shouldn't happen
            return GenerationResult(
                conversation=Conversation(goal=goal, attack_method=self.attack.name,
                                          target_llm=self.target_model),
                error="Goal in existing set but not found in file",
            )

        # Check generation cache
        key = _cache_key(
            self.attack.name, goal, self.target_model,
            self.temperature, self.max_turns,
        )
        cached = self._get_cached(key)
        if cached is not None:
            logger.info("Cache hit for goal: %s", goal[:60])
            return GenerationResult(conversation=cached, from_cache=True)

        if dry_run:
            logger.info("[DRY RUN] Would generate conversation for goal: %s", goal[:60])
            return GenerationResult(
                conversation=Conversation(
                    goal=goal, attack_method=self.attack.name,
                    target_llm=self.target_model, max_turns=self.max_turns,
                ),
            )

        # Generate with retry
        last_error = None
        for attempt in range(self.retry_max + 1):
            try:
                conv = self.attack.generate(
                    goal=goal,
                    target_llm=self.target_llm,
                    max_turns=self.max_turns,
                    temperature=self.temperature,
                )
                # Validate
                if not self._validate_conversation(conv):
                    raise ValueError(f"Generated conversation failed validation for goal: {goal[:60]}")

                # Cache and persist
                self._save_cache(key, conv)
                self._append_jsonl(conv)
                self._existing_goals.add(goal)

                return GenerationResult(conversation=conv, retries=attempt)

            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Attempt %d/%d failed for goal '%s': %s",
                    attempt + 1, self.retry_max + 1, goal[:60], exc,
                )
                if attempt < self.retry_max:
                    wait = self.retry_backoff_base ** attempt
                    logger.info("Retrying in %.1fs...", wait)
                    time.sleep(wait)

        return GenerationResult(
            conversation=Conversation(
                goal=goal, attack_method=self.attack.name,
                target_llm=self.target_model, max_turns=self.max_turns,
            ),
            error=last_error,
            retries=self.retry_max,
        )

    def generate_batch(
        self,
        goals: list[str],
        dry_run: bool = False,
    ) -> list[GenerationResult]:
        """Generate conversations for a batch of goals."""
        results = []
        for i, goal in enumerate(goals):
            logger.info("Generating %d/%d: %s", i + 1, len(goals), goal[:60])
            result = self.generate_one(goal, dry_run=dry_run)
            results.append(result)
            if result.error:
                logger.error("Failed goal '%s': %s", goal[:60], result.error)
        return results

    @staticmethod
    def _validate_conversation(conv: Conversation) -> bool:
        """Validate a generated conversation meets minimum requirements."""
        if not conv.goal:
            return False
        if not conv.attack_method:
            return False
        if not conv.target_llm:
            return False
        if len(conv.turns) == 0:
            return False
        for turn in conv.turns:
            if not turn.query:
                return False
            if turn.response is None:
                return False
        return True
