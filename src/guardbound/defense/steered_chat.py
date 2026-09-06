"""Inference-time NBF safety steering: SteeredLLMChat.

Implements the Q-filter defense from Sec. 4.3:

    At turn k:
        1. Embed query: u_k = embedding(U_k)
        2. Evaluate: h_k = h(x_{k-1}, u_k)
        3. If h_k + eta >= 0: FILTER (no LLM call, no state update)
        4. Else: call LLM, advance state x_k = f_theta(x_{k-1}, u_k)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..llm.base import ChatLLM, DEFAULT_TEMPERATURE, Message
from ..logging_utils import get_logger
from ..models.predictor import NeuralBarrierFunction
from ..schemas import Conversation, Turn

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Max turns exceeded
# --------------------------------------------------------------------------- #

class MaxTurnsExceeded(Exception):
    """Raised when the maximum number of turns is reached."""
    pass


# --------------------------------------------------------------------------- #
# Steered result
# --------------------------------------------------------------------------- #

@dataclass
class SteeredResult:
    """Result of processing a single query through the NBF defense."""
    filtered: bool
    h_value: float
    response: str | None
    turn_index: int              # processed-turn counter (includes filtered)
    accepted_turn_index: int     # dynamics state counter (accepted only)
    state_norm: float
    refusal: bool = False        # True if LLM response was a refusal
    regenerate: bool = False     # True if turn should be regenerated (refusal but not filtered)


# --------------------------------------------------------------------------- #
# Refusal detection
# --------------------------------------------------------------------------- #

# Not specified in the paper — default: conservative regex/string matching detector.
_REFUSAL_PATTERNS = [
    r"\bI('m| am) sorry\b",
    r"\bI cannot\b",
    r"\bI can('t|not)\b",
    r"\bI am unable\b",
    r"\bI won('t|t)\b",
    r"\bI cannot assist\b",
    r"\bI can('t|not) help\b",
    r"\bI('m| am) not able\b",
    r"\bI('m| am) unable to\b",
    r"\bI do not (feel |comfortable |able )",
    r"\bI('m| am) programmed to (not |decline )",
    r"\bI must (decline|refuse)\b",
    r"\bI('m| am) designed to (not |refuse )",
    r"\bAs an AI\b",
    r"\bI don('t|not) have (the |any )?ability\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def default_refusal_detector(response: str) -> bool:
    """Detect if an LLM response is a refusal.

    Not specified in the paper — default: conservative regex/string matching.
    """
    return bool(_REFUSAL_RE.search(response))


# --------------------------------------------------------------------------- #
# SteeredLLMChat
# --------------------------------------------------------------------------- #

class SteeredLLMChat:
    """Stateful NBF defense wrapper around any ChatLLM.

    Maintains dialogue state x, applies the Q-filter before each LLM call,
    and records all turns for downstream evaluation.

    For refusal detection, supports two modes:
    - regex: Conservative regex matching (default, not paper-specified)
    - llm: LLM-based detection per the author's implementation (B.1)
    """

    def __init__(
        self,
        target_llm: ChatLLM,
        barrier: NeuralBarrierFunction,
        eta: float = 5e-4,
        max_turns: int = 8,
        refusal_detection: bool = True,
        refusal_mode: str = "regex",
        refusal_llm: ChatLLM | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str | None = None,
        on_refusal: Callable[[SteeredResult], None] | None = None,
        allow_regeneration: bool = False,
        regeneration_max: int = 3,
    ):
        self.target_llm = target_llm
        self.barrier = barrier
        self.eta = eta
        self.max_turns = max_turns
        self.refusal_detection = refusal_detection
        self.refusal_mode = refusal_mode
        self.refusal_llm = refusal_llm
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.on_refusal = on_refusal
        self.allow_regeneration = allow_regeneration
        self.regeneration_max = regeneration_max
        self._regeneration_count = 0

        # Set up LLM-based refusal checker if requested
        self._refusal_checker = None
        if refusal_mode == "llm" and refusal_llm is not None:
            from ..evaluation.refusal_checker import RefusalChecker
            self._refusal_checker = RefusalChecker(refusal_llm)

        # Device first (needed for state initialization)
        self._device = next(barrier.parameters()).device

        # State
        self._state_dim = barrier.predictor.state_dim
        self._embedding_dim = barrier.predictor.embedding_dim
        self._state: torch.Tensor = torch.zeros(1, self._state_dim, device=self._device)
        self._turn_index: int = 0          # processed-turn counter
        self._accepted_turn_index: int = 0  # dynamics state counter
        self._history: list[Message] = []
        self._turns: list[Turn] = []

    def reset(self) -> None:
        """Reset state to x_0 = zeros(768) and clear all counters."""
        self._state = torch.zeros(1, self._state_dim, device=self._device)
        self._turn_index = 0
        self._accepted_turn_index = 0
        self._history = []
        self._turns = []
        self._regeneration_count = 0
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})

    def chat(
        self,
        query_text: str,
        embed_fn: Callable[[str], torch.Tensor],
        temperature: float | None = None,
    ) -> SteeredResult:
        """Process a single query through the NBF Q-filter.

        Parameters
        ----------
        query_text : str
            The user's query.
        embed_fn : callable
            Function that maps text -> tensor [1, embedding_dim].
        temperature : float, optional
            Override generation temperature for this turn.

        Returns
        -------
        SteeredResult
        """
        temp = temperature if temperature is not None else self.temperature

        # Step 1: Check max turns
        if self._turn_index >= self.max_turns:
            raise MaxTurnsExceeded(
                f"Maximum turns ({self.max_turns}) exceeded. "
                f"Processed {self._turn_index} turns."
            )

        # Step 2: Embed query
        u = embed_fn(query_text)  # [1, emb_dim]
        u = u.to(self._device)

        # Step 3: Evaluate barrier (no grad for inference)
        with torch.no_grad():
            h_val = self.barrier.h(self._state, u)
            h_value = h_val.item()

        # Step 4: Q-filter decision
        filtered = (h_value + self.eta) >= 0

        # Step 5/6: Execute based on decision
        if filtered:
            # FILTERED: no LLM call, no state update
            result = SteeredResult(
                filtered=True,
                h_value=h_value,
                response=None,
                turn_index=self._turn_index,
                accepted_turn_index=self._accepted_turn_index,
                state_norm=torch.norm(self._state).item(),
                refusal=False,
            )
            self._turns.append(Turn(
                query=query_text,
                response=None,
                was_filtered=True,
            ))
            self._turn_index += 1
        else:
            # ACCEPTED: call LLM
            # Save state for potential rollback (paper: regenerate on refusal not caught by NBF)
            state_before = self._state.clone()
            history_len_before = len(self._history)

            self._history.append({"role": "user", "content": query_text})
            response = self.target_llm.generate(self._history, temperature=temp)
            self._history.append({"role": "assistant", "content": response})

            # Step 7: Advance state
            with torch.no_grad():
                self._state = self.barrier.advance_state(self._state, u)

            # Detect refusal
            refusal = False
            if self.refusal_detection:
                if self.refusal_mode == "llm" and self._refusal_checker is not None:
                    # Use LLM-based refusal detection (author's implementation)
                    refusal = self._refusal_checker.is_refusal(query_text, response)
                else:
                    # Use regex-based fallback
                    refusal = default_refusal_detector(response)
                if refusal and self.on_refusal:
                    self.on_refusal(SteeredResult(
                        filtered=False, h_value=h_value, response=response,
                        turn_index=self._turn_index,
                        accepted_turn_index=self._accepted_turn_index,
                        state_norm=torch.norm(self._state).item(),
                        refusal=True,
                    ))

            # Paper spec: "If a rejective response is generated by LLMs but not
            # detected by the neural barrier function, this turn will not count,
            # and another attack query will be regenerated."
            should_regenerate = (
                self.allow_regeneration
                and refusal
                and not filtered
                and self._regeneration_count < self.regeneration_max
            )

            if should_regenerate:
                # Rollback state and history
                self._state = state_before
                while len(self._history) > history_len_before:
                    self._history.pop()
                self._regeneration_count += 1

                result = SteeredResult(
                    filtered=False,
                    h_value=h_value,
                    response=response,
                    turn_index=self._turn_index,
                    accepted_turn_index=self._accepted_turn_index,
                    state_norm=torch.norm(self._state).item(),
                    refusal=refusal,
                    regenerate=True,
                )
            else:
                result = SteeredResult(
                    filtered=False,
                    h_value=h_value,
                    response=response,
                    turn_index=self._turn_index,
                    accepted_turn_index=self._accepted_turn_index + 1,
                    state_norm=torch.norm(self._state).item(),
                    refusal=refusal,
                    regenerate=False,
                )
                self._turns.append(Turn(
                    query=query_text,
                    response=response,
                    was_filtered=False,
                ))
                self._turn_index += 1
                self._accepted_turn_index += 1

        return result

    def get_state(self) -> torch.Tensor:
        """Return the current latent state (detached clone)."""
        return self._state.clone()

    def get_turns(self) -> list[Turn]:
        """Return the recorded turns."""
        return list(self._turns)

    def get_conversation(
        self,
        goal: str = "",
        attack_method: str = "",
        target_llm: str = "",
    ) -> Conversation:
        """Build a Conversation from recorded turns."""
        return Conversation(
            goal=goal,
            attack_method=attack_method,
            target_llm=target_llm,
            turns=list(self._turns),
            max_turns=self.max_turns,
        )

    @property
    def turn_index(self) -> int:
        return self._turn_index

    @property
    def accepted_turn_index(self) -> int:
        return self._accepted_turn_index
