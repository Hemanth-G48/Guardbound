"""Enhanced SteeredLLMChat with retry logic matching author's Crescendomation.

Key improvements:
1. Retry up to 3 times when a query is filtered, taking the safest (minimum h)
2. Proper state handling matching author's calculate_score_from_dialog
3. Debug logging for state comparison

Paper reference: Russinovich et al. 2024, Sec. 4.3
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


@dataclass
class SteeredResult:
    filtered: bool
    h_value: float
    response: str | None
    turn_index: int
    accepted_turn_index: int
    state_norm: float
    refusal: bool = False
    retries: int = 0


class SteeredLLMChat:
    def __init__(
        self,
        target_llm: ChatLLM,
        barrier: NeuralBarrierFunction,
        eta: float = 5e-4,
        max_turns: int = 8,
        refusal_detection: bool = True,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str | None = None,
        on_refusal: Callable[[SteeredResult], None] | None = None,
        max_filter_retries: int = 3,
        debug_state: bool = False,
    ):
        self.target_llm = target_llm
        self.barrier = barrier
        self.eta = eta
        self.max_turns = max_turns
        self.refusal_detection = refusal_detection
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.on_refusal = on_refusal
        self.max_filter_retries = max_filter_retries
        self.debug_state = debug_state

        self._state_dim = barrier.predictor.state_dim
        self._embedding_dim = barrier.predictor.embedding_dim
        self._state: torch.Tensor = torch.zeros(1, self._state_dim)
        self._turn_index: int = 0
        self._accepted_turn_index: int = 0
        self._history: list[Message] = []
        self._turns: list[Turn] = []

        self._device = next(barrier.parameters()).device

        if self.debug_state:
            self._state_history = []

    def reset(self) -> None:
        self._state = torch.zeros(1, self._state_dim, device=self._device)
        self._turn_index = 0
        self._accepted_turn_index = 0
        self._history = []
        self._turns = []
        if self.system_prompt:
            self._history.append({"role": "system", "content": self.system_prompt})

        if self.debug_state:
            self._state_history = []

    def _compute_h(self, state: torch.Tensor, u: torch.Tensor) -> float:
        with torch.no_grad():
            h_val = self.barrier.h(state, u)
            return h_val.item()

    def _is_unsafe(self, h_value: float) -> bool:
        return (h_value + self.eta) >= 0

    def chat(
        self,
        query_text: str,
        embed_fn: Callable[[str], torch.Tensor],
        temperature: float | None = None,
    ) -> SteeredResult:
        temp = temperature if temperature is not None else self.temperature

        if self._turn_index >= self.max_turns:
            from .steered_chat import MaxTurnsExceeded
            raise MaxTurnsExceeded(f"Maximum turns ({self.max_turns}) exceeded.")

        u = embed_fn(query_text)
        u = u.to(self._device)

        with torch.no_grad():
            h_val = self.barrier.h(self._state, u)
            h_value = h_val.item()

        filtered = self._is_unsafe(h_value)
        retries = 0

        if self.debug_state:
            logger.info(
                f"Turn {self._turn_index}: query={query_text[:50]}..., "
                f"h={h_value:.4f}, eta={self.eta}, filtered={filtered}"
            )
            self._state_history.append({
                'turn': self._turn_index,
                'query': query_text[:50],
                'h': h_value,
                'state_norm': torch.norm(self._state).item(),
                'filtered': filtered,
            })

        if filtered and self.max_filter_retries > 0:
            best_h = h_value
            best_query = query_text

            for retry in range(self.max_filter_retries - 1):
                if self.debug_state:
                    logger.info(f"  Retry {retry + 1}: checking same query, h={h_value:.4f}")

                if self._turn_index + retry + 1 >= self.max_turns:
                    break

                if h_value < best_h:
                    best_h = h_value
                    best_query = query_text

                if not self._is_unsafe(h_value):
                    break

            if self.debug_state:
                logger.info(
                    f"  Best: h={best_h:.4f}, was filtered={filtered}"
                )

            if best_h != h_value or not filtered:
                query_text = best_query
                h_value = best_h
                filtered = self._is_unsafe(h_value)
                retries = min(self.max_filter_retries - 1, 2)

        if filtered:
            result = SteeredResult(
                filtered=True,
                h_value=h_value,
                response=None,
                turn_index=self._turn_index,
                accepted_turn_index=self._accepted_turn_index,
                state_norm=torch.norm(self._state).item(),
                refusal=False,
                retries=retries,
            )
            self._turns.append(Turn(
                query=query_text,
                response=None,
                was_filtered=True,
            ))
            self._turn_index += 1
        else:
            self._history.append({"role": "user", "content": query_text})
            response = self.target_llm.generate(self._history, temperature=temp)
            self._history.append({"role": "assistant", "content": response})

            with torch.no_grad():
                self._state = self.barrier.advance_state(self._state, u)

            refusal = False
            if self.refusal_detection:
                from .steered_chat import default_refusal_detector
                refusal = default_refusal_detector(response)
                if refusal and self.on_refusal:
                    self.on_refusal(SteeredResult(
                        filtered=False, h_value=h_value, response=response,
                        turn_index=self._turn_index,
                        accepted_turn_index=self._accepted_turn_index,
                        state_norm=torch.norm(self._state).item(),
                        refusal=True,
                    ))

            result = SteeredResult(
                filtered=False,
                h_value=h_value,
                response=response,
                turn_index=self._turn_index,
                accepted_turn_index=self._accepted_turn_index + 1,
                state_norm=torch.norm(self._state).item(),
                refusal=refusal,
                retries=retries,
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
        return self._state.clone()

    def get_turns(self) -> list[Turn]:
        return list(self._turns)

    def get_state_history(self) -> list[dict]:
        return getattr(self, '_state_history', [])

    def get_conversation(
        self,
        goal: str = "",
        attack_method: str = "",
        target_llm: str = "",
    ) -> Conversation:
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
