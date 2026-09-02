"""Phase 6/7 evaluation runner interface.

Provides run_attack_under_steering() for running attacks through the NBF
defense. The attack object is treated as an external dependency — this
module does NOT implement attack algorithms.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

from ..llm.base import ChatLLM
from ..logging_utils import get_logger
from ..models.predictor import NeuralBarrierFunction
from ..schemas import Conversation, Turn
from .steered_chat import SteeredLLMChat

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Attack protocol (minimal interface for Phase 6)
# --------------------------------------------------------------------------- #

class AttackHarness(Protocol):
    """Minimal interface that attack implementations must satisfy.

    Phase 6 will implement concrete attacks conforming to this protocol.
    """

    def generate(
        self,
        goal: str,
        target_llm: ChatLLM,
        max_turns: int = 8,
        temperature: float = 0.7,
    ) -> Conversation:
        """Generate a multi-turn attack conversation."""
        ...


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run_attack_under_steering(
    attack: Any,
    target_llm: ChatLLM,
    barrier: NeuralBarrierFunction,
    eta: float,
    embed_fn: Callable[[str], torch.Tensor],
    max_turns: int = 8,
    temperature: float = 0.7,
    goal: str = "",
    attack_method: str = "",
    target_llm_name: str = "",
) -> Conversation:
    """Run an attack through the NBF defense.

    Parameters
    ----------
    attack : AttackHarness or similar
        Attack object with generate() method.
    target_llm : ChatLLM
        Target language model.
    barrier : NeuralBarrierFunction
        The NBF defense bundle.
    eta : float
        Steering threshold.
    embed_fn : callable
        Embedding function: text -> tensor [1, embedding_dim].
    max_turns : int
        Maximum attack turns.
    temperature : float
        Generation temperature.
    goal : str
        Attack goal description.
    attack_method : str
        Attack method name.
    target_llm_name : str
        Target LLM identifier.

    Returns
    -------
    Conversation
        The steered conversation with filtering decisions recorded.
    """
    # Create steered chat
    chat = SteeredLLMChat(
        target_llm=target_llm,
        barrier=barrier,
        eta=eta,
        max_turns=max_turns,
        temperature=temperature,
    )

    # Generate attack conversation
    conv = attack.generate(
        goal=goal,
        target_llm=chat,  # Pass the steered chat as the "target"
        max_turns=max_turns,
        temperature=temperature,
    )

    return conv
