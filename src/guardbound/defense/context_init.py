"""Context initialization for MMLU-style pre-question state setup.

Initializes the dialogue state using system prompts or in-context examples
WITHOUT applying the Q-filter. Each context text is processed through
f_theta to advance the state:

    x_0 = zeros(768)
    for each context_text:
        u_i = embedding(context_text)
        x_i = f_theta(x_{i-1}, u_i)

No LLM calls are made. No barrier filter is applied.
"""
from __future__ import annotations

from typing import Callable

import torch

from ..models.predictor import NeuralBarrierFunction
from ..logging_utils import get_logger

logger = get_logger(__name__)


def initialize_with_context(
    barrier: NeuralBarrierFunction,
    state: torch.Tensor,
    context_texts: list[str],
    embed_fn: Callable[[str], torch.Tensor],
) -> torch.Tensor:
    """Initialize state by processing context texts through f_theta.

    Parameters
    ----------
    barrier : NeuralBarrierFunction
        The NBF bundle containing the dynamics model.
    state : torch.Tensor
        Initial state [1, state_dim] (typically zeros).
    context_texts : list[str]
        Context texts to process (e.g., system prompts, in-context examples).
    embed_fn : callable
        Function mapping text -> tensor [1, embedding_dim].

    Returns
    -------
    new_state : torch.Tensor
        The state after processing all context texts [1, state_dim].
    """
    current_state = state.clone()

    for i, text in enumerate(context_texts):
        u = embed_fn(text)  # [1, emb_dim]
        u = u.to(current_state.device)

        with torch.no_grad():
            current_state = barrier.advance_state(current_state, u)

        logger.debug(
            "Context init step %d/%d: processed '%s', state norm=%.4f",
            i + 1, len(context_texts), text[:50],
            torch.norm(current_state).item(),
        )

    return current_state
