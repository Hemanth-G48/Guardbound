"""Chat-LLM abstraction used by every phase that talks to a target model.

All clients default to ``temperature=0.7`` — the paper-wide generation setting
(Sec. 5.1). Concrete backends import their SDK lazily so this package stays
importable without API keys or optional dependencies installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}

DEFAULT_TEMPERATURE = 0.7  # paper-wide setting

# Evaluator / judge temperature.
#
# Paper (arXiv:2503.00187v3, p9 Sec. 5.1): "All temperatures are set to be 0.7."
# The paper states no exception for judge, rubric, refusal or disclaimer calls,
# so the paper-authoritative value is PAPER_EVALUATOR_TEMPERATURE for all of them.
#
# The author's release instead passes 0.0 to check_refusal / check_disclaimer /
# evaluate_with_rubric. Guardbound inherited that, and every frozen local
# experiment (Phase 14-19) was produced with it. LEGACY_EVALUATOR_TEMPERATURE
# therefore remains the default so historical configurations are unchanged; the
# paper-reproduction configuration opts in to the paper value explicitly.
PAPER_EVALUATOR_TEMPERATURE = 0.7
LEGACY_EVALUATOR_TEMPERATURE = 0.0

# ActorAttack's step judge inherits generate()'s default in the author's release
# (no explicit override), i.e. 0.7 - which already equals the paper value.
ACTOR_STEP_JUDGE_TEMPERATURE = DEFAULT_TEMPERATURE


class ChatLLM(ABC):
    """Minimal chat interface shared by API, local, and mock backends."""

    name: str = "chat-llm"

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: int | None = None,
        json_format: bool = False,
        structured_output_mode: str | None = None,
    ) -> str | dict:
        """Return the assistant reply for a list of chat messages.

        Parameters
        ----------
        messages:
            Chat history as a list of role/content dicts.
        temperature:
            Sampling temperature (paper-wide default: 0.7).
        max_turns_context:
            Optional cap on how many recent turns to include in context.
            ``None`` means use all provided messages.
        json_format:
            If True, parse the response as JSON and return a dict.
            If False (default), return the raw text as a str.
            When True and parsing fails, the raw text is still returned
            so callers can handle the failure themselves.
        structured_output_mode:
            None = default behavior (local JSON stopping criteria).
            "constrained_json" = Phase 14.1 constrained decoding.

        Returns
        -------
        str or dict
            When ``json_format=False``: the raw text response.
            When ``json_format=True``: a parsed dict if the response is
            valid JSON, otherwise the raw text as a str.
        """

    # -- convenience --------------------------------------------------------- #

    def chat(self, user_text: str, system: str | None = None,
             temperature: float = DEFAULT_TEMPERATURE,
             max_turns_context: int | None = None,
             json_format: bool = False,
             structured_output_mode: str | None = None) -> str | dict:
        messages: list[Message] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_text})
        return self.generate(messages, temperature=temperature,
                             max_turns_context=max_turns_context,
                             json_format=json_format,
                             structured_output_mode=structured_output_mode)
