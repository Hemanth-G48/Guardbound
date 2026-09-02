"""Common defense wrapper interface.

The baseline registry exposes each variant as a ``DefenseWrapper``.
``build_defense(variant, llm, ...)`` returns a ``ChatLLM``-compatible
object.  Downstream Phase 6 attacks and Phase 7 metrics are unaware
of which defense is active.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..llm.base import ChatLLM


class DefenseWrapper(ABC):
    """Abstract base for a defense variant.

    A defense wrapper is *not* a model — it transforms a target LLM
    into a defended LLM.  Concrete subclasses for ``original``,
    ``system_prompt``, ``lora_*``, and ``guardbound`` live in their
    own modules.
    """

    #: Stable name used by the registry and as a results tag.
    name: str = "base"

    @abstractmethod
    def wrap(self, llm: ChatLLM) -> ChatLLM:
        """Return a ``ChatLLM``-compatible defended target.

        The returned object MUST satisfy the ``ChatLLM`` interface
        (have a ``generate(messages, temperature=0.7, ...) -> str``
        method).  It MAY be a wrapper class, a different concrete
        backend, or the same object (for ``original``).
        """

    @property
    def results_tag(self) -> str:
        """Short tag used in result rows / file names."""
        return self.name
