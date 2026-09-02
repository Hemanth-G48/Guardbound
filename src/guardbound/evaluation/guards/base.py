"""PromptGuard ABC and shared utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod


class PromptGuard(ABC):
    """Common interface for prompt harmfulness classifiers.

    Every backend must return one of the canonical labels:
        "harmful"  — the prompt is judged harmful
        "harmless" — the prompt is judged harmless
    """

    name: str = "base"

    @abstractmethod
    def predict(self, text: str) -> str:
        """Classify a single prompt.  Returns ``"harmful"`` or ``"harmless"``."""

    def predict_batch(self, texts: list[str]) -> list[str]:
        return [self.predict(t) for t in texts]
