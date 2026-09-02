"""OpenAI Moderation API guard.

Uses the ``omni-moderation-latest`` model by default.  Returns
``"harmful"`` if any category is flagged, otherwise ``"harmless"``.

If the openai SDK or ``OPENAI_API_KEY`` is not available, ``__init__``
succeeds but the first :meth:`predict` call surfaces a clear error.
This is intentional: tests use mocks, real evaluations must have
credentials, and the call must NEVER silently substitute another
guard.
"""
from __future__ import annotations

from .base import PromptGuard

# OpenAI's omni-moderation returns categories with values "true"/"false".
_HARMFUL_CATEGORIES = (
    "harassment", "harassment/threatening",
    "hate", "hate/threatening",
    "illicit", "illicit/violent",
    "self-harm", "self-harm/intent", "self-harm/instructions",
    "sexual", "sexual/minors",
    "violence", "violence/graphic",
    "jailbreak",
)


class OpenAIModerationGuard(PromptGuard):
    name = "openai_moderation"

    def __init__(self, model: str = "omni-moderation-latest", api_key_env: str = "OPENAI_API_KEY"):
        self.model = model
        self.api_key_env = api_key_env
        self._client = None  # lazy

    def _get_client(self):
        if self._client is None:
            try:
                import os
                import openai
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "OpenAI Moderation guard requires the openai package. "
                    "Install with: pip install openai"
                ) from exc
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"OpenAI Moderation guard requires ${self.api_key_env}. "
                    f"Set the environment variable before evaluation."
                )
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def predict(self, text: str) -> str:
        client = self._get_client()
        resp = client.moderations.create(model=self.model, input=text)
        if not resp.results:
            return "harmless"
        result = resp.results[0]
        categories = getattr(result, "categories", None)
        if categories is None:
            return "harmless"
        # categories is an object with attributes per OpenAI SDK shape
        for cat in _HARMFUL_CATEGORIES:
            if getattr(categories, cat.replace("/", "_").replace("-", "_"), False):
                return "harmful"
        return "harmless"
