"""``system_prompt`` baseline: prepend the Llama-2 safety system prompt.

The wrapper is a transparent ``ChatLLM`` decorator that prepends a
single system message to every outgoing request.  It:

- preserves the user's original query
- preserves the existing conversation history (user/assistant pairs)
- does NOT mutate the caller-owned ``messages`` list (it is copied)
- does NOT add the system prompt twice (idempotent: if the leading
  message already has the same content, no second copy is added)
- is compatible with any ``ChatLLM`` (including ``MockChatLLM``)
- is deterministic

The default system prompt text lives at
``src/guardbound/baselines/prompts/llama2_safety_system.txt`` with
provenance metadata in the file's trailing comment.  The content is
the canonical Llama-2-Chat default safety system prompt as released
by Meta / used in the HF chat template; see the file header for the
verification status.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..llm.base import DEFAULT_TEMPERATURE, ChatLLM
from .base import DefenseWrapper

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "llama2_safety_system.txt"


def _strip_provenance(text: str) -> str:
    """Remove the trailing provenance comment block from a prompt file.

    The shipped prompt file contains a ``---`` block with metadata
    that the LLM must never see.  This helper returns only the
    leading prompt body.
    """
    sep = "\n---\n"
    if sep in text:
        return text.split(sep, 1)[0].rstrip() + "\n"
    return text


class SystemPromptChatLLM(ChatLLM):
    """ChatLLM wrapper that prepends a safety system prompt.

    The wrapped LLM is the *real* target; this class only intercepts
    the ``messages`` list and delegates to ``target_llm.generate``.
    """

    name = "system-prompt"

    def __init__(self, target_llm: ChatLLM, system_prompt: str):
        self.target_llm = target_llm
        self.system_prompt = system_prompt

    def generate(self, messages, temperature=DEFAULT_TEMPERATURE,
                 max_turns_context: int | None = None) -> str:
        # CRITICAL: do not mutate the caller's list.  Copy first.
        msgs = [dict(m) for m in messages]
        # Idempotency: do not add the system prompt twice.
        if not msgs or msgs[0].get("role") != "system" \
                or msgs[0].get("content") != self.system_prompt:
            msgs.insert(0, {"role": "system", "content": self.system_prompt})
        return self.target_llm.generate(
            msgs, temperature=temperature, max_turns_context=max_turns_context,
        )


class SystemPromptDefense(DefenseWrapper):
    """Defense wrapper for the ``system_prompt`` variant."""

    name = "system_prompt"

    def __init__(self, prompt_text: str | None = None,
                 prompt_path: str | Path | None = None):
        if prompt_text is not None:
            self._prompt = prompt_text
        else:
            path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
            if not path.exists():
                raise FileNotFoundError(
                    f"Llama-2 safety system prompt not found at {path}.  "
                    f"Provide --system-prompt or place the file at the "
                    f"default location."
                )
            self._prompt = _strip_provenance(path.read_text(encoding="utf-8"))
        self._prompt_path = str(Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH)

    def wrap(self, llm: ChatLLM) -> ChatLLM:
        return SystemPromptChatLLM(target_llm=llm, system_prompt=self._prompt)

    @property
    def prompt_path(self) -> str:
        return self._prompt_path
