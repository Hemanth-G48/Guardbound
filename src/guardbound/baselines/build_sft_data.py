"""LoRA SFT dataset builder (Phase 8).

Takes Phase 2 training conversations and a *separately verified*
safety-aligned response dataset (Ren et al. 2024 release) and emits
LLaMA-Factory-compatible SFT data in the canonical multi-turn chat
format:

    {
        "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},   # replaced with safe response
            ...
        ],
        "goal_id": "...",
        "source": "..."
    }

This implementation is **schema-only** and **non-fabricating**:

- If the Phase 2 corpus is missing, it raises a clear error.
- If the safety-aligned response file is missing, it raises a clear
  error with the expected path and provenance requirements.
- The harmful / jailbreaking responses from Phase 2 are REPLACED with
  the corresponding safe responses from the verified alignment
  dataset — never with locally-generated text.

Provenance for the alignment data is recorded in the run manifest.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..schemas import Conversation, load_conversations_jsonl

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass
class SFTExample:
    messages: list[dict[str, str]]
    goal_id: str
    source: str


def safe_responses_required_message(path: str | Path) -> str:
    return (
        "Ren et al. 2024 safety-aligned response dataset not found at "
        f"'{path}'.\n\n"
        "Phase 8 requires the RELEASED Ren et al. (2024) rejective "
        "responses, NOT a locally-generated substitute.  Provide the "
        "JSONL at the expected path; each line must include at least:\n"
        "  - 'goal_id'  (or 'behavior' / 'prompt'): matching key to the\n"
        "    Phase 2 training conversation goal\n"
        "  - 'response' (or 'rejective_response' / 'safe_response'):\n"
        "    the canonical safety-aligned response\n\n"
        "See Phase 8 PROVENANCE notes in docs/IMPLEMENTATION_ROADMAP.md "
        "for the full requirement."
    )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _load_safety_responses(path: str | Path) -> dict[str, str]:
    """Load a Ren-et-al-style safety-aligned response file.

    The file is a JSONL with one record per goal.  Each record must
    contain a goal key (``goal_id`` / ``behavior`` / ``prompt``) and
    a response key (``response`` / ``rejective_response`` /
    ``safe_response``).  The function returns a ``{goal_key: safe_response}``
    dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(safe_responses_required_message(p))
    out: dict[str, str] = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            key = (
                d.get("goal_id")
                or d.get("behavior")
                or d.get("prompt")
                or d.get("goal")
            )
            response = (
                d.get("response")
                or d.get("rejective_response")
                or d.get("safe_response")
            )
            if key is None or response is None:
                logger.warning(
                    "Skipping alignment record with missing key/response: %r", d,
                )
                continue
            out[str(key)] = str(response)
    if not out:
        raise ValueError(
            f"No valid safety-aligned response records found in {p}.  "
            f"At least one record with a goal key and a response is required."
        )
    return out


def _goal_key(conv: Conversation) -> str:
    return conv.goal or ""


def build_sft_examples(
    conversations: list[Conversation],
    safety_responses: dict[str, str],
    *,
    system_prompt: str | None = None,
    response_lookup: Callable[[Conversation], str | None] | None = None,
) -> list[SFTExample]:
    """Build LLaMA-Factory-compatible SFT examples from Phase 2 conversations.

    For every conversation, every assistant turn is replaced with the
    corresponding safe response from ``safety_responses``.  If a
    conversation's goal has no matching safe response, it is
    **skipped with a warning** — we do NOT fabricate training labels.
    """
    out: list[SFTExample] = []
    for conv in conversations:
        if response_lookup is not None:
            safe = response_lookup(conv)
        else:
            safe = safety_responses.get(_goal_key(conv))
        if safe is None:
            logger.warning(
                "Skipping conversation with no verified safe response: %r",
                conv.goal[:80],
            )
            continue
        msgs: list[dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for t in conv.turns:
            msgs.append({"role": "user", "content": t.query})
            msgs.append({"role": "assistant", "content": safe})
        out.append(SFTExample(
            messages=msgs,
            goal_id=_goal_key(conv),
            source="phase2+ren2024",
        ))
    return out


def write_sft_jsonl(examples: list[SFTExample], path: str | Path) -> int:
    """Write SFT examples as JSONL (LLaMA-Factory ``messages`` schema)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "messages": ex.messages,
                "goal_id": ex.goal_id,
                "source": ex.source,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Convenience: full pipeline
# --------------------------------------------------------------------------- #

def build_sft_dataset(
    phase2_corpus: str | Path,
    safety_response_file: str | Path,
    out_jsonl: str | Path,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """End-to-end: load Phase 2 corpus + verified safe responses,
    emit LLaMA-Factory-compatible JSONL.

    Returns a small dict with the counts and provenance, suitable
    for inclusion in a run manifest.
    """
    conversations = load_conversations_jsonl(phase2_corpus)
    safe_map = _load_safety_responses(safety_response_file)
    examples = build_sft_examples(
        conversations, safe_map, system_prompt=system_prompt,
    )
    n = write_sft_jsonl(examples, out_jsonl)
    return {
        "source_corpus": str(phase2_corpus),
        "safety_response_file": str(safety_response_file),
        "out_jsonl": str(out_jsonl),
        "n_conversations": len(conversations),
        "n_examples_written": n,
        "n_examples_skipped": len(conversations) - n,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema": "llama-factory/messages",
    }
