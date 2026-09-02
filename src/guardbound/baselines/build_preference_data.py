"""Preference-data builders for LoRA DPO and LoRA KTO (Phase 8).

Constructs preference records from the Phase 8 SFT alignment data
(matching goals -> verified safe response from Ren et al. 2024).

DPO schema (LLaMA-Factory):

    {
        "prompt":   [{ "role": "user", "content": "..." }, ...],
        "chosen":   [{ "role": "assistant", "content": "..." }],
        "rejected": [{ "role": "assistant", "content": "..." }]
    }

KTO schema (LLaMA-Factory):

    {
        "prompt":          [{ "role": "user", "content": "..." }, ...],
        "completion":      [{ "role": "assistant", "content": "..." }],
        "label":           true | false       # True = desirable
    }

For rejected responses we use the *original* harmful response from
the Phase 2 corpus.  The paper does not specify the rejected-source
construction rule; using the original jailbreak response is the most
direct and reproducible choice, documented as:

    "Not specified in the NBF paper — locally constructed from
     the Phase 2 corpus: 'rejected' = the original jailbreak response."

This module is **non-fabricating** — it never invents either the
chosen or the rejected text.  Both come from verified sources:
chosen from the Ren et al. safety response file, rejected from the
Phase 2 conversation corpus.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_utils import get_logger
from ..schemas import Conversation, load_conversations_jsonl
from .build_sft_data import _load_safety_responses

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

@dataclass
class DPOExample:
    prompt: list[dict[str, str]]
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]


@dataclass
class KTOExample:
    prompt: list[dict[str, str]]
    completion: list[dict[str, str]]
    label: bool


# --------------------------------------------------------------------------- #
# DPO
# --------------------------------------------------------------------------- #

def _conv_to_prompt_messages(
    conv: Conversation, system_prompt: str | None
) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for t in conv.turns:
        msgs.append({"role": "user", "content": t.query})
    return msgs


def build_dpo_examples(
    conversations: list[Conversation],
    safety_responses: dict[str, str],
    *,
    system_prompt: str | None = None,
) -> list[DPOExample]:
    """Build DPO pairs.

    ``chosen``   = the verified safe response from Ren et al. 2024.
    ``rejected`` = the original Phase 2 jailbreak response.

    Conversations without a matching safe response are skipped.
    """
    out: list[DPOExample] = []
    for conv in conversations:
        safe = safety_responses.get(conv.goal or "")
        if safe is None:
            logger.warning("DPO: skipping conversation without safe response: %r",
                           (conv.goal or "")[:80])
            continue
        # Find the first existing assistant turn to use as the rejected
        # response.  If none, skip with a warning.
        rejected = next(
            (t.response for t in conv.turns if t.response), None,
        )
        if rejected is None:
            logger.warning("DPO: no assistant turn in conversation: %r",
                           (conv.goal or "")[:80])
            continue
        out.append(DPOExample(
            prompt=_conv_to_prompt_messages(conv, system_prompt),
            chosen=[{"role": "assistant", "content": safe}],
            rejected=[{"role": "assistant", "content": rejected}],
        ))
    return out


def write_dpo_jsonl(examples: list[DPOExample], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "prompt": ex.prompt,
                "chosen": ex.chosen,
                "rejected": ex.rejected,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# KTO
# --------------------------------------------------------------------------- #

def build_kto_examples(
    conversations: list[Conversation],
    safety_responses: dict[str, str],
    *,
    system_prompt: str | None = None,
) -> list[KTOExample]:
    """Build KTO records.

    Each Phase 2 conversation produces TWO KTO records:
      - desirable (label=True)  with the verified safe response.
      - undesirable (label=False) with the original jailbreak response.
    """
    out: list[KTOExample] = []
    for conv in conversations:
        safe = safety_responses.get(conv.goal or "")
        rejected = next(
            (t.response for t in conv.turns if t.response), None,
        )
        prompt = _conv_to_prompt_messages(conv, system_prompt)
        if safe is not None:
            out.append(KTOExample(
                prompt=prompt,
                completion=[{"role": "assistant", "content": safe}],
                label=True,
            ))
        if rejected is not None:
            out.append(KTOExample(
                prompt=prompt,
                completion=[{"role": "assistant", "content": rejected}],
                label=False,
            ))
    return out


def write_kto_jsonl(examples: list[KTOExample], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({
                "prompt": ex.prompt,
                "completion": ex.completion,
                "label": ex.label,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #

def build_preference_datasets(
    phase2_corpus: str | Path,
    safety_response_file: str | Path,
    out_dir: str | Path,
    *,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conversations = load_conversations_jsonl(phase2_corpus)
    safe_map = _load_safety_responses(safety_response_file)

    dpo = build_dpo_examples(conversations, safe_map, system_prompt=system_prompt)
    kto = build_kto_examples(conversations, safe_map, system_prompt=system_prompt)
    dpo_path = out_dir / "dpo.jsonl"
    kto_path = out_dir / "kto.jsonl"
    n_dpo = write_dpo_jsonl(dpo, dpo_path)
    n_kto = write_kto_jsonl(kto, kto_path)
    return {
        "source_corpus": str(phase2_corpus),
        "safety_response_file": str(safety_response_file),
        "n_conversations": len(conversations),
        "n_dpo_examples": n_dpo,
        "n_kto_examples": n_kto,
        "dpo_path": str(dpo_path),
        "kto_path": str(kto_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema": "llama-factory/dpo+kto",
        "construction_note": (
            "Not specified in the NBF paper — locally constructed from "
            "the Phase 2 corpus: 'rejected' = the original jailbreak "
            "response; 'chosen' = the verified Ren et al. (2024) "
            "safety-aligned response."
        ),
    }
