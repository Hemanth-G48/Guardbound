"""Core conversation data model shared by every phase.

Turn/Conversation mirror the paper's notation:

    U_k — user query at turn k            -> Turn.query
    Z_k — LLM response at turn k          -> Turn.response
    y_k — GPT-4o judge score in {1..5}    -> Turn.judge_score   (unsafe <=> 5)

Query/response embeddings (u_k, z_k in R^768) are attached in Phase 2;
``was_filtered`` is set by the NBF defense runtime in Phase 5.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np


@dataclass
class Turn:
    """One dialogue turn k (query U_k + response Z_k + optional labels)."""

    query: str                                   # U_k
    response: str | None = None                  # Z_k
    query_embedding: np.ndarray | None = None    # u_k ∈ R^768 (Phase 2)
    response_embedding: np.ndarray | None = None # z_k ∈ R^768 (Phase 2)
    judge_score: int | None = None               # y_k ∈ {1..5}; unsafe ⇔ 5 (Phase 2)
    was_filtered: bool = False                   # blocked by NBF Q-filter (Phase 5)

    @staticmethod
    def _eq(a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return (
                isinstance(a, np.ndarray)
                and isinstance(b, np.ndarray)
                and a.shape == b.shape
                and bool(np.allclose(a, b))
            )
        return a == b

    def __eq__(self, other: object) -> bool:
        """Array-aware equality (numpy arrays compare via allclose)."""
        if not isinstance(other, Turn):
            return NotImplemented
        return all(
            self._eq(getattr(self, f), getattr(other, f))
            for f in ("query", "response", "query_embedding",
                      "response_embedding", "judge_score", "was_filtered")
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("query_embedding", "response_embedding"):
            arr = getattr(self, key)
            d[key] = arr.tolist() if arr is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        kwargs = dict(d)
        for key in ("query_embedding", "response_embedding"):
            val = kwargs.get(key)
            kwargs[key] = np.asarray(val, dtype=np.float32) if val is not None else None
        return cls(**kwargs)


@dataclass
class Conversation:
    """A full multi-turn dialogue trajectory."""

    goal: str                                    # attack goal / harmful behavior
    attack_method: str                           # actorattack|crescendo|opposite_day|acronym|redqueen|...
    target_llm: str                              # model identifier
    turns: list[Turn] = field(default_factory=list)
    max_turns: int = 8                           # paper B.1: K_max = 8

    # -- convenience -------------------------------------------------------- #

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def conversation_id(self) -> str:
        """Stable identifier derived from conversation key attributes.

        Uses a hash of (goal, attack_method, target_llm, max_turns) to provide
        a stable ID for resume/caching without storing additional state.
        """
        payload = json.dumps({
            "goal": self.goal,
            "attack_method": self.attack_method,
            "target_llm": self.target_llm,
            "max_turns": self.max_turns,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def queries(self) -> Iterator[str]:
        for t in self.turns:
            yield t.query

    # -- serialization -------------------------------------------------------#

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "attack_method": self.attack_method,
            "target_llm": self.target_llm,
            "max_turns": self.max_turns,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        return cls(
            goal=d["goal"],
            attack_method=d["attack_method"],
            target_llm=d["target_llm"],
            max_turns=int(d.get("max_turns", 8)),
            turns=[Turn.from_dict(t) for t in d.get("turns", [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "Conversation":
        return cls.from_dict(json.loads(line))


def save_conversations_jsonl(conversations: list[Conversation], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv.to_json() + "\n")


def load_conversations_jsonl(path: str | Path) -> list[Conversation]:
    conversations: list[Conversation] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(Conversation.from_json(line))
    return conversations
