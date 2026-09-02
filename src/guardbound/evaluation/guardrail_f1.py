"""Guardrail F1 evaluation.

Compares prompt harmfulness classifiers against ground-truth labels on:

    - HarmBench
    - AegisSafetyTest
    - WildGuardTest

Binary classification rule for the NBF guard (paper):

    argmax predictor class == 1  -> harmless
    otherwise                    -> harmful

This is the ARGMAX CLASS rule — NOT the runtime Q-filter rule
``h + eta >= 0``.  The two are different evaluation criteria (see
the docs/IMPLEMENTATION_ROADMAP.md for the metric distinctions table).

The single-prompt initial state is ``x_0 = zeros(768)``.  This is
a local default: the paper does not specify the state-init rule for
the single-prompt guardrail benchmark.

Dataset field mappings:
    - HarmBench:   "label" ∈ {0, 1} (1 = harmful)
    - AegisSafetyTest: "prompt_safety" / "response_safety" labels,
                       "safe" or "unsafe"
    - WildGuardTest:    "label" or "prompt_harm_label" ∈ {0, 1}
                       (1 = harmful)
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from ..embeddings import resolve_embedding_model
from ..llm.base import ChatLLM
from ..logging_utils import get_logger
from ..models.predictor import NeuralBarrierFunction, SafetyPredictor
from ..schemas import Conversation
from .cache import JudgeCache, make_cache_key
from .guards.base import PromptGuard
from .reporting import EvaluationResult, write_jsonl

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

@dataclass
class GuardExample:
    text: str
    label: str   # "harmful" or "harmless"


def _norm_label(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"1", "true", "harmful", "unsafe", "yes"}:
        return "harmful"
    if s in {"0", "false", "harmless", "safe", "no"}:
        return "harmless"
    return None


def _load_harmbench(path: str | Path) -> list[GuardExample]:
    """HarmBench: each line has ``text`` (the prompt) and ``label`` (0/1)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"HarmBench guardrail dataset not found at {p}.  Each line must "
            f"have a ``text`` field and a ``label`` field (0=harmless, "
            f"1=harmful)."
        )
    out: list[GuardExample] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("prompt") or d.get("behavior") or ""
            label = _norm_label(d.get("label", d.get("is_harmful")))
            if not text or label is None:
                continue
            out.append(GuardExample(text=text, label=label))
    return out


def _load_aegis(path: str | Path) -> list[GuardExample]:
    """AegisSafetyTest: each line has ``text`` and ``prompt_safety`` ∈
    {safe, unsafe}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"AegisSafetyTest guardrail dataset not found at {p}.  Each line "
            f"must have a ``text`` field and a ``prompt_safety`` field "
            f"(safe=harmless, unsafe=harmful)."
        )
    out: list[GuardExample] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("prompt") or ""
            label = _norm_label(d.get("prompt_safety", d.get("label")))
            if not text or label is None:
                continue
            out.append(GuardExample(text=text, label=label))
    return out


def _load_wildguard(path: str | Path) -> list[GuardExample]:
    """WildGuardTest: each line has ``text`` and ``prompt_harm_label`` (0/1)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"WildGuardTest guardrail dataset not found at {p}.  Each line "
            f"must have a ``text`` field and a ``prompt_harm_label`` field "
            f"(0=harmless, 1=harmful)."
        )
    out: list[GuardExample] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("prompt") or ""
            label = _norm_label(d.get("prompt_harm_label", d.get("label")))
            if not text or label is None:
                continue
            out.append(GuardExample(text=text, label=label))
    return out


LOADERS: dict[str, Callable[[str | Path], list[GuardExample]]] = {
    "harmbench": _load_harmbench,
    "aegis": _load_aegis,
    "wildguard": _load_wildguard,
}


# --------------------------------------------------------------------------- #
# NBF guard
# --------------------------------------------------------------------------- #

class NBFPromptGuard(PromptGuard):
    """Single-prompt NBF guard.

    Classification rule (paper):

        argmax predictor class == 1  -> harmless
        otherwise                    -> harmful

    Internal class indexing is 0-based ({0,1,2,3,4}); paper labels
    are 1-based ({1,2,3,4,5}).  ``predicted_class == 1`` (paper
    convention) corresponds to internal index 0 — see
    ``SafetyPredictor.PAPER_LABEL_TO_CE``.

    State initialization for single-prompt guardrail F1:

        x_0 = zeros(768)        (local default; not specified in paper)

    Embedding model: must match the embedding used to train the
    predictor.  Pass ``embedding_alias="mpnet"`` or
    ``"distilroberta"``.
    """

    name = "nbf_guard"

    def __init__(
        self,
        barrier: NeuralBarrierFunction,
        embed_fn: Callable[[str], torch.Tensor],
        *,
        cache: JudgeCache | None = None,
    ):
        self.barrier = barrier
        self.embed_fn = embed_fn
        self.cache = cache or JudgeCache("data/processed/.cache/nbf_guard")
        self.state_dim = barrier.predictor.state_dim

    def predict(self, text: str) -> str:
        parts = {
            "protocol": "nbf_guard_argmax_v1",
            "embedding_model": getattr(self.barrier, "embedding_model", "unknown"),
            "text": text,
        }
        key = make_cache_key(parts)
        cached = self.cache.get(key)
        if cached is not None:
            return (cached.get("result") or {}).get("label", "harmless")

        with torch.no_grad():
            x0 = torch.zeros(1, self.state_dim,
                             device=next(self.barrier.parameters()).device)
            u = self.embed_fn(text)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            u = u.to(x0.device)
            p = self.barrier.predictor.class_probs(x0, u)
        # Argmax over the 0-based class index.
        ce_index = int(p.argmax(dim=-1).detach().cpu().reshape(-1)[0].item())
        # Convert to paper convention (1..5).  Paper "class 1" == internal 0.
        paper_label = ce_index + 1
        label = "harmless" if paper_label == 1 else "harmful"
        self.cache.set(key, {
            "result": {"label": label, "paper_class": paper_label,
                       "ce_index": ce_index},
            "cache_key": key,
            **parts,
        })
        return label


def make_nbf_guard_from_checkpoint(
    dynamics_dir: str | Path,
    predictor_path: str | Path,
    embed_fn: Callable[[str], torch.Tensor],
) -> NBFPromptGuard:
    """Convenience builder: load a checkpoint and wrap it as a guard."""
    barrier = NeuralBarrierFunction.load(
        dynamics_dir=dynamics_dir, predictor_path=predictor_path,
    )
    return NBFPromptGuard(barrier=barrier, embed_fn=embed_fn)


# --------------------------------------------------------------------------- #
# Binary metrics
# --------------------------------------------------------------------------- #

@dataclass
class BinaryMetrics:
    tp: int
    tn: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    n: int
    support_harmful: int
    support_harmless: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_binary_metrics(
    y_true: list[str], y_pred: list[str], *, positive: str = "harmful"
) -> BinaryMetrics:
    """Compute precision / recall / F1 / accuracy for binary classification.

    ``positive`` is the class treated as the positive label for
    precision/recall (default ``"harmful"``).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp = tn = fp = fn = 0
    support_pos = support_neg = 0
    for t, p in zip(y_true, y_pred):
        if t == positive:
            support_pos += 1
        else:
            support_neg += 1
        if t == positive and p == positive:
            tp += 1
        elif t != positive and p != positive:
            tn += 1
        elif t != positive and p == positive:
            fp += 1
        else:
            fn += 1
    n = len(y_true)
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / n) if n else 0.0
    return BinaryMetrics(
        tp=tp, tn=tn, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1, accuracy=accuracy,
        n=n, support_harmful=support_pos, support_harmless=support_neg,
    )


def compute_f1(
    y_true: list[str], y_pred: list[str], *, positive: str = "harmful"
) -> float:
    """Convenience wrapper returning just the F1 score."""
    return compute_binary_metrics(y_true, y_pred, positive=positive).f1


# --------------------------------------------------------------------------- #
# Top-level evaluation
# --------------------------------------------------------------------------- #

@dataclass
class GuardResult:
    metrics: BinaryMetrics
    guard_name: str
    dataset: str
    embedding_model: str | None = None
    nbf_params: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self.metrics)
        d["guard_name"] = self.guard_name
        d["dataset"] = self.dataset
        d["embedding_model"] = self.embedding_model
        d["nbf_params"] = self.nbf_params
        return d


def evaluate_guard(
    examples: Iterable[GuardExample],
    guard: PromptGuard,
    *,
    dataset_name: str = "unknown",
    out_path: str | Path | None = None,
) -> GuardResult:
    """Score ``guard`` on ``examples`` and write EvaluationResult rows."""
    examples = list(examples)
    y_true = [e.label for e in examples]
    y_pred: list[str] = []
    for e in examples:
        try:
            y_pred.append(guard.predict(e.text))
        except Exception as exc:
            logger.warning("Guard %s failed: %s", guard.name, exc)
            y_pred.append("harmless")
    metrics = compute_binary_metrics(y_true, y_pred)
    nbf_params = None
    embedding_model = None
    if isinstance(guard, NBFPromptGuard):
        nbf_params = sum(p.numel() for p in guard.barrier.parameters())
        embedding_model = getattr(guard.barrier, "embedding_model", None)
    result = GuardResult(
        metrics=metrics,
        guard_name=guard.name,
        dataset=dataset_name,
        embedding_model=embedding_model,
        nbf_params=nbf_params,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = out_path.with_suffix(".results.jsonl")
        write_jsonl([
            EvaluationResult(
                model=guard.name, defense_variant="guardrail",
                metric="F1", value=metrics.f1,
                dataset=dataset_name,
                embedding_model=embedding_model,
                n=metrics.n,
                extras=metrics.to_dict(),
            ),
            EvaluationResult(
                model=guard.name, defense_variant="guardrail",
                metric="precision", value=metrics.precision,
                dataset=dataset_name,
                n=metrics.n,
            ),
            EvaluationResult(
                model=guard.name, defense_variant="guardrail",
                metric="recall", value=metrics.recall,
                dataset=dataset_name,
                n=metrics.n,
            ),
            EvaluationResult(
                model=guard.name, defense_variant="guardrail",
                metric="accuracy", value=metrics.accuracy,
                dataset=dataset_name,
                n=metrics.n,
            ),
        ], sidecar)
    return result
