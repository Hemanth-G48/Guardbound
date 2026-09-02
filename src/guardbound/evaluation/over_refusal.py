"""Over-refusal evaluation on XSTest / JBB-Benign / PHTest-Harmless.

Metric (paper B.1):

    refusal_rate = refused_queries / total_queries

The refusal detector is the SAME one used by every benchmark (and
every model, in both bare and steered mode) — see
:mod:`guardbound.evaluation.refusal_detector`.

The dataset loaders are intentionally minimal JSONL consumers; they
expect each line to have a ``prompt`` or ``text`` field.  If the
expected dataset file is missing, a clear error is raised with the
expected path and format.

Not specified in the NBF paper:
    - refusal-detection algorithm (see refusal_detector.py)
    - dataset field names
    - exact dataset splits (we treat each file as the full set)
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from ..llm.base import ChatLLM
from ..logging_utils import get_logger
from ..models.predictor import NeuralBarrierFunction
from .refusal_detector import is_refusal
from .reporting import EvaluationResult, write_jsonl

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

@dataclass
class OverRefusalExample:
    prompt: str
    label: str | None = None   # raw benchmark label, if present
    extra: dict[str, Any] = field(default_factory=dict)


def _load_jsonl_field(path: str | Path, fields: tuple[str, ...] = ("prompt", "text", "query")) -> list[OverRefusalExample]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Over-refusal dataset not found at {p}.  Provide a JSONL where "
            f"each line has one of these text fields: {fields}.  See "
            f"evaluation/over_refusal.py for the schema."
        )
    out: list[OverRefusalExample] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = None
            for fld in fields:
                if fld in d and isinstance(d[fld], str):
                    text = d[fld]
                    break
            if text is None:
                # Last resort: stringify the dict
                text = json.dumps(d, ensure_ascii=False)
            label = d.get("label") or d.get("type") or d.get("category")
            extra = {k: v for k, v in d.items() if k not in (fields + ("label", "type", "category"))}
            out.append(OverRefusalExample(prompt=text, label=label, extra=extra))
    return out


def load_xstest(path: str | Path) -> list[OverRefusalExample]:
    """Load XSTest prompts.

    XSTest fields: ``prompt``, ``type`` (contrast / safe).
    """
    return _load_jsonl_field(path, fields=("prompt", "text"))


def load_jbb_benign(path: str | Path) -> list[OverRefusalExample]:
    """Load JailbreakBench-Benign behaviors.

    Fields: ``prompt`` or ``behavior`` or ``text``.
    """
    return _load_jsonl_field(path, fields=("prompt", "behavior", "text"))


def load_phtest_harmless(path: str | Path) -> list[OverRefusalExample]:
    """Load PHTest-Harmless prompts.

    Fields: ``prompt`` or ``text``.
    """
    return _load_jsonl_field(path, fields=("prompt", "text"))


LOADERS: dict[str, Callable[[str | Path], list[OverRefusalExample]]] = {
    "xstest": load_xstest,
    "jbb_benign": load_jbb_benign,
    "phtest_harmless": load_phtest_harmless,
}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class OverRefusalResult:
    total: int
    refused: int
    refusal_rate: float
    dataset: str
    model: str
    defense: str
    eta: float | None
    per_example: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_name(llm: ChatLLM) -> str:
    return (
        getattr(llm, "model_id", None)
        or getattr(llm, "name", "model")
    )


def evaluate_over_refusal(
    examples: Iterable[OverRefusalExample],
    llm: ChatLLM,
    *,
    dataset_name: str = "unknown",
    defense: str | None = None,
    eta: float | None = None,
    out_path: str | Path | None = None,
) -> OverRefusalResult:
    """Run each prompt through the LLM and apply the shared refusal detector.

    Bare vs steered: this function does not apply the NBF Q-filter; it
    only checks the LLM's *response* with the refusal detector.  For
    the paper's over-refusal experiment, the steered runs use the
    standard SteeredLLMChat (which may pre-block some queries) — the
    caller passes the post-defense response to this function.  When
    using bare mode, the caller passes the bare LLM directly.
    """
    examples = list(examples)
    per: list[dict[str, Any]] = []
    refused = 0
    for ex in examples:
        try:
            response = llm.generate(
                [{"role": "user", "content": ex.prompt}],
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Over-refusal LLM call failed: %s", exc)
            response = ""
        flag = is_refusal(response)
        if flag:
            refused += 1
        per.append({
            "prompt": ex.prompt[:120],
            "label": ex.label,
            "refused": flag,
            "response_snippet": (response or "")[:120],
        })

    total = len(examples)
    rate = (refused / total) if total else 0.0
    result = OverRefusalResult(
        total=total,
        refused=refused,
        refusal_rate=rate,
        dataset=dataset_name,
        model=_model_name(llm),
        defense=defense or "original",
        eta=eta,
        per_example=per,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = out_path.with_suffix(".results.jsonl")
        write_jsonl([EvaluationResult(
            model=result.model,
            defense_variant=result.defense,
            metric="over_refusal_rate",
            value=result.refusal_rate,
            dataset=result.dataset,
            eta=result.eta,
            n=result.total,
            extras={"refused": result.refused},
        )], sidecar)
    return result
