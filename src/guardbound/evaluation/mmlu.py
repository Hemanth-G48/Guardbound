"""MMLU helpfulness evaluation.

Bare mode: ask the question to the LLM, parse the answer letter, mark correct.

Steered mode: pre-roll the NBF dialogue state through any system/in-context
prompts via Phase 5's :func:`initialize_with_context`, then evaluate the
question.  After the LLM answers, the barrier value ``h`` is computed for
the question embedding; the result is post-filtered using Phase 5's
``mmlu_filter_answer`` (h > 0 -> force WRONG).  This is paper B.1's rule.

The dataset loader is intentionally minimal: it expects a JSONL file with
fields ``question``, ``choices`` (list of 4 strings), and ``answer`` (0-3
index, the correct choice).  A real MMLU loader from HuggingFace
``hendrycks_test`` or similar is *not* included to avoid pinning
downloads.  Synthetic fixtures of the same shape are used in unit tests.

Not specified in the NBF paper:
    - exact MMLU dataset loader (we accept any JSONL with the schema above)
    - exact LLM prompt wording to elicit a multiple-choice letter
    - default answer parser (we extract the first standalone A-D token)
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from ..defense.context_init import initialize_with_context
from ..defense.post_processing import mmlu_filter_answer
from ..llm.base import ChatLLM
from ..logging_utils import get_logger
from ..models.predictor import NeuralBarrierFunction
from .reporting import EvaluationResult, write_jsonl

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #

@dataclass
class MMLUQuestion:
    question: str
    choices: list[str]            # 4 strings (A, B, C, D)
    answer: int                   # 0-3
    context: list[str] | None = None  # optional in-context prompts

    @property
    def correct_letter(self) -> str:
        return "ABCD"[self.answer]


def load_mmlu_jsonl(path: str | Path) -> list[MMLUQuestion]:
    """Load MMLU-shaped JSONL.  Each line: {question, choices, answer, context?}."""
    import json
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"MMLU dataset not found at {p}.  Provide a JSONL with fields "
            f"{{question, choices, answer}} (see evaluation/mmlu.py for schema)."
        )
    out: list[MMLUQuestion] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(MMLUQuestion(
                question=d["question"],
                choices=list(d["choices"]),
                answer=int(d["answer"]),
                context=d.get("context"),
            ))
    return out


# --------------------------------------------------------------------------- #
# Prompt + answer parsing
# --------------------------------------------------------------------------- #

MMLU_QUESTION_PROMPT = (
    "Answer the following multiple-choice question. Respond with a single "
    "letter (A, B, C, or D) on the first line, then optionally explain.\n\n"
    "Question: {question}\n"
    "{choices}\n"
)


def _format_choices(choices: list[str]) -> str:
    return "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))


_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _parse_answer_letter(text: str) -> str | None:
    if not text:
        return None
    # The first standalone A-D token on the first non-empty line wins.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LETTER_RE.search(line)
        if m:
            return m.group(1).upper()
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class MMLUResult:
    total: int
    correct: int
    incorrect: int
    accuracy: float
    model: str
    defense: str
    eta: float | None
    embedding_model: str | None
    details: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _model_name(llm: ChatLLM) -> str:
    return (
        getattr(llm, "model_id", None)
        or getattr(llm, "name", "model")
    )


def evaluate_mmlu(
    questions: Iterable[MMLUQuestion],
    llm: ChatLLM,
    *,
    barrier: NeuralBarrierFunction | None = None,
    embed_fn: Callable[[str], torch.Tensor] | None = None,
    eta: float | None = None,
    defense: str | None = None,
    defense_eta: float | None = None,
    out_path: str | Path | None = None,
) -> MMLUResult:
    """Evaluate an MMLU-shaped dataset.

    Parameters
    ----------
    questions
        Iterable of MMLUQuestion.  For steered mode, questions with a
        non-empty ``context`` field are pre-rolled through the NBF
        dynamics before answering.
    llm
        Target LLM (bare or steered wrapper).
    barrier
        NeuralBarrierFunction (only used for steered mode).
    embed_fn
        Function ``text -> Tensor[1, 768]`` (only used for steered mode).
    eta
        Steering threshold for the post-filtering step (h > 0 forces
        wrong regardless of eta; eta is recorded for reporting).
    defense
        Human-readable name of the defense variant in use (e.g.
        "guardbound" or "original").  If None, the function
        auto-detects: steered iff barrier+embed_fn are provided.
    defense_eta
        The eta actually used for the Q-filter (for reporting).
    out_path
        If provided, write a per-question JSONL plus a one-row
        EvaluationResult sidecar.
    """
    questions = list(questions)
    use_steered = barrier is not None and embed_fn is not None
    defense = defense or ("guardbound" if use_steered else "original")
    defense_eta = eta if use_steered else None

    details: list[dict[str, Any]] = []
    correct = 0
    for q in questions:
        # 1. Reset / initialize state
        if use_steered:
            assert barrier is not None and embed_fn is not None
            state = torch.zeros(1, barrier.predictor.state_dim,
                                device=next(barrier.parameters()).device)
            if q.context:
                state = initialize_with_context(
                    barrier, state, q.context, embed_fn
                )
        else:
            state = None

        # 2. Build prompt and get LLM answer
        prompt = MMLU_QUESTION_PROMPT.format(
            question=q.question,
            choices=_format_choices(q.choices),
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            response = llm.generate(messages, temperature=0.0)
        except Exception as exc:
            logger.warning("MMLU LLM call failed: %s", exc)
            response = ""
        original_correct = _parse_answer_letter(response) == q.correct_letter

        # 3. Steered post-filter
        h_value: float | None = None
        if use_steered:
            assert state is not None and embed_fn is not None
            with torch.no_grad():
                u = embed_fn(prompt).to(state.device)
                h_value = float(barrier.h(state, u).detach().cpu().reshape(-1)[0].item())
            final_correct = mmlu_filter_answer(h_value, original_correct)
        else:
            final_correct = original_correct

        if final_correct:
            correct += 1
        details.append({
            "question": q.question,
            "answer_letter": _parse_answer_letter(response),
            "correct_letter": q.correct_letter,
            "original_correct": original_correct,
            "h_value": h_value,
            "final_correct": final_correct,
            "response_snippet": (response or "")[:120],
        })

    total = len(questions)
    accuracy = (correct / total) if total else 0.0
    result = MMLUResult(
        total=total,
        correct=correct,
        incorrect=total - correct,
        accuracy=accuracy,
        model=_model_name(llm),
        defense=defense,
        eta=defense_eta,
        embedding_model=(getattr(barrier, "embedding_model", None)
                         if barrier is not None else None),
        details=details,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = out_path.with_suffix(".results.jsonl")
        write_jsonl([EvaluationResult(
            model=result.model,
            defense_variant=result.defense,
            metric="MMLU",
            value=result.accuracy,
            dataset="MMLU",
            eta=result.eta,
            embedding_model=result.embedding_model,
            n=result.total,
            extras={"correct": result.correct, "incorrect": result.incorrect},
        )], sidecar)
    return result
