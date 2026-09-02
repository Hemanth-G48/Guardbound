"""Common evaluation result container + table renderers.

Every metric in the project (ASR, MMLU, MTBench, over-refusal,
guardrail F1, etc.) emits an ``EvaluationResult`` row, and the
reporting layer consumes these rows to produce JSONL, Markdown, and
LaTeX tables.

Design notes:
    - Results are simple dataclasses so they can be JSON-serialized
      and consumed by downstream phases (Phase 8 baselines, Phase 9
      experiments) without coupling to any one metric module.
    - The renderers sort and highlight best/runner-up dynamically
      from the result data; no model rankings are hard-coded.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

@dataclass
class EvaluationResult:
    """A single (model, variant, eta, attack/dataset, metric) measurement."""

    model: str
    defense_variant: str
    metric: str
    value: float
    dataset: str | None = None
    attack: str | None = None
    eta: float | None = None
    embedding_model: str | None = None
    checkpoint: str | None = None
    judge_model: str | None = None
    judge_prompt_version: str | None = None
    seed: int | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str | None = None
    protocol_version: str | None = None
    n: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #

def write_jsonl(results: Iterable[EvaluationResult], path: str | Path) -> int:
    """Write ``results`` as one JSON object per line.  Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[EvaluationResult]:
    """Load results from a JSONL file written by :func:`write_jsonl`."""
    out: list[EvaluationResult] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            extras = d.pop("extras", {}) or {}
            r = EvaluationResult(**d)
            r.extras = extras
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Filtering helpers
# --------------------------------------------------------------------------- #

def filter_results(
    results: Sequence[EvaluationResult],
    *,
    metric: str | None = None,
    attack: str | None = None,
    dataset: str | None = None,
    model: str | None = None,
    variant: str | None = None,
) -> list[EvaluationResult]:
    """Return results matching all provided (non-None) filter criteria."""
    out: list[EvaluationResult] = []
    for r in results:
        if metric is not None and r.metric != metric:
            continue
        if attack is not None and r.attack != attack:
            continue
        if dataset is not None and r.dataset != dataset:
            continue
        if model is not None and r.model != model:
            continue
        if variant is not None and r.defense_variant != variant:
            continue
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Direction conventions for ranking
# --------------------------------------------------------------------------- #

# Metrics where HIGHER is better
HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "MMLU",
    "MTBench",
    "MTBench_turn1",
    "MTBench_turn2",
    "accuracy",
    "F1",
    "precision",
    "recall",
})

# Metrics where LOWER is better
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "ASR",
    "over_refusal_rate",
    "FNR",
})


def better(metric: str, a: float, b: float) -> bool:
    """Return True if ``a`` is a better value than ``b`` for ``metric``."""
    if a == b:
        return False
    if metric in LOWER_IS_BETTER:
        return a < b
    if metric in HIGHER_IS_BETTER:
        return a > b
    # Default: higher is better
    return a > b


# --------------------------------------------------------------------------- #
# Table renderers
# --------------------------------------------------------------------------- #

def _rank_within_metric(results: Sequence[EvaluationResult]) -> dict[int, int]:
    """Return mapping ``idx_in_results -> rank_in_metric (0 = best)``."""
    indexed = sorted(
        enumerate(results),
        key=lambda kv: (kv[1].value, kv[1].model, kv[1].defense_variant),
    )
    rank: dict[int, int] = {}
    sorted_metric = sorted(
        {r.metric for r in results},
    )
    for metric in sorted_metric:
        sub = [(i, r) for i, r in indexed if r.metric == metric]
        # Sort by direction (higher-is-better for unknown metrics by default)
        reverse = metric in HIGHER_IS_BETTER or metric not in LOWER_IS_BETTER
        sub.sort(key=lambda kv: kv[1].value, reverse=reverse)
        for rank_idx, (orig_idx, _) in enumerate(sub):
            rank[orig_idx] = rank_idx
    return rank


def to_markdown_table(
    results: Sequence[EvaluationResult],
    *,
    title: str | None = None,
    columns: Sequence[str] = ("model", "defense_variant", "eta", "value"),
) -> str:
    """Render ``results`` as a Markdown table.

    Best and runner-up per metric are highlighted with **bold**.  Rank
    is computed dynamically from the result data; no hard-coded
    rankings.
    """
    if not results:
        return "_No results._\n"

    # Compute per-metric rank
    rank: dict[int, int] = _rank_within_metric(results)

    lines: list[str] = []
    if title:
        lines.append(f"### {title}\n")
    header = "| " + " | ".join(columns) + " | rank |"
    sep = "| " + " | ".join("---" for _ in columns) + " | --- |"
    lines.append(header)
    lines.append(sep)
    for i, r in enumerate(results):
        cells = [str(_pick(r, c)) for c in columns]
        if rank.get(i) == 0:
            cells = [f"**{c}**" for c in cells]
        elif rank.get(i) == 1:
            cells = [f"*{c}*" for c in cells]
        rank_cell = "🥇" if rank.get(i) == 0 else ("🥈" if rank.get(i) == 1 else "")
        lines.append("| " + " | ".join(cells) + f" | {rank_cell} |")
    return "\n".join(lines) + "\n"


def to_latex_table(
    results: Sequence[EvaluationResult],
    *,
    title: str | None = None,
    columns: Sequence[str] = ("model", "defense_variant", "eta", "value"),
) -> str:
    """Render ``results`` as a LaTeX tabular."""
    if not results:
        return "% No results.\n"
    rank: dict[int, int] = _rank_within_metric(results)
    lines: list[str] = []
    if title:
        lines.append(f"% {title}")
    col_spec = " | ".join("l" for _ in columns) + " | l"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(" & ".join(columns) + r" & rank \\")
    lines.append(r"\hline")
    for i, r in enumerate(results):
        cells = [_latex_escape(str(_pick(r, c))) for c in columns]
        if rank.get(i) == 0:
            cells = [f"\\textbf{{{c}}}" for c in cells]
        elif rank.get(i) == 1:
            cells = [f"\\textit{{{c}}}" for c in cells]
        rank_cell = "1" if rank.get(i) == 0 else ("2" if rank.get(i) == 1 else "")
        lines.append(" & ".join(cells) + f" & {rank_cell}" + r" \\")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def _pick(r: EvaluationResult, key: str) -> Any:
    if key == "value":
        return _format_value(r.value)
    return getattr(r, key, "")


def _format_value(v: float | None) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def _latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)
