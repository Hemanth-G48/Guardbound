"""Table generation (Phase 9).

Phase 7's reporting module already produces Markdown / LaTeX tables
from ``EvaluationResult`` rows.  This module adds:
  - ``pivot_results`` for reshaping long-form rows into a wide-form
    (model, variant) x metric table.
  - helpers that produce Table-1 / Table-2 / Table-4 / Table-13 style
    tables dynamically from result data (no hard-coded rankings).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from ..evaluation import (
    EvaluationResult,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    to_latex_table,
    to_markdown_table,
)


def pivot_results(
    results: Iterable[EvaluationResult],
    *,
    rows_field: str = "model",
    cols_field: str = "defense_variant",
    metric: str = "ASR",
) -> list[dict[str, float | None]]:
    """Pivot long-form results into a wide-form (row x col) table.

    Returns a list of dicts; one per row field value.  Each dict
    contains the row key plus one entry per col field value whose
    value is the mean of ``metric`` over matching rows.
    """
    grid: dict[tuple[str, str], list[float]] = defaultdict(list)
    row_keys: set[str] = set()
    col_keys: set[str] = set()
    for r in results:
        if r.metric != metric:
            continue
        row_key = _get_field(r, rows_field) or "(none)"
        col_key = _get_field(r, cols_field) or "(none)"
        row_keys.add(row_key)
        col_keys.add(col_key)
        if r.value is not None:
            grid[(row_key, col_key)].append(float(r.value))
    out: list[dict[str, float | None]] = []
    for rk in sorted(row_keys):
        row: dict[str, float | None] = {rows_field: rk}
        for ck in sorted(col_keys):
            vals = grid.get((rk, ck), [])
            if vals:
                row[ck] = sum(vals) / len(vals)
            else:
                row[ck] = None
        out.append(row)
    return out


def _get_field(r: EvaluationResult, key: str) -> str | None:
    if key == "model":
        return r.model
    if key == "defense_variant":
        return r.defense_variant
    if key == "metric":
        return r.metric
    if key == "attack":
        return r.attack
    if key == "dataset":
        return r.dataset
    return getattr(r, key, None)


def to_pivot_markdown(
    pivot: list[dict[str, float | None]],
    *,
    rows_field: str = "model",
    cols_field: str = "defense_variant",
    metric: str = "ASR",
    better_is: str | None = None,
) -> str:
    """Render a pivot table as Markdown with best/runner-up highlighting.

    ``better_is`` may be "lower" or "higher".  When None, the metric
    direction is inferred from the Phase 7 conventions
    (``LOWER_IS_BETTER`` / ``HIGHER_IS_BETTER``).
    """
    if not pivot:
        return "_No data._\n"
    if better_is is None:
        better_is = "lower" if metric in LOWER_IS_BETTER else "higher"
    cols = sorted({k for row in pivot for k in row if k != rows_field})
    header = "| " + rows_field + " | " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for row in pivot:
        cells = []
        for c in cols:
            v = row.get(c)
            if v is None:
                cells.append("")
            else:
                cells.append(f"{v:.4f}")
        # Compute best / runner-up for this row.
        numeric = [(c, row.get(c)) for c in cols if row.get(c) is not None]
        if numeric:
            numeric.sort(key=lambda kv: kv[1] if kv[1] is not None else 0,
                          reverse=(better_is == "higher"))
            best_col = numeric[0][0]
            runnerup_col = numeric[1][0] if len(numeric) > 1 else None
        else:
            best_col = None
            runnerup_col = None
        cells_marked: list[str] = []
        for c, txt in zip(cols, cells):
            if c == best_col:
                cells_marked.append(f"**{txt}**")
            elif c == runnerup_col:
                cells_marked.append(f"*{txt}*")
            else:
                cells_marked.append(txt)
        lines.append(f"| {row[rows_field]} | " + " | ".join(cells_marked) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["pivot_results", "to_pivot_markdown", "to_markdown_table", "to_latex_table"]
