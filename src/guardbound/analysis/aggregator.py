"""Aggregation helpers (Phase 9).

These helpers operate on the project's existing
``EvaluationResult`` rows and do not introduce a new schema.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable, Sequence

from ..evaluation import EvaluationResult


def mean_of(values: Sequence[float | None]) -> float | None:
    """Return the mean of a list of numeric values, ignoring None.

    Returns None if the list is empty after filtering.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(mean(clean))


def aggregate_by(
    results: Iterable[EvaluationResult],
    *,
    key_fields: tuple[str, ...] = ("model", "defense_variant", "eta", "seed"),
    metric: str | None = None,
) -> list[dict]:
    """Group results by ``key_fields`` and return aggregated rows.

    Each output row contains:
        the key fields
        ``n``   number of underlying rows
        ``value``  the mean of ``value`` across the group
    """
    buckets: dict[tuple, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        if metric is not None and r.metric != metric:
            continue
        key = tuple(_safe_get(r, k) for k in key_fields)
        buckets[key].append(r)
    out: list[dict] = []
    for key, items in buckets.items():
        row: dict = {k: v for k, v in zip(key_fields, key)}
        vals = [r.value for r in items if r.value is not None]
        row["n"] = len(items)
        row["value"] = mean_of(vals)
        out.append(row)
    out.sort(key=lambda r: tuple(str(r.get(k, "")) for k in key_fields))
    return out


def stratify_by_attack(results: Iterable[EvaluationResult]) -> dict[str, list[EvaluationResult]]:
    out: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        if r.attack is None:
            out.setdefault("(none)", []).append(r)
        else:
            out[r.attack].append(r)
    return dict(out)


def stratify_by_model(results: Iterable[EvaluationResult]) -> dict[str, list[EvaluationResult]]:
    out: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        out[r.model].append(r)
    return dict(out)


def _safe_get(obj: Any, key: str) -> Any:
    """Read a field from an EvaluationResult or a dict-like."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
