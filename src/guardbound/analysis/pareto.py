"""Pareto-frontier analysis (Phase 9).

A point ``(asr, helpfulness)`` is dominated if there exists another
point with strictly lower ASR and strictly higher helpfulness, or
equal ASR and strictly higher helpfulness, or equal helpfulness and
strictly lower ASR.  This is the standard weak-Pareto rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Point:
    """A point in (asr, helpfulness) space.  NaN/None values are
    excluded from dominance calculations."""
    asr: float
    helpfulness: float
    label: Any = None

    def is_valid(self) -> bool:
        if self.asr is None or self.helpfulness is None:
            return False
        try:
            a = float(self.asr); h = float(self.helpfulness)
        except (TypeError, ValueError):
            return False
        if a != a or h != h:  # NaN
            return False
        return True


def dominates(a: Point, b: Point) -> bool:
    """Return True if point ``a`` weakly dominates point ``b``.

    A weakly dominates B iff A's ASR is <= B's ASR AND A's
    helpfulness is >= B's helpfulness, with at least one strict
    inequality.  This is the standard Pareto rule for the
    (minimize, maximize) axis.
    """
    if not (a.is_valid() and b.is_valid()):
        return False
    if a.asr < b.asr and a.helpfulness > b.helpfulness:
        return True
    if a.asr == b.asr and a.helpfulness > b.helpfulness:
        return True
    if a.asr < b.asr and a.helpfulness == b.helpfulness:
        return True
    return False


def compute_pareto_front(points: Iterable[Point]) -> list[Point]:
    """Return the non-dominated subset of ``points`` in input order.

    Edge cases:
        - empty input -> empty output
        - one point   -> [point] if valid else []
        - duplicates  -> deduplicated by (asr, helpfulness) values
    """
    seen: set[tuple[float, float]] = set()
    cleaned: list[Point] = []
    for p in points:
        if not p.is_valid():
            continue
        key = (float(p.asr), float(p.helpfulness))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(p)
    front: list[Point] = []
    for i, pi in enumerate(cleaned):
        dominated_by_others = False
        for j, pj in enumerate(cleaned):
            if i == j:
                continue
            if dominates(pj, pi):
                dominated_by_others = True
                break
        if not dominated_by_others:
            front.append(pi)
    return front
