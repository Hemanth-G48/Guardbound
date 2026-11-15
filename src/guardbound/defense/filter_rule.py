"""The paper's NBF Q-filter decision rule, defined exactly once.

Paper (arXiv:2503.00187v3, Sec. 5.2): a query is *filtered* when

    h(x, u) + η ≥ 0

and therefore *admitted* only when ``h < −η``. η is a non-negative margin
(``η = 1e-3`` is the headline evaluation value).

The author's release spells the same rule with a raw admission threshold:

    threshold = -args.threshold          # args.threshold is η
    if score < threshold: accept         # accept iff score < -η

This module owns both the inequality (:func:`admits`) and the conversion
(:func:`threshold_from_eta`) so the steered runtime and every attack-level
candidate filter cannot drift apart again (Phase 23 defect D-01).
"""
from __future__ import annotations


def threshold_from_eta(eta: float) -> float:
    """Raw admission threshold for the paper's margin: ``-η``.

    ``admits(score, threshold_from_eta(eta))`` is exactly ``score < -eta``.
    """
    return -float(eta)


def admits(score: float, threshold: float) -> bool:
    """Raw admission inequality shared by every attack-level filter.

    ``threshold`` is a raw admission threshold, not a margin. Convert a margin
    with :func:`threshold_from_eta` before calling this.
    """
    return float(score) < float(threshold)


def is_admitted(score: float, eta: float) -> bool:
    """True when the paper's rule admits the query (``score < −η``)."""
    return admits(score, threshold_from_eta(eta))


def is_filtered(score: float, eta: float) -> bool:
    """True when the paper's rule filters the query (``score + η ≥ 0``)."""
    return not is_admitted(score, eta)
