"""Qualitative-claim verification (Phase 9).

Each claim is a deterministic, data-only check.  Given the project's
``EvaluationResult`` rows, the verifier reports:

    PASS                  claim supported by the data
    FAIL                  claim contradicted by the data
    DIRECTIONALLY-CONSISTENT  data moves in the claimed direction
                              but is not statistically conclusive
    UNVERIFIED            insufficient / no relevant data

The verifier never fabricates results.  Missing data is reported
as ``UNVERIFIED — insufficient result data``; it is never silently
mapped to ``PASS``.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..evaluation import EvaluationResult


class ClaimStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    DIRECTIONALLY_CONSISTENT = "DIRECTIONALLY-CONSISTENT"
    UNVERIFIED = "UNVERIFIED"


@dataclass
class ClaimCheck:
    claim_id: str
    claim: str
    status: ClaimStatus
    required_result_rows: int
    observed_values: list[float] = field(default_factory=list)
    expected_direction: str = ""
    actual_direction: str = ""
    notes: str = ""


@dataclass
class FindingsReport:
    checks: list[ClaimCheck]
    missing_data: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines: list[str] = ["# Phase 9 Findings Verification", ""]
        lines.append("| claim_id | claim | status | observed | notes |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in self.checks:
            obs = ", ".join(f"{v:.4f}" for v in c.observed_values[:5])
            if len(c.observed_values) > 5:
                obs += f" (+{len(c.observed_values) - 5} more)"
            lines.append(
                f"| {c.claim_id} | {c.claim} | {c.status.value} | "
                f"{obs or '—'} | {c.notes} |"
            )
        if self.missing_data:
            lines.append("")
            lines.append("## Missing data")
            for m in self.missing_data:
                lines.append(f"- {m}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Default claims
# --------------------------------------------------------------------------- #

def list_default_claims() -> list[dict[str, Any]]:
    """Return the canonical paper-claim set.

    Each entry has: ``claim_id``, ``claim``, ``filter_fn``,
    ``metric``, ``direction`` (lower/equal/higher).  The filter
    selects the result rows relevant to the claim.
    """
    return [
        {
            "claim_id": "C1",
            "claim": "Steering at eta=1e-3 substantially reduces ASR.",
            "metric": "ASR",
            "direction": "lower",
            "filter_fn": lambda r: r.metric == "ASR" and r.attack is not None,
        },
        {
            "claim_id": "C2",
            "claim": "LoRA SFT may increase ASR in some settings.",
            "metric": "ASR",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "ASR" and r.defense_variant == "lora_sft",
        },
        {
            "claim_id": "C3",
            "claim": "Increasing eta generally reduces ASR (with diminishing returns).",
            "metric": "ASR",
            "direction": "lower",
            "filter_fn": lambda r: (r.metric == "ASR"
                                    and r.defense_variant == "guardbound"
                                    and r.eta is not None),
        },
        {
            "claim_id": "C4",
            "claim": "Utility may decay beyond the practical knee.",
            "metric": "MTBench",
            "direction": "lower",
            "filter_fn": lambda r: (r.metric == "MTBench"
                                    and r.defense_variant == "guardbound"
                                    and r.eta is not None),
        },
        {
            "claim_id": "C5",
            "claim": "Removing L_SS increases ASR.",
            "metric": "ASR",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "ASR" and r.extras.get("ablation") == "drop_ss",
        },
        {
            "claim_id": "C6",
            "claim": "Removing L_SI increases ASR.",
            "metric": "ASR",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "ASR" and r.extras.get("ablation") == "drop_si",
        },
        {
            "claim_id": "C7",
            "claim": "kappa=3 provides a strong overall trade-off.",
            "metric": "MTBench",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "MTBench" and r.extras.get("kappa") is not None,
        },
        {
            "claim_id": "C8",
            "claim": "DistilRoBERTa-NBF may have lower ASR but higher over-refusal than MPNet-NBF.",
            "metric": "over_refusal_rate",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "over_refusal_rate",
        },
        {
            "claim_id": "C9",
            "claim": "NBF trained without an attack can generalize to the unseen attack.",
            "metric": "ASR",
            "direction": "lower",
            "filter_fn": lambda r: (r.metric == "ASR"
                                    and r.extras.get("exclude_attacks")),
        },
        {
            "claim_id": "C10",
            "claim": "Adaptive attacks can increase ASR under steering.",
            "metric": "ASR",
            "direction": "higher",
            "filter_fn": lambda r: r.metric == "ASR" and r.attack in ("adaptive", "adaptive_crescendo"),
        },
    ]


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #

def check_claims(
    results: Iterable[EvaluationResult],
    claims: Sequence[dict] | None = None,
) -> FindingsReport:
    """Run every claim against the supplied result rows.

    A claim is reported as:
      - PASS  if the direction matches AND the magnitude is large.
      - DIRECTIONALLY-CONSISTENT  if direction matches but magnitude
        is small / single-point.
      - FAIL  if direction contradicts.
      - UNVERIFIED  if no rows match the filter.
    """
    claims = claims or list_default_claims()
    results = list(results)
    checks: list[ClaimCheck] = []
    missing: list[str] = []

    for spec in claims:
        rows = [r for r in results if spec["filter_fn"](r)]
        if not rows:
            checks.append(ClaimCheck(
                claim_id=spec["claim_id"],
                claim=spec["claim"],
                status=ClaimStatus.UNVERIFIED,
                required_result_rows=0,
                notes="UNVERIFIED — insufficient result data",
            ))
            missing.append(f"{spec['claim_id']}: no rows matched the filter")
            continue
        values = [r.value for r in rows if r.value is not None]
        if not values:
            checks.append(ClaimCheck(
                claim_id=spec["claim_id"],
                claim=spec["claim"],
                status=ClaimStatus.UNVERIFIED,
                required_result_rows=len(rows),
                notes="UNVERIFIED — no numeric values in matching rows",
            ))
            continue

        direction = _infer_direction(values, spec["direction"])
        status = _status_from_direction(direction, spec["direction"], values)
        checks.append(ClaimCheck(
            claim_id=spec["claim_id"],
            claim=spec["claim"],
            status=status,
            required_result_rows=len(rows),
            observed_values=values,
            expected_direction=spec["direction"],
            actual_direction=direction,
            notes=_make_notes(spec, values, direction, status),
        ))
    return FindingsReport(checks=checks, missing_data=missing)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _infer_direction(values: Sequence[float], expected: str) -> str:
    """Return 'higher', 'lower', or 'flat' based on the trend of values."""
    if len(values) < 2:
        return "flat"
    first, last = values[0], values[-1]
    diff = last - first
    if abs(diff) < 1e-9:
        return "flat"
    return "higher" if diff > 0 else "lower"


def _status_from_direction(actual: str, expected: str,
                          values: Sequence[float]) -> ClaimStatus:
    if actual == "flat" and expected in ("lower", "higher"):
        return ClaimStatus.DIRECTIONALLY_CONSISTENT
    if actual == expected:
        # Magnitude check: a small but consistent difference counts
        # as DIRECTIONALLY-CONSISTENT rather than PASS.
        if len(values) >= 2:
            span = max(values) - min(values)
            if span < 0.01:
                return ClaimStatus.DIRECTIONALLY_CONSISTENT
        return ClaimStatus.PASS
    if actual in ("higher", "lower") and expected in ("higher", "lower"):
        return ClaimStatus.FAIL
    return ClaimStatus.UNVERIFIED


def _make_notes(spec: dict, values: Sequence[float],
                direction: str, status: ClaimStatus) -> str:
    return (
        f"expected={spec['direction']}, observed={direction}, "
        f"n_values={len(values)}, span={max(values) - min(values):.4f}, "
        f"status={status.value}"
    )
