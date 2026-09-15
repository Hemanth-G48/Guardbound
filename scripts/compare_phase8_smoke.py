"""Phase 8 — compare smoke runs (NBF OFF vs NBF ON).

Reads two JSONL result files produced by scripts/run_reproduction.py
(same goals, same models, same generation settings; only NBF differs) and
prints, per goal:

    Attack-only (A):   success / turns / refusals
    Attack+NBF  (B):   success / turns / refusals / filtered queries / NBF scores

Also proves NBF participation: mode-B records must contain at least one of
{nonempty nbf_scores, filtered_queries > 0} while mode-A records must contain
none. Exit code 1 if the participation invariant is violated.

Usage:
    python scripts/compare_phase8_smoke.py \
        --a results/phase8/smoke_A_crescendo_qwenphi.jsonl \
        --b results/phase8/smoke_B_crescendo_qwenphi.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    """Map goal -> record (last occurrence wins)."""
    if not path.exists():
        raise SystemExit(f"missing smoke file: {path}")
    by_goal: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_goal[rec["goal"]] = rec
    return by_goal


def short_goal(goal: str, width: int = 46) -> str:
    return goal if len(goal) <= width else goal[: width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="NBF-OFF JSONL")
    ap.add_argument("--b", required=True, help="NBF-ON JSONL")
    args = ap.parse_args()

    a = load(Path(args.a))
    b = load(Path(args.b))
    goals = [g for g in a if g in b]
    if not goals:
        raise SystemExit("no overlapping goals between the two runs")

    print(f"{'goal':<46} | {'A: succ':<7} {'turns':<5} {'ref':<4} | "
          f"{'B: succ':<7} {'turns':<5} {'ref':<4} {'filt':<4} nbf_scores")
    print("-" * 130)

    violations: list[str] = []
    totals = {"a_succ": 0, "b_succ": 0, "a_ref": 0, "b_ref": 0, "filt": 0}

    for goal in goals:
        ra, rb = a[goal], b[goal]
        # Record schema (run_reproduction.run_one) is top-level:
        #   nbf_scores[]      every candidate the barrier scored (incl. the
        #                     ones that were filtered before reaching target)
        #   filtered_queries  count of candidates the barrier rejected
        #   refusal_count     refused/disclaimer turns
        nbf_scores = list(rb.get("nbf_scores") or [])
        filtered = int(rb.get("filtered_queries") or 0)
        a_nbf_scores = list(ra.get("nbf_scores") or [])
        a_filtered = int(ra.get("filtered_queries") or 0)
        if a_nbf_scores or a_filtered:
            violations.append(f"A record has NBF artifacts: {short_goal(goal)}")
        # Participation proof: mode B must show scored candidates or an actual
        # filter event. A B record with turns but no NBF artifacts means the
        # barrier never entered the loop — a harness bug, not a result.
        if not nbf_scores and not filtered and rb.get("num_turns", 0) > 0:
            violations.append(f"B record shows no NBF participation: "
                              f"{short_goal(goal)}")

        ra_ref = ra.get("refusal_count", 0) or 0
        rb_ref = rb.get("refusal_count", 0) or 0
        totals["a_succ"] += bool(ra["success"])
        totals["b_succ"] += bool(rb["success"])
        totals["a_ref"] += ra_ref
        totals["b_ref"] += rb_ref
        totals["filt"] += filtered

        scores = ",".join(
            f"{s:.2f}" for s in nbf_scores[:4] if isinstance(s, (int, float))
        ) or "-"
        print(f"{short_goal(goal):<46} | {str(ra['success']):<7} "
              f"{ra['num_turns']:<5} {ra_ref:<4} | "
              f"{str(rb['success']):<7} {rb['num_turns']:<5} "
              f"{rb_ref:<4} {filtered:<4} [{scores}]")

    print("-" * 130)
    print(f"totals: A success={totals['a_succ']}/{len(goals)} "
          f"refusals={totals['a_ref']} | B success={totals['b_succ']}/{len(goals)} "
          f"refusals={totals['b_ref']} filtered={totals['filt']}")

    if violations:
        print("\nPARTICIPATION VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        raise SystemExit(1)
    print("\nNBF participation invariant: OK "
          "(A has no NBF artifacts; B has per-turn NBF scores and/or "
          "filter events)")


if __name__ == "__main__":
    main()
