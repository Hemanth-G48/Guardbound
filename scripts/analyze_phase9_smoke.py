"""Phase 9 smoke analysis: attack x NBF matrix, invariant checks, saturation.

Reads one or more result JSONL files produced by ``run_reproduction.py`` and
reports, per attack and NBF mode:

  * goals / successes / ASR
  * average turns, refusals, filtered queries
  * the rubric-score distribution (the saturation check)
  * termination reasons
  * runtime and peak VRAM

It also enforces the Phase 9 artifact invariants:

  * every required telemetry field is present
  * NBF OFF -> no NBF artifacts at all (no scores, zero filtered)
  * NBF ON -> the barrier computation was actually invoked

Usage:
    python scripts/analyze_phase9_smoke.py results/phase9/A_all_3goals.jsonl \
                                          results/phase9/B_all_3goals.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Telemetry required by the Phase 9 spec (section 4).
REQUIRED_FIELDS = [
    "goal_id",
    "goal",
    "attack",
    "nbf_enabled",
    "success",
    "num_turns",
    "refusal_count",
    "filtered_queries",
    "nbf_scores",
    "final_response",
    "runtime",
    "peak_vram_gb",
    "termination_reason",
    "attacker_model",
    "target_model",
    "evaluator_model",
    "embedding_model",
    "checkpoint_sha256",
    "seed",
    "mode",
    "max_turns",
    "rubric_scores",
    "llm_calls",
]

ATTACK_ORDER = ["crescendo", "actor_attack", "opposite_day", "acronym"]


def load(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"WARNING: missing result file: {path}")
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"MALFORMED RECORD {path}:{lineno}: {exc}")
    return records


def check_fields(records: list[dict]) -> list[str]:
    problems: list[str] = []
    for rec in records:
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            problems.append(
                f"  {rec.get('attack')} / goal {rec.get('goal_id')}: missing {missing}"
            )
    return problems


def check_invariants(records: list[dict]) -> list[str]:
    """NBF OFF must contain no NBF artifacts; NBF ON must have invoked the barrier."""
    problems: list[str] = []
    for rec in records:
        tag = f"{rec.get('attack')} / goal {rec.get('goal_id')} / nbf={rec.get('nbf_enabled')}"
        scores = rec.get("nbf_scores") or []
        filtered = rec.get("filtered_queries") or 0

        if rec.get("nbf_enabled") is False:
            if scores:
                problems.append(f"  VIOLATION {tag}: NBF OFF but nbf_scores={len(scores)}")
            if filtered:
                problems.append(f"  VIOLATION {tag}: NBF OFF but filtered={filtered}")
        else:
            # The invariant is that the barrier computation ran, not that it
            # filtered anything: a run may legitimately accept every candidate.
            if not scores:
                problems.append(
                    f"  VIOLATION {tag}: NBF ON but no barrier scores recorded "
                    "(computation not demonstrably invoked)"
                )
    return problems


def table(records: list[dict]) -> None:
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for rec in records:
        groups[(rec.get("attack"), bool(rec.get("nbf_enabled")))].append(rec)

    header = (
        f"{'Attack':<14} {'NBF':<5} {'Goals':>5} {'Succ':>5} {'ASR':>7} "
        f"{'AvgTurns':>8} {'Refuse':>7} {'Filtered':>8} {'Runtime(s)':>10} {'PeakVRAM':>9}"
    )
    print(header)
    print("-" * len(header))

    attacks = [a for a in ATTACK_ORDER if any(k[0] == a for k in groups)]
    attacks += sorted({k[0] for k in groups} - set(attacks))

    for attack in attacks:
        for nbf in (False, True):
            rows = groups.get((attack, nbf))
            if not rows:
                continue
            n = len(rows)
            succ = sum(1 for r in rows if r.get("success"))
            turns = [r.get("num_turns") or 0 for r in rows]
            runtime = sum(r.get("runtime") or 0.0 for r in rows)
            peak = max((r.get("peak_vram_gb") or 0.0) for r in rows)
            print(
                f"{attack:<14} {('ON' if nbf else 'OFF'):<5} {n:>5} {succ:>5} "
                f"{(succ / n * 100 if n else 0):>6.1f}% "
                f"{(sum(turns) / n if n else 0):>8.2f} "
                f"{sum(r.get('refusal_count') or 0 for r in rows):>7} "
                f"{sum(r.get('filtered_queries') or 0 for r in rows):>8} "
                f"{runtime:>10.1f} {peak:>8.2f}G"
            )


def saturation(records: list[dict]) -> None:
    """Rubric score histogram — the key test for a saturating judge."""
    dist: Counter[int] = Counter()
    per_attack: dict[str, Counter] = defaultdict(Counter)
    per_attack_nbf: dict[str, dict[bool, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    nbf_on_dist: Counter[int] = Counter()
    nbf_off_dist: Counter[int] = Counter()

    for rec in records:
        attack = rec.get("attack", "?")
        nbf = bool(rec.get("nbf_enabled"))
        for score in rec.get("rubric_scores") or []:
            try:
                s = int(score)
            except (TypeError, ValueError):
                continue
            dist[s] += 1
            per_attack[attack][s] += 1
            per_attack_nbf[attack][nbf][s] += 1
            (nbf_on_dist if nbf else nbf_off_dist)[s] += 1

    total = sum(dist.values())
    print(f"Rubric score distribution over {total} scored turns")
    print(f"{'Score':<7} {'Count':>7} {'Share':>8}")
    for s in sorted(dist):
        print(f"{s:<7} {dist[s]:>7} {dist[s] / total * 100:>7.1f}%")
    never = [s for s in (1, 2, 3, 4, 5) if s not in dist]
    print(f"  scores never emitted: {never if never else 'none'}")
    print(f"  max score reached   : {max(dist) if dist else 'n/a'}")
    print()
    print("Rubric score distribution by attack × NBF mode:")
    for attack in ATTACK_ORDER:
        for nbf in (False, True):
            c = per_attack_nbf[attack][nbf]
            if not c:
                continue
            s = sum(c.values())
            print(f"  {attack:<14} {'ON' if nbf else 'OFF':<5} : "
                  f"{dict(sorted(c.items()))} (n={s})")
    print()
    print("Marginal rubric histograms:")
    print(f"  NBF OFF   : {dict(sorted(nbf_off_dist.items()))}")
    print(f"  NBF ON    : {dict(sorted(nbf_on_dist.items()))}")
    print()


def reasons(records: list[dict]) -> None:
    print("Termination reasons")
    counts: Counter[str] = Counter()
    for rec in records:
        counts[str(rec.get("termination_reason"))] += 1
    for reason, n in counts.most_common():
        print(f"  {reason:<26} {n}")
    print()


def models(records: list[dict]) -> None:
    print("Model identity check (must be uniform across every record)")
    for field in (
        "attacker_model",
        "target_model",
        "evaluator_model",
        "embedding_model",
        "checkpoint_sha256",
    ):
        vals = {str(r.get(field)) for r in records}
        flag = "OK " if len(vals) == 1 else "MIXED"
        print(f"  {flag} {field:<18} {sorted(vals)}")
    print()


def llm_calls(records: list[dict]) -> None:
    print("Model call counts (per run, summed)")
    by_role: dict[str, int] = defaultdict(int)
    for rec in records:
        calls = rec.get("llm_calls") or {}
        for role in ("attacker", "target", "evaluator"):
            by_role[role] += calls.get(role) or 0
    for role, total in by_role.items():
        print(f"  {role:<10} {total}")
    print()


def main(argv: list[str]) -> int:
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2

    records = load(paths)
    if not records:
        print("No records loaded.")
        return 1

    print("=" * 78)
    print(f"PHASE 9 SMOKE ANALYSIS — {len(records)} records from {len(paths)} file(s)")
    print("=" * 78)

    models(records)

    print("Attack x NBF matrix")
    table(records)
    print()

    saturation(records)

    reasons(records)

    llm_calls(records)

    print("Artifact validation")
    field_problems = check_fields(records)
    inv_problems = check_invariants(records)
    if not field_problems:
        print("  required fields: OK (all present in all records)")
    else:
        print("  required fields: PROBLEMS")
        print("\n".join(field_problems))
    if not inv_problems:
        print("  NBF participation invariant: OK")
    else:
        print("  NBF participation invariant: PROBLEMS")
        print("\n".join(inv_problems))

    total = len(records)
    misses = sum(1 for r in records if not r.get("rubric_scores"))
    print(f"  runs with zero scored turns: {misses}/{total}")
    empty = sum(1 for r in records if not r.get("turns"))
    print(f"  runs with no turns at all  : {empty}/{total}")

    ok = not field_problems and not inv_problems
    print()
    print(f"ARTIFACT VALIDATION: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
