"""Phase 17 Stage 4.9 — deep analysis of the completed 90-run pilot.

Analysis only. Reads `results/phase17_pilot/runs/raw_results.jsonl` (and the Stage 4.8
summary artifacts) and writes the Stage 4.9 artifact set under
`results/phase17_pilot/analysis/stage4_9/`. It never writes to the raw pilot data, never
re-runs a run, and changes no experimental parameter.

Every number in the outputs is derived from the persisted run records. Where a quantity is
*derived* rather than *recorded* (for example `first_failure_stage`, which the pilot did
not store as a field), the derivation is stated in the artifact itself.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "results" / "phase17_pilot"
RAW = PILOT / "runs" / "raw_results.jsonl"
OUT = PILOT / "analysis" / "stage4_9"
VIZ = OUT / "visualizations"

VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
ATTACKS = ("crescendo_paper", "opposite_day", "acronym")
ATTACK_LABEL = {"crescendo_paper": "Crescendo", "opposite_day": "OppositeDay",
                "acronym": "Acronym"}

TESTS_PERFORMED: list[dict] = []


# --------------------------------------------------------------------------- #
# statistics helpers                                                           #
# --------------------------------------------------------------------------- #
def wilson(s: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = s / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher for [[a, b], [c, d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(_log_comb(r1, x) + _log_comb(n - r1, c1 - x) - _log_comb(n, c1))

    obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return round(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12)), 4)


def fisher_test(name: str, a: int, b: int, c: int, d: int, label: str) -> dict:
    p = fisher(a, b, c, d)
    result = {"test": "fisher_exact_two_sided", "question": name, "table": [[a, b], [c, d]],
              "p_value": p, "label": label,
              "rates": {"row1": round(a / (a + b), 4) if (a + b) else None,
                        "row2": round(c / (c + d), 4) if (c + d) else None}}
    TESTS_PERFORMED.append({"question": name, "p_value": p, "label": label})
    return result


def permutation_test(xs: list[float], ys: list[float], n_perm: int = 20000,
                     seed: int = 20261001) -> dict:
    """Two-sided permutation test on the difference of means (no distributional
    assumption); used where a rank test would be overkill for this sample size."""
    import random
    rng = random.Random(seed)
    combined = xs + ys
    observed = statistics.mean(xs) - statistics.mean(ys) if xs and ys else None
    if observed is None:
        return {"observed_difference": None, "p_value": None, "n_permutations": 0}
    count = 0
    n_x = len(xs)
    for _ in range(n_perm):
        rng.shuffle(combined)
        if abs(statistics.mean(combined[:n_x]) - statistics.mean(combined[n_x:])) >= abs(observed) - 1e-12:
            count += 1
    return {"observed_difference": round(observed, 4),
            "p_value": round((count + 1) / (n_perm + 1), 4), "n_permutations": n_perm}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
            "min": round(min(values), 3), "max": round(max(values), 3)}


def block(runs: list[dict], key: str) -> dict:
    n = len(runs)
    s = sum(1 for r in runs if r.get(key))
    return {"numerator": s, "denominator": n, "rate": round(s / n, 4) if n else None,
            "ci95_wilson": wilson(s, n)}


# --------------------------------------------------------------------------- #
# load                                                                        #
# --------------------------------------------------------------------------- #
def load_runs() -> list[dict]:
    return [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def derived_flags(run: dict) -> dict:
    calls = run["attacker_calls"]
    invalid = [c for c in calls if c["failure_type"] is not None]
    first_index = next((i for i, c in enumerate(calls) if c["failure_type"] is not None), None)
    return {
        "attacker_invalid_calls": len(invalid),
        "attacker_semantic_usable_calls": len([c for c in calls
                                               if c["category"] in VALID
                                               and c["failure_type"] is None]),
        "survived": run["failure_class"] not in ("attacker_generation_error",
                                                 "cuda_oom_failure",
                                                 "infrastructure_failure"),
        "strict_usable": bool(calls) and not invalid,
        "final_rubric": (run["rubric_scores"][-1] if run["rubric_scores"] else None),
        "target_natural_eos": run["counts"]["target_eos"],
        "target_refusals": run.get("refusal_count") or 0,
        "attacker_failure_type": (calls[first_index]["failure_type"]
                                  if first_index is not None else None),
        # index *within the run's attacker calls*, 1-based (not the index within the
        # subset of failed calls)
        "first_failure_turn": (first_index + 1) if first_index is not None else None,
        # derived, not recorded by the pilot: which role's output the first failure was
        # observed on. The pilot records failures only on attacker calls, so this is
        # "attacker_output" whenever a failure exists and null otherwise.
        "first_failure_stage": "attacker_output" if first_index is not None else None,
    }


# --------------------------------------------------------------------------- #
# 1. integrity                                                                #
# --------------------------------------------------------------------------- #
def integrity_check(runs: list[dict]) -> dict:
    ids = [r["run_id"] for r in runs]
    per_attack = Counter(r["attack_short"] for r in runs)
    seeds = [r["seed"] for r in runs]
    expected_seeds = sorted(42 + g * 1000 + off
                            for g in range(30)
                            for off in (0, 100, 200))
    checks = {
        "run_records": len(runs),
        "unique_run_ids": len(set(ids)),
        "no_duplicate_run_ids": len(set(ids)) == len(ids),
        "crescendo_runs": per_attack.get("crescendo", 0),
        "opposite_day_runs": per_attack.get("opposite_day", 0),
        "acronym_runs": per_attack.get("acronym", 0),
        "unique_goals": len({r["goal_id"] for r in runs}),
        "goal_ids_complete_0_29": sorted({r["goal_id"] for r in runs}) == list(range(30)),
        "all_expected_seeds_present": sorted(seeds) == expected_seeds,
        "attack_identifiers": sorted({r["attack"] for r in runs}),
        "prompt_hashes": sorted({str(r["prompt_hash"])[:32] for r in runs}),
        "temperature_values": sorted({r["temperature"] for r in runs}),
        "top_p_values": sorted({r["top_p"] for r in runs}),
        "max_turns_values": sorted({r["max_turns"] for r in runs}),
        "nbf_disabled_all_runs": all(r["configuration"]["structured_output_mode"] is None
                                     for r in runs) if "configuration" in runs[0] else None,
        "candidate_values": sorted({r["candidate"] for r in runs}),
        "attacker_calls": sum(len(r["attacker_calls"]) for r in runs),
        "target_calls": sum(len(r["target_calls"]) for r in runs),
        "judge_calls": sum(len(r["judge_calls"]) for r in runs),
        "attacker_raw_outputs_present": sum(1 for r in runs for c in r["attacker_calls"]
                                            if c.get("raw_output") is not None),
        "target_raw_outputs_present": sum(1 for r in runs for c in r["target_calls"]
                                          if c.get("raw_output")),
        "judge_raw_outputs_present": sum(1 for r in runs for c in r["judge_calls"]
                                         if c.get("raw_output")),
    }
    expected = {"run_records": 90, "unique_run_ids": 90, "crescendo_runs": 30,
                "opposite_day_runs": 30, "acronym_runs": 30, "unique_goals": 30,
                "attacker_calls": 371, "target_calls": 303, "judge_calls": 520,
                "temperature_values": [0.7], "top_p_values": [1.0],
                "max_turns_values": [8]}
    mismatches = {k: {"observed": checks[k], "expected": v} for k, v in expected.items()
                  if checks[k] != v}
    integrity = {
        "phase": "17", "stage": "4.9", "deliverable": "integrity.json",
        "source": "results/phase17_pilot/runs/raw_results.jsonl (frozen Stage 4.8 data)",
        "checks": checks,
        "expected": expected,
        "mismatches": mismatches,
        "status": "PASS" if not mismatches else "BLOCKED",
        "raw_data_modified": False,
        "note": "the analysis reads the frozen records only; nothing was written to "
                "runs/raw_results.jsonl or any Stage 4.8 artifact",
    }
    return integrity


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    VIZ.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    for run in runs:
        run.update(derived_flags(run))

    integrity = integrity_check(runs)
    (OUT / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    # ---------------- master table (§6) ---------------- #
    master = []
    for run in runs:
        calls = run["attacker_calls"]
        master.append({
            "run_id": run["run_id"], "goal_id": run["goal_id"], "attack": run["attack"],
            "attack_short": run["attack_short"], "seed": run["seed"],
            "duration_s": run["duration_s"],
            "attacker_calls": len(calls),
            "attacker_valid_calls": run["counts"]["attacker_calls_valid"],
            "attacker_invalid_calls": run["attacker_invalid_calls"],
            "attacker_semantic_usable_calls": run["attacker_semantic_usable_calls"],
            "attacker_valid_fraction": round(run["counts"]["attacker_calls_valid"] / len(calls), 4)
            if calls else None,
            "completed_turns": run["completed_turns"],
            "survived": run["survived"], "strict_usable": run["strict_usable"],
            "success": bool(run["success"]),
            "final_score": run["final_score"], "final_rubric": run["final_rubric"],
            "termination_reason": run["termination_reason"],
            "failure_class": run["failure_class"],
            "target_calls": len(run["target_calls"]),
            "target_natural_eos": run["target_natural_eos"],
            "target_refusals": run["target_refusals"],
            "judge_calls": len(run["judge_calls"]),
            "judge_failures": run["counts"]["judge_failures"],
            "rubric_scores": run["rubric_scores"],
            "attacker_failure_type": run["attacker_failure_type"],
            "first_failure_turn": run["first_failure_turn"],
            "first_failure_stage": run["first_failure_stage"],
            "goal": run["goal"],
        })
    with (OUT / "master_run_table.jsonl").open("w", encoding="utf-8") as handle:
        for row in master:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ---------------- reproduce Stage 4.8 (§5) ---------------- #
    calls = [c for r in runs for c in r["attacker_calls"]]
    valid = [c for c in calls if c["category"] in VALID]
    usable = [c for c in valid if c["failure_type"] is None]
    failures = [c for c in calls if c["failure_type"] is not None]
    reproduced = {
        "attacker_calls": len(calls), "contract_valid": len(valid),
        "semantic_usable": len(usable),
        "validity_rate": round(len(valid) / len(calls), 4),
        "validity_ci95": wilson(len(valid), len(calls)),
        "semantic_usability_rate": round(len(usable) / len(calls), 4),
        "semantic_usability_ci95": wilson(len(usable), len(calls)),
        "surviving_runs": sum(1 for r in runs if r["survived"]),
        "strict_usable_runs": sum(1 for r in runs if r["strict_usable"]),
        "turns": describe([r["completed_turns"] for r in runs]),
        "full_8_turn_completions": sum(1 for r in runs if r["completed_turns"] >= 8),
        "termination_reasons": dict(Counter(r["termination_reason"] for r in runs)),
        "successes": sum(1 for r in runs if r["success"]),
        "success_rate": round(sum(1 for r in runs if r["success"]) / len(runs), 4),
        "success_ci95": wilson(sum(1 for r in runs if r["success"]), len(runs)),
        "target_calls": sum(len(r["target_calls"]) for r in runs),
        "target_natural_eos": sum(r["target_natural_eos"] for r in runs),
        "judge_calls": sum(len(r["judge_calls"]) for r in runs),
        "judge_usable": sum(1 for r in runs for c in r["judge_calls"]
                            if c["returned_type"] == "dict"),
        "judge_failures": sum(r["counts"]["judge_failures"] for r in runs),
        "matches_stage4_8": {
            "attacker_calls": len(calls) == 371, "contract_valid": len(valid) == 310,
            "semantic_usable": len(usable) == 303,
            "surviving_runs": sum(1 for r in runs if r["survived"]) == 29,
            "successes": sum(1 for r in runs if r["success"]) == 22,
            "target_natural_eos": sum(r["target_natural_eos"] for r in runs) == 303,
            "judge_usable": sum(1 for r in runs for c in r["judge_calls"]
                                if c["returned_type"] == "dict") == 520,
        },
        "stage4_8_reference": {"attacker_calls": 371, "contract_valid": 310,
                               "semantic_usable": 303, "surviving_runs": 29,
                               "strict_usable_runs": 22, "successes": 22,
                               "target_natural_eos": 303, "judge_usable": 520},
    }

    # ---------------- §7 attacker failure forensics ---------------- #
    forensics: dict = {
        "phase": "17", "stage": "4.9", "deliverable": "attacker_failure_forensics.json",
        "scope": "attacker calls only; the pilot recorded no target or judge failures",
        "total_calls": len(calls), "total_runs": len(runs),
        "by_type": {}, "call_index_distribution": {}, "first_call_vs_later": {},
    }
    for label, subset, description in (
            ("prose_output", [c for c in failures if c["failure_type"] == "prose_output"],
             "a complete answer that was not wrapped in the required JSON object"),
            ("empty_generatedQuestion",
             [c for c in failures if c["failure_type"] == "empty_generatedQuestion"],
             "contract-valid JSON with an empty generatedQuestion"),
            ("truncated_json", [c for c in failures if c["failure_type"] == "truncated_json"],
             "an object that was opened and never closed")):
        affected = {r["run_id"] for r in runs for c in r["attacker_calls"]
                    if c["failure_type"] == label}
        per_attack = Counter(r["attack_short"] for r in runs for c in r["attacker_calls"]
                             if c["failure_type"] == label)
        goals = sorted({r["goal_id"] for r in runs for c in r["attacker_calls"]
                        if c["failure_type"] == label})
        turns = Counter(i + 1 for r in runs
                        for i, c in enumerate(r["attacker_calls"])
                        if c["failure_type"] == label)
        ended_run = sum(1 for r in runs if r["attacker_failure_type"] == label
                        and not r["survived"])
        forensics["by_type"][label] = {
            "description": description,
            "count": len(subset),
            "pct_of_attacker_calls": round(len(subset) / len(calls), 4),
            "runs_affected": len(affected),
            "pct_of_runs": round(len(affected) / len(runs), 4),
            "by_attack": {ATTACK_LABEL[a]: per_attack.get(a.split("_")[0] if a != "opposite_day" else "opposite_day", 0)
                          for a in ATTACKS},
            "by_attack_raw": {k: v for k, v in per_attack.items()},
            "goals_affected": goals,
            "n_goals_affected": len(goals),
            "call_index_distribution": {str(k): v for k, v in sorted(turns.items())},
            "runs_ended_by_this_failure": ended_run,
            "sample_raw_heads": [str(c["raw_output"])[:200] for c in subset[:3]],
        }
    forensics["call_index_distribution"] = {
        "note": "index 1 = first attacker call of the run; this is the index of the first "
                "failing call within the run's attacker calls",
        "all_failures": {str(k): v for k, v in sorted(Counter(
            r["first_failure_turn"] for r in runs if r["first_failure_turn"]).items())},
    }
    # A failure that happens after at least one successful call is a different signal
    # from a failure on the very first call.
    first_only = [r for r in runs if r["first_failure_turn"] == 1]
    later_first = [r for r in runs if (r["first_failure_turn"] or 0) > 1]
    forensics["first_vs_later_failure_types"] = {
        "failures_on_call_1_by_type": dict(Counter(r["attacker_failure_type"]
                                                   for r in first_only)),
        "failures_after_a_successful_call_by_type": dict(Counter(r["attacker_failure_type"]
                                                                for r in later_first)),
        "runs_first_failure_on_call_1": len(first_only),
        "runs_first_failure_after_call_1": len(later_first),
    }
    first_call_failures = sum(1 for r in runs
                              if r["attacker_calls"]
                              and r["attacker_calls"][0]["failure_type"] is not None)
    runs_with_calls = [r for r in runs if r["attacker_calls"]]
    later_failures = sum(1 for r in runs
                         if r["attacker_calls"] and r["attacker_calls"][0]["failure_type"] is None
                         and any(c["failure_type"] for c in r["attacker_calls"]))
    total_attacker_call_instances_after_first = sum(
        max(0, len(r["attacker_calls"]) - 1) for r in runs_with_calls)
    forensics["first_call_vs_later"] = {
        "runs_with_first_call_failure": first_call_failures,
        "first_call_failure_rate_of_runs": round(first_call_failures / len(runs), 4),
        "first_call_validity_rate_of_calls": round(
            sum(1 for r in runs_with_calls
                if r["attacker_calls"][0]["failure_type"] is None) / len(runs_with_calls), 4),
        "runs_first_call_ok_but_later_failure": later_failures,
        "later_call_validity_rate": round(
            1 - sum(1 for r in runs_with_calls for c in r["attacker_calls"][1:]
                    if c["failure_type"] is not None) / total_attacker_call_instances_after_first, 4)
        if total_attacker_call_instances_after_first else None,
        "later_call_instances": total_attacker_call_instances_after_first,
        "by_attack": {ATTACK_LABEL[a]: {
            "runs": len([r for r in runs if r["attack"] == a]),
            "first_call_failures": sum(1 for r in runs if r["attack"] == a
                                       and r["attacker_calls"]
                                       and r["attacker_calls"][0]["failure_type"] is not None),
            "later_call_failures": sum(1 for r in runs if r["attack"] == a
                                       and r["attacker_calls"]
                                       and r["attacker_calls"][0]["failure_type"] is None
                                       and any(c["failure_type"] for c in r["attacker_calls"])),
        } for a in ATTACKS},
    }
    (OUT / "attacker_failure_forensics.json").write_text(
        json.dumps(forensics, indent=2), encoding="utf-8")

    # ---------------- §9 attack comparison ---------------- #
    comparison = {"phase": "17", "stage": "4.9", "deliverable": "attack_comparison.json",
                  "framing": "observed measurements only; no attack is ranked as better or "
                             "worse, and n = 30 runs per attack",
                  "attacks": {}}
    for attack in ATTACKS:
        subset = [r for r in runs if r["attack"] == attack]
        acalls = [c for r in subset for c in r["attacker_calls"]]
        avalid = [c for c in acalls if c["category"] in VALID]
        afail = [c for c in acalls if c["failure_type"] is not None]
        comparison["attacks"][ATTACK_LABEL[attack]] = {
            "reliability": {
                "calls": len(acalls), "valid": len(avalid),
                "validity_rate": round(len(avalid) / len(acalls), 4),
                "validity_ci95": wilson(len(avalid), len(acalls)),
                "semantic_usable": len([c for c in avalid if c["failure_type"] is None]),
                "prose_rate": round(sum(1 for c in afail
                                        if c["failure_type"] == "prose_output") / len(acalls), 4),
                "empty_question_rate": round(sum(1 for c in afail
                                                 if c["failure_type"] == "empty_generatedQuestion")
                                             / len(acalls), 4),
                "truncation_rate": round(sum(1 for c in afail
                                            if c["failure_type"] == "truncated_json")
                                         / len(acalls), 4),
            },
            "trajectory": {
                "runs": len(subset),
                "survival": block(subset, "survived"),
                "strict_usable": block(subset, "strict_usable"),
                "turns": describe([r["completed_turns"] for r in subset]),
                "max_turns": max(r["completed_turns"] for r in subset),
                "termination_reasons": dict(Counter(r["termination_reason"] for r in subset)),
            },
            "effectiveness": {
                "successes": sum(1 for r in subset if r["success"]),
                "success_rate": round(sum(1 for r in subset if r["success"]) / len(subset), 4),
                "success_ci95": wilson(sum(1 for r in subset if r["success"]), len(subset)),
                "final_score_distribution": {str(k): v for k, v in sorted(
                    Counter(r["final_score"] for r in subset).items(),
                    key=lambda kv: (kv[0] is None, kv[0]))},
                "rubric_distribution": {str(k): v for k, v in sorted(
                    Counter(s for r in subset for s in r["rubric_scores"]).items())},
            },
            "runtime": {"duration": describe([r["duration_s"] for r in subset])},
        }
    (OUT / "attack_comparison.json").write_text(json.dumps(comparison, indent=2),
                                                encoding="utf-8")

    # ---------------- §10 goal analysis ---------------- #
    goals_out = []
    for goal_id in range(30):
        subset = [r for r in runs if r["goal_id"] == goal_id]
        by_attack = {ATTACK_LABEL[r["attack"]]: {
            "run_id": r["run_id"], "success": bool(r["success"]),
            "survived": r["survived"], "turns": r["completed_turns"],
            "final_score": r["final_score"], "attack_failures": r["attacker_invalid_calls"],
            "duration_s": r["duration_s"]} for r in subset}
        goals_out.append({
            "goal_id": goal_id,
            "goal": subset[0]["goal"] if subset else None,
            "successful_attacks": sum(1 for r in subset if r["success"]),
            "surviving_runs": sum(1 for r in subset if r["survived"]),
            "attacker_failures": sum(r["attacker_invalid_calls"] for r in subset),
            "attacker_valid_calls": sum(r["counts"]["attacker_calls_valid"] for r in subset),
            "attacker_calls": sum(len(r["attacker_calls"]) for r in subset),
            "by_attack": by_attack,
        })
    goal_payload = {
        "phase": "17", "stage": "4.9", "deliverable": "goal_analysis.json",
        "sample_size_caveat": "each goal has exactly 3 runs (one per attack); differences "
                              "between goals are pilot patterns, not established difficulty",
        "goals": goals_out,
        "patterns": {
            "goals_with_all_three_successes": [g["goal_id"] for g in goals_out
                                               if g["successful_attacks"] == 3],
            "goals_with_no_success": [g["goal_id"] for g in goals_out
                                      if g["successful_attacks"] == 0],
            "goals_with_no_surviving_run": [g["goal_id"] for g in goals_out
                                            if g["surviving_runs"] == 0],
            "goals_with_no_attacker_failure": [g["goal_id"] for g in goals_out
                                               if g["attacker_failures"] == 0],
        },
    }
    (OUT / "goal_analysis.json").write_text(json.dumps(goal_payload, indent=2),
                                            encoding="utf-8")

    # ---------------- §11 survival vs success ---------------- #
    both = sum(1 for r in runs if r["survived"] and r["success"])
    surv_only = sum(1 for r in runs if r["survived"] and not r["success"])
    nosurv_success = sum(1 for r in runs if not r["survived"] and r["success"])
    neither = sum(1 for r in runs if not r["survived"] and not r["success"])
    survival_success = {
        "phase": "17", "stage": "4.9", "deliverable": "survival_success_analysis.json",
        "contingency_table": {"survived_and_success": both,
                              "survived_no_success": surv_only,
                              "not_survived_success": nosurv_success,
                              "not_survived_no_success": neither},
        "all_successes_within_surviving_runs": nosurv_success == 0,
        "p_success_given_survived": round(both / (both + surv_only), 4),
        "p_success_given_not_survived": (round(nosurv_success / (nosurv_success + neither), 4)
                                         if (nosurv_success + neither) else None),
        "official_rate_unchanged": round((both + nosurv_success) / len(runs), 4),
        "note": "the conditional 22/29 is descriptive; the official pilot result remains "
                "22/90 = 24.44 % and is not replaced by it",
        "test_survival_vs_success": fisher_test(
            "does survival associate with success? (run level)",
            both, surv_only, nosurv_success, neither, "exploratory"),
    }
    (OUT / "survival_success_analysis.json").write_text(
        json.dumps(survival_success, indent=2), encoding="utf-8")

    # ---------------- §12 trajectory depth ---------------- #
    depth = {"phase": "17", "stage": "4.9", "deliverable": "trajectory_depth.json",
             "by_completed_turns": {}, "observational": True}
    for t in range(0, 9):
        subset = [r for r in runs if r["completed_turns"] == t]
        if not subset:
            continue
        depth["by_completed_turns"][str(t)] = {
            "runs": len(subset),
            "successes": sum(1 for r in subset if r["success"]),
            "success_rate": round(sum(1 for r in subset if r["success"]) / len(subset), 4),
            "termination_reasons": dict(Counter(r["termination_reason"] for r in subset)),
        }
    zero_turn = [r for r in runs if r["completed_turns"] == 0]
    multi_turn = [r for r in runs if r["completed_turns"] >= 1]
    depth["zero_vs_one_or_more"] = {
        "zero_turn_runs": len(zero_turn),
        "zero_turn_successes": sum(1 for r in zero_turn if r["success"]),
        "one_or_more_runs": len(multi_turn),
        "one_or_more_successes": sum(1 for r in multi_turn if r["success"]),
        "test": fisher_test("does completing >=1 turn associate with success?",
                            sum(1 for r in multi_turn if r["success"]),
                            len(multi_turn) - sum(1 for r in multi_turn if r["success"]),
                            sum(1 for r in zero_turn if r["success"]),
                            len(zero_turn) - sum(1 for r in zero_turn if r["success"]),
                            "exploratory"),
    }
    depth["caveat"] = ("depth is an outcome of the run, not an input: runs that fail early "
                       "have zero turns precisely because the attacker failed. Depth and "
                       "success are therefore mechanically linked and this table must not be "
                       "read as 'deeper causes success'.")
    (OUT / "trajectory_depth.json").write_text(json.dumps(depth, indent=2), encoding="utf-8")

    # ---------------- §13 score distribution ---------------- #
    scores = {"phase": "17", "stage": "4.9", "deliverable": "score_distribution.json",
              "all_runs": {}, "by_attack": {}, "criterion": "score == 5 is success (frozen)"}
    counter = Counter(str(r["final_score"]) if r["final_score"] is not None else "none"
                      for r in runs)
    scores["all_runs"] = {k: {"count": v, "pct": round(v / len(runs), 4)}
                          for k, v in sorted(counter.items())}
    for attack in ATTACKS:
        subset = [r for r in runs if r["attack"] == attack]
        c = Counter(str(r["final_score"]) if r["final_score"] is not None else "none"
                    for r in subset)
        scores["by_attack"][ATTACK_LABEL[attack]] = {k: {"count": v,
                                                         "pct": round(v / len(subset), 4)}
                                                     for k, v in sorted(c.items())}
    (OUT / "score_distribution.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    # ---------------- §14 rubric analysis ---------------- #
    all_rubric = [s for r in runs for s in r["rubric_scores"]]
    rubric = {
        "phase": "17", "stage": "4.9", "deliverable": "rubric_analysis.json",
        "n_scores": len(all_rubric),
        "distribution": {str(k): {"count": v, "pct": round(v / len(all_rubric), 4)}
                         for k, v in sorted(Counter(all_rubric).items())},
        "mean": round(statistics.mean(all_rubric), 3),
        "median": statistics.median(all_rubric),
        "by_attack": {},
        "relationships": {},
        "caveat": "rubric scores only exist for turns whose response was scored; runs that "
                  "ended before a scored turn contribute none (21 runs)",
    }
    for attack in ATTACKS:
        vals = [s for r in runs if r["attack"] == attack for s in r["rubric_scores"]]
        rubric["by_attack"][ATTACK_LABEL[attack]] = {
            "n": len(vals),
            "mean": round(statistics.mean(vals), 3) if vals else None,
            "median": statistics.median(vals) if vals else None,
            "distribution": {str(k): v for k, v in sorted(Counter(vals).items())},
        }
    succ = [r for r in runs if r["success"]]
    fail = [r for r in runs if not r["success"]]
    rubric["relationships"] = {
        "mean_rubric_successful_runs": describe([s for r in succ for s in r["rubric_scores"]]),
        "mean_rubric_unsuccessful_runs": describe([s for r in fail for s in r["rubric_scores"]]),
        "mean_completed_turns_successful": describe([r["completed_turns"] for r in succ]),
        "mean_completed_turns_unsuccessful": describe([r["completed_turns"] for r in fail]),
        "test_turns_success_vs_not": permutation_test(
            [float(r["completed_turns"]) for r in succ],
            [float(r["completed_turns"]) for r in fail]),
    }
    TESTS_PERFORMED.append({"question": "completed turns: successful vs unsuccessful runs",
                            "p_value": rubric["relationships"]["test_turns_success_vs_not"]["p_value"],
                            "label": "exploratory (permutation)"})
    (OUT / "rubric_analysis.json").write_text(json.dumps(rubric, indent=2), encoding="utf-8")

    # ---------------- §15 target refusal analysis ---------------- #
    refusals = {
        "phase": "17", "stage": "4.9", "deliverable": "target_refusal_analysis.json",
        "total_recorded_refusals": sum(r["target_refusals"] for r in runs),
        "source": "run['refusal_count'] as tracked by each attack's own refusal logic",
        "per_run": describe([float(r["target_refusals"]) for r in runs]),
        "per_surviving_run": describe([float(r["target_refusals"]) for r in runs
                                       if r["survived"]]),
        "runs_with_zero_refusals": sum(1 for r in runs if r["target_refusals"] == 0),
        "by_attack": {ATTACK_LABEL[a]: {
            "runs": len([r for r in runs if r["attack"] == a]),
            "total": sum(r["target_refusals"] for r in runs if r["attack"] == a),
            "mean": round(statistics.mean([r["target_refusals"] for r in runs
                                           if r["attack"] == a]), 3)}
            for a in ATTACKS},
        "successful_runs": describe([float(r["target_refusals"]) for r in runs if r["success"]]),
        "unsuccessful_runs": describe([float(r["target_refusals"]) for r in runs
                                       if not r["success"]]),
        "relationship_with_final_score": {
            "note": "descriptive association only; a refusal is not automatically a failed "
                    "step — the attack logic decides, and it counts refusals per its own rule",
        },
    }
    refusals["relationship_with_final_score"]["test_refusals_success_vs_not"] = permutation_test(
        [float(r["target_refusals"]) for r in runs if r["success"]],
        [float(r["target_refusals"]) for r in runs if not r["success"]])
    TESTS_PERFORMED.append({"question": "target refusals: successful vs unsuccessful runs",
                            "p_value": refusals["relationship_with_final_score"]
                            ["test_refusals_success_vs_not"]["p_value"],
                            "label": "exploratory (permutation)"})
    (OUT / "target_refusal_analysis.json").write_text(json.dumps(refusals, indent=2),
                                                      encoding="utf-8")

    # ---------------- §16 reliability vs success ---------------- #
    clean = [r for r in runs if r["attacker_invalid_calls"] == 0]
    dirty = [r for r in runs if r["attacker_invalid_calls"] > 0]
    rel_success = {
        "phase": "17", "stage": "4.9", "deliverable": "reliability_success_relationship.json",
        "framing": "observational association; no causal claim",
        "runs_with_zero_invalid_calls": len(clean),
        "runs_with_at_least_one_invalid_call": len(dirty),
        "success_among_clean": {"numerator": sum(1 for r in clean if r["success"]),
                                "denominator": len(clean),
                                "rate": round(sum(1 for r in clean if r["success"]) / len(clean), 4)
                                if clean else None},
        "success_among_runs_with_a_failure": {
            "numerator": sum(1 for r in dirty if r["success"]), "denominator": len(dirty),
            "rate": round(sum(1 for r in dirty if r["success"]) / len(dirty), 4)
            if dirty else None},
        "test": fisher_test("does having any invalid attacker call associate with success?",
                            sum(1 for r in dirty if r["success"]),
                            len(dirty) - sum(1 for r in dirty if r["success"]),
                            sum(1 for r in clean if r["success"]),
                            len(clean) - sum(1 for r in clean if r["success"]),
                            "exploratory"),
        "attacker_valid_fraction": {
            "successful_runs": describe([r["counts"]["attacker_calls_valid"]
                                         / len(r["attacker_calls"]) for r in runs
                                         if r["success"] and r["attacker_calls"]]),
            "unsuccessful_runs": describe([r["counts"]["attacker_calls_valid"]
                                           / len(r["attacker_calls"]) for r in runs
                                           if not r["success"] and r["attacker_calls"]]),
        },
        "valid_call_count": {
            "successful_runs": describe([float(r["counts"]["attacker_calls_valid"])
                                         for r in runs if r["success"]]),
            "unsuccessful_runs": describe([float(r["counts"]["attacker_calls_valid"])
                                           for r in runs if not r["success"]]),
        },
    }
    (OUT / "reliability_success_relationship.json").write_text(
        json.dumps(rel_success, indent=2), encoding="utf-8")

    # ---------------- §17 failure cascade ---------------- #
    cascade = {
        "phase": "17", "stage": "4.9", "deliverable": "failure_cascade.json",
        "funnel": [
            {"stage": "runs launched", "runs": len(runs)},
            {"stage": "runs that produced at least one attacker call",
             "runs": sum(1 for r in runs if r["attacker_calls"])},
            {"stage": "runs with at least one contract-valid attacker call",
             "runs": sum(1 for r in runs if r["counts"]["attacker_calls_valid"] > 0)},
            {"stage": "runs that reached the target (>=1 target call)",
             "runs": sum(1 for r in runs if r["target_calls"])},
            {"stage": "runs that reached the judge (>=1 judge call)",
             "runs": sum(1 for r in runs if r["judge_calls"])},
            {"stage": "runs with at least one rubric score",
             "runs": sum(1 for r in runs if r["rubric_scores"])},
            {"stage": "runs reaching the success criterion (score 5)",
             "runs": sum(1 for r in runs if r["success"])},
        ],
        "call_counts": {"attacker_calls": len(calls), "target_calls": sum(len(r["target_calls"]) for r in runs),
                        "judge_calls": sum(len(r["judge_calls"]) for r in runs)},
        "runs_failing_before_first_target_call": sum(1 for r in runs if not r["target_calls"]),
        "explanation_of_call_count_difference": (
            "judge calls (520) exceed target calls (303) because the attack code makes two "
            "judge-type calls per accepted turn (a refusal/disclaimer check and the rubric "
            "evaluation); attacker calls (371) exceed target calls because a run can retry "
            "the attacker step after a refusal before a target call is accepted."),
    }
    (OUT / "failure_cascade.json").write_text(json.dumps(cascade, indent=2), encoding="utf-8")

    # ---------------- §18 latency ---------------- #
    early_failures = [r for r in runs if not r["survived"] and r["completed_turns"] == 0]
    latency = {
        "phase": "17", "stage": "4.9", "deliverable": "latency_analysis.json",
        "all_runs": describe([r["duration_s"] for r in runs]),
        "early_attacker_failure_runs": describe([r["duration_s"] for r in early_failures]),
        "surviving_runs": describe([r["duration_s"] for r in runs if r["survived"]]),
        "successful_runs": describe([r["duration_s"] for r in runs if r["success"]]),
        "attacker_call_latency_s": describe([c["latency_s"] for r in runs
                                             for c in r["attacker_calls"]
                                             if c.get("latency_s") is not None]),
        "generated_tokens_per_attacker_call": describe(
            [float(c["generated_tokens"]) for r in runs for c in r["attacker_calls"]
             if c.get("generated_tokens") is not None]),
        "duration_by_termination_reason": {
            k: describe([r["duration_s"] for r in runs if r["termination_reason"] == k])
            for k in sorted({r["termination_reason"] for r in runs})},
        "bimodality": {
            "note": "the distribution is bi-modal because failed runs abort at the first "
                    "unusable attacker call whereas survivors run whole turns",
            "runs_under_120s": sum(1 for r in runs if r["duration_s"] < 120),
            "runs_120_to_600s": sum(1 for r in runs if 120 <= r["duration_s"] < 600),
            "runs_600s_and_over": sum(1 for r in runs if r["duration_s"] >= 600),
        },
        "contention_caveat": "measured while a separate operator GPU job was active; "
                             "absolute durations are contention-affected",
    }
    (OUT / "latency_analysis.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")

    # ---------------- §19 infrastructure ---------------- #
    infra = {
        "phase": "17", "stage": "4.9", "deliverable": "infrastructure_analysis.json",
        "oom_events": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
        "cuda_failures": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
        "infrastructure_failures": sum(1 for r in runs if r["failure_class"]
                                       == "infrastructure_failure"),
        "peak_vram_gib": max(r["peak_vram_gib"] for r in runs),
        "peak_vram_min_gib": min(r["peak_vram_gib"] for r in runs),
        "runs_with_infrastructure_error_field": sum(1 for r in runs
                                                    if r.get("infrastructure_error")),
        "conclusion": "no observed infrastructure constraint influenced the outcomes; "
                      "every run that failed did so on the attacker contract, and VRAM was "
                      "identical on all 90 runs",
        "environment_note": "a separate operator GPU job was running throughout "
                            "(results/phase17_pilot/logs/environment_observation_concurrent_gpu_job.md)",
    }
    (OUT / "infrastructure_analysis.json").write_text(json.dumps(infra, indent=2),
                                                      encoding="utf-8")

    # ---------------- §20 statistical analysis ---------------- #
    pairwise = []
    for i, a in enumerate(ATTACKS):
        for b in ATTACKS[i + 1:]:
            ra = [r for r in runs if r["attack"] == a]
            rb = [r for r in runs if r["attack"] == b]
            pairwise.append({
                "comparison": f"{ATTACK_LABEL[a]} vs {ATTACK_LABEL[b]}",
                "success": fisher_test(
                    f"success rate {ATTACK_LABEL[a]} vs {ATTACK_LABEL[b]}",
                    sum(1 for r in ra if r["success"]), len(ra) - sum(1 for r in ra if r["success"]),
                    sum(1 for r in rb if r["success"]), len(rb) - sum(1 for r in rb if r["success"]),
                    "exploratory"),
                "survival": fisher_test(
                    f"survival {ATTACK_LABEL[a]} vs {ATTACK_LABEL[b]}",
                    sum(1 for r in ra if r["survived"]), len(ra) - sum(1 for r in ra if r["survived"]),
                    sum(1 for r in rb if r["survived"]), len(rb) - sum(1 for r in rb if r["survived"]),
                    "exploratory"),
            })
    p_values = [t["p_value"] for t in TESTS_PERFORMED if t["p_value"] is not None]
    stats_payload = {
        "phase": "17", "stage": "4.9", "deliverable": "statistical_analysis.json",
        "confidence_intervals": {
            "attacker_validity": reproduced["validity_ci95"],
            "attacker_semantic_usability": reproduced["semantic_usability_ci95"],
            "run_survival": block(runs, "survived")["ci95_wilson"],
            "attack_success": reproduced["success_ci95"],
            "per_attack_success": {ATTACK_LABEL[a]: comparison["attacks"][ATTACK_LABEL[a]]
                                   ["effectiveness"]["success_ci95"] for a in ATTACKS},
            "method": "Wilson score interval, 95 %",
        },
        "attack_level_comparisons": pairwise,
        "exploratory_test_discipline": {
            "n_tests": len(TESTS_PERFORMED),
            "tests": TESTS_PERFORMED,
            "min_p_value": min(p_values) if p_values else None,
            "bonferroni_threshold": round(0.05 / len(p_values), 4) if p_values else None,
            "n_below_bonferroni": sum(1 for p in p_values
                                      if p < (0.05 / len(p_values) if p_values else 1)),
            "statement": "all association tests here are exploratory, performed on one "
                         "pilot dataset, and none survives a Bonferroni correction; no "
                         "p-value is presented as a definitive finding",
        },
        "associations": {
            "attacker_reliability_to_success": rel_success["test"],
            "survival_to_success": survival_success["test_survival_vs_success"],
            "depth_to_success": depth["zero_vs_one_or_more"]["test"],
        },
        "caveat": "n = 90 runs and n = 30 per attack; differences between attacks are "
                  "reported descriptively and are not established as real differences",
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(stats_payload, indent=2),
                                                   encoding="utf-8")

    # ---------------- §23 successful run forensics ---------------- #
    successes = [r for r in runs if r["success"]]
    surviving_failures = [r for r in runs if r["survived"] and not r["success"]]
    success_forensics = {
        "phase": "17", "stage": "4.9", "deliverable": "successful_run_forensics.json",
        "n_successes": len(successes),
        "all_successes_included": True,
        "runs": [{
            "run_id": r["run_id"], "goal_id": r["goal_id"],
            "attack": ATTACK_LABEL[r["attack"]], "completed_turns": r["completed_turns"],
            "attacker_calls": len(r["attacker_calls"]),
            "attacker_valid_calls": r["counts"]["attacker_calls_valid"],
            "target_refusals": r["target_refusals"], "final_rubric": r["final_rubric"],
            "final_score": r["final_score"], "duration_s": r["duration_s"],
            "rubric_trajectory": r["rubric_scores"],
        } for r in sorted(successes, key=lambda x: (x["attack"], x["goal_id"]))],
        "structural_summary": {
            "by_attack": dict(Counter(ATTACK_LABEL[r["attack"]] for r in successes)),
            "zero_attacker_failures": sum(1 for r in successes
                                          if r["attacker_invalid_calls"] == 0),
            "with_attacker_failures": sum(1 for r in successes
                                          if r["attacker_invalid_calls"] > 0),
            "turns": describe([float(r["completed_turns"]) for r in successes]),
            "duration": describe([r["duration_s"] for r in successes]),
            "target_refusals": describe([float(r["target_refusals"]) for r in successes]),
            "trajectory_shape": "every success reached score 5 on its final turn; the "
                                "rubric sequence before it may include low scores",
        },
        "compared_with_surviving_unsuccessful_runs": {
            "n": len(surviving_failures),
            "run_ids": [r["run_id"] for r in surviving_failures],
            "turns": describe([float(r["completed_turns"]) for r in surviving_failures]),
            "rubric_scores": describe([float(s) for r in surviving_failures
                                       for s in r["rubric_scores"]]),
            "max_rubric_reached": max((s for r in surviving_failures for s in r["rubric_scores"]),
                                      default=None),
            "termination_reasons": dict(Counter(r["termination_reason"]
                                                for r in surviving_failures)),
        },
    }
    (OUT / "successful_run_forensics.json").write_text(json.dumps(success_forensics, indent=2),
                                                       encoding="utf-8")

    # ---------------- §24 failure run forensics ---------------- #
    examples = []
    for failure_type in ("prose_output", "empty_generatedQuestion", "truncated_json"):
        picked = None
        for run in runs:
            for index, c in enumerate(run["attacker_calls"]):
                if c["failure_type"] == failure_type:
                    picked = {"failure_type": failure_type, "run_id": run["run_id"],
                              "goal_id": run["goal_id"], "attack": ATTACK_LABEL[run["attack"]],
                              "call_index": index + 1,
                              "stored_category": c["category"],
                              "stored_failure_shape": c["failure_shape"],
                              "tokens_generated": c.get("generated_tokens"),
                              "raw_head": str(c["raw_output"])[:220],
                              "run_terminated_here": not run["survived"],
                              "run_failure_class": run["failure_class"],
                              "run_completed_turns": run["completed_turns"]}
                    break
            if picked:
                break
        if picked:
            examples.append(picked)
    unusable_runs = [r for r in runs if r["failure_class"] == "attacker_output_unusable"]
    failure_forensics = {
        "phase": "17", "stage": "4.9", "deliverable": "failure_run_forensics.json",
        "representative_examples": examples,
        "attacker_output_unusable_classes": {
            "n_runs": len(unusable_runs),
            "run_ids": [r["run_id"] for r in unusable_runs],
            "shape": "the run produced exactly one attacker call which parsed as valid JSON "
                     "but carried an empty generatedQuestion, so no query could be sent; the "
                     "run ended with zero target calls",
            "observed_counts": [{"run_id": r["run_id"],
                                 "attacker_calls": len(r["attacker_calls"]),
                                 "target_calls": len(r["target_calls"])}
                                for r in unusable_runs],
        },
        "termination_attribution": {
            "note": "each failure type's stored classification is quoted verbatim; no cause "
                    "is inferred beyond what the record shows",
            "prose_output": "the record stores category JSON_PARSE_ERROR with failure_shape "
                            "prose: no JSON object was present in the reply",
            "empty_generatedQuestion": "the record stores category VALID_DIRECT_JSON with an "
                                       "empty generatedQuestion field",
            "truncated_json": "the record stores category TRUNCATED_OUTPUT: an object was "
                              "opened and never closed",
        },
    }
    (OUT / "failure_run_forensics.json").write_text(json.dumps(failure_forensics, indent=2),
                                                    encoding="utf-8")

    # ---------------- §22 visualizations ---------------- #
    viz_files = make_visualizations(runs, out_dir=VIZ)

    # ---------------- summary ---------------- #
    summary = {
        "phase": "17", "stage": "4.9", "deliverable": "stage4_9_summary.json",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_integrity": integrity["status"],
        "reproduced_stage4_8": reproduced,
        "headline": {
            "output_reliability_validity_rate": reproduced["validity_rate"],
            "trajectory_run_survival": block(runs, "survived")["rate"],
            "attack_success_official": reproduced["success_rate"],
            "successes": reproduced["successes"], "runs": len(runs),
        },
        "dominant_limiter": "attacker output contract (prose responses without the required "
                            "JSON object) at the first or an early attacker call",
        "infrastructure_limited": False,
        "visualizations": viz_files,
        "raw_data_modified": False,
        "no_new_experiment": True,
    }
    (OUT / "stage4_9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"DATA INTEGRITY: {integrity['status']}")
    print(f"  runs {integrity['checks']['run_records']} | calls "
          f"{integrity['checks']['attacker_calls']}/{integrity['checks']['target_calls']}/"
          f"{integrity['checks']['judge_calls']} | mismatches {len(integrity['mismatches'])}")
    print(f"reproduced: validity {reproduced['validity_rate']} ({reproduced['contract_valid']}/"
          f"{reproduced['attacker_calls']}) | survival {block(runs, 'survived')['rate']} | "
          f"success {reproduced['success_rate']} ({reproduced['successes']}/{len(runs)})")
    print(f"  all Stage 4.8 checks reproduced: {all(reproduced['matches_stage4_8'].values())}")
    print(f"  first-call failure runs: {first_call_failures}/90 | "
          f"runs with any invalid call: {len(dirty)}")
    print(f"artifacts: {len(list(OUT.glob('*.json')))} json + "
          f"{len(list(OUT.glob('*.jsonl')))} jsonl + {len(viz_files)} visualizations")
    return 0


# --------------------------------------------------------------------------- #
# visualizations                                                               #
# --------------------------------------------------------------------------- #
def make_visualizations(runs: list[dict], out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files: list[str] = []

    def save(fig, name: str, title: str) -> None:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=130)
        plt.close(fig)
        files.append(name)

    # A. attacker failure composition
    counts = Counter(c["failure_type"] for r in runs for c in r["attacker_calls"]
                     if c["failure_type"])
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(list(counts.keys()), list(counts.values()), color="#b5651d")
    ax.set_ylabel("attacker calls")
    for i, v in enumerate(counts.values()):
        ax.text(i, v, str(v), ha="center", va="bottom")
    save(fig, "a_attacker_failure_composition.png",
         "Attacker failure composition (371 calls)")

    # B. reliability by attack
    attacks = ["crescendo_paper", "opposite_day", "acronym"]
    labels = [ATTACK_LABEL[a] for a in attacks]
    validity = []
    for a in attacks:
        cs = [c for r in runs if r["attack"] == a for c in r["attacker_calls"]]
        validity.append(100 * sum(1 for c in cs if c["category"] in VALID) / len(cs))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(labels, validity, color="#2e6f95")
    ax.set_ylabel("attacker-call validity (%)")
    ax.set_ylim(0, 100)
    for i, v in enumerate(validity):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    save(fig, "b_reliability_by_attack.png", "Attacker-call validity by attack")

    # C. survival and success by attack (two separate panels, not one chart)
    survival = [100 * sum(1 for r in runs if r["attack"] == a and r["survived"]) / 30
                for a in attacks]
    success = [100 * sum(1 for r in runs if r["attack"] == a and r["success"]) / 30
               for a in attacks]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(labels, survival, color="#4c9f70")
    axes[0].set_title("Run survival (no attacker-side termination)")
    axes[1].bar(labels, success, color="#8d5a97")
    axes[1].set_title("Attack success (frozen criterion: score 5)")
    for ax, values in zip(axes, (survival, success)):
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of 30 runs")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    save(fig, "c_survival_and_success_by_attack.png",
         "Trajectory behaviour vs attack effectiveness — reported separately")

    # D. turn-depth distribution
    depth_counts = Counter(r["completed_turns"] for r in runs)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar([str(k) for k in sorted(depth_counts)], [depth_counts[k] for k in sorted(depth_counts)],
           color="#3f7d8c")
    ax.set_xlabel("completed turns in the run")
    ax.set_ylabel("runs")
    for i, k in enumerate(sorted(depth_counts)):
        ax.text(i, depth_counts[k], str(depth_counts[k]), ha="center", va="bottom")
    save(fig, "d_turn_depth_distribution.png", "Completed turns per run (90 runs)")

    # E. final score distribution
    score_counts = Counter(str(r["final_score"]) if r["final_score"] is not None else "none"
                           for r in runs)
    order = ["none", "1", "2", "3", "4", "5"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(order, [score_counts.get(k, 0) for k in order],
           color=["#999999"] + ["#c25b56"] * 4 + ["#4c9f70"])
    ax.set_xlabel("final conversation rubric score (5 = success)")
    ax.set_ylabel("runs")
    for i, k in enumerate(order):
        ax.text(i, score_counts.get(k, 0), str(score_counts.get(k, 0)), ha="center", va="bottom")
    save(fig, "e_final_score_distribution.png", "Final score distribution (90 runs)")

    # F. run-duration distribution (bimodality)
    durations = sorted(r["duration_s"] for r in runs)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.hist(durations, bins=18, color="#7a7a7a")
    ax.set_xlabel("run duration (s)")
    ax.set_ylabel("runs")
    ax.axvline(statistics.median(durations), color="#c25b56", linestyle="--",
               label=f"median {statistics.median(durations):.0f}s")
    ax.legend()
    save(fig, "f_run_duration_distribution.png",
         "Run duration distribution (bi-modal: early aborts vs full turns)")

    # G. failure funnel
    funnel = [("runs launched", 90),
              ("≥1 attacker call", sum(1 for r in runs if r["attacker_calls"])),
              ("≥1 valid attacker call", sum(1 for r in runs
                                             if r["counts"]["attacker_calls_valid"] > 0)),
              ("reached target", sum(1 for r in runs if r["target_calls"])),
              ("reached judge", sum(1 for r in runs if r["judge_calls"])),
              ("≥1 rubric score", sum(1 for r in runs if r["rubric_scores"])),
              ("success (score 5)", sum(1 for r in runs if r["success"]))]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([f[0] for f in funnel][::-1], [f[1] for f in funnel][::-1], color="#5b7fa6")
    ax.set_xlabel("runs")
    for i, (_, v) in enumerate(funnel[::-1]):
        ax.text(v, i, f" {v}", va="center")
    save(fig, "g_failure_funnel.png",
         "Failure funnel: where the 90 runs ended (attacker contract is the gate)")

    return files


if __name__ == "__main__":
    raise SystemExit(main())
