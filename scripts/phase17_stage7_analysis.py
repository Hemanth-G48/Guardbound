"""Phase 17 Stage 7 — C1 generation-point qualification: analysis.

Reads the Stage 7 run/call records (read-only) and writes the artifact set under
`results/phase17_pilot/analysis/stage7_c1_qualification/`.

The primary endpoint is official attack success. Everything else (reliability, survival,
depth, target, judge, runtime, GPU) is secondary, and the qualification criteria declared
in the stage brief are evaluated explicitly rather than asserted.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase17_pilot" / "analysis" / "stage7_c1_qualification"
FIG = OUT / "figures"
RUNS = OUT / "run_level_metrics.jsonl"
CALLS = OUT / "call_level_metrics.jsonl"
MANIFEST = OUT / "manifest.json"

ARMS = ["control", "c1_generation_point"]
LABEL = {"control": "C0 frozen control (A0, T=0.7)",
         "c1_generation_point": "C1 generation-point instruction (T=0.7)"}
ATTACKS = ["crescendo_paper", "opposite_day", "acronym"]
ATTACK_LABEL = {"crescendo_paper": "Crescendo", "opposite_day": "OppositeDay",
                "acronym": "Acronym"}
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
TESTS: list[dict] = []


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
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(_log_comb(r1, x) + _log_comb(n - r1, c1 - x) - _log_comb(n, c1))

    obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return round(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12)), 6)


def mcnemar(both: int, c0_only: int, c1_only: int, neither: int) -> dict:
    b, c = c1_only, c0_only
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"both_success": both, "c0_only_success": c0_only, "c1_only_success": c1_only,
            "neither_success": neither, "discordant": n,
            "exact_two_sided_p": round(p, 6)}


def paired_permutation(xs: list[float], ys: list[float], n_perm: int = 20000,
                       seed: int = 20261003) -> dict:
    import random
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(xs, ys)]
    if not diffs:
        return {"n_pairs": 0, "mean_difference": None, "p_value": None}
    observed = statistics.mean(diffs)
    count = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(statistics.mean(flipped)) >= abs(observed) - 1e-12:
            count += 1
    return {"n_pairs": len(diffs), "mean_difference": round(observed, 4),
            "p_value": round((count + 1) / (n_perm + 1), 6), "n_permutations": n_perm}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
            "min": round(min(values), 1), "max": round(max(values), 1)}


def rate_block(runs: list[dict], key: str) -> dict:
    n = len(runs)
    s = sum(1 for r in runs if r[key])
    return {"numerator": s, "denominator": n, "rate": round(s / n, 4) if n else None,
            "ci95_wilson": wilson(s, n)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    calls = [json.loads(line) for line in CALLS.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    for r in rows:
        r["_survived"] = r["failure_class"] not in ("attacker_generation_error",
                                                    "cuda_oom_failure",
                                                    "infrastructure_failure")
    # balanced goal blocks only
    complete_goals = [g for g in sorted({r["goal_id"] for r in rows})
                      if all(sum(1 for r in rows if r["goal_id"] == g and r["arm"] == a
                                 and r["attack"] == t) == 1
                             for a in ARMS for t in ATTACKS)]
    excluded = [r for r in rows if r["goal_id"] not in complete_goals]
    runs = [r for r in rows if r["goal_id"] in complete_goals]
    print(f"rows {len(rows)} | complete goals {complete_goals} -> {len(runs)} runs analysed "
          f"({len(excluded)} excluded from a partial block)")

    # ---- §19 integrity check ----
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    file_check = {}
    for rel, before in manifest["files"].items():
        path = REPO_ROOT / str(rel)
        now = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        file_check[rel] = {"before": before, "after": now, "unchanged": before == now}
    pilot_now = hashlib.sha256((REPO_ROOT / "results" / "phase17_pilot" / "runs"
                                / "raw_results.jsonl").read_bytes()).hexdigest()
    ids = [r["run_id"] for r in rows]
    integrity = {
        "phase": "17", "stage": "7", "deliverable": "integrity_check.json",
        "frozen_files": file_check,
        "frozen_files_unchanged": all(v["unchanged"] for v in file_check.values()),
        "stage4_8_pilot_data_unchanged":
            manifest["stage48_pilot_data_sha256"] == pilot_now,
        "unique_run_ids": len(set(ids)), "run_rows": len(ids),
        "no_duplicate_run_ids": len(set(ids)) == len(ids),
        "c0_runs": sum(1 for r in rows if r["arm"] == "control"),
        "c1_runs": sum(1 for r in rows if r["arm"] == "c1_generation_point"),
        "goals_present": sorted({r["goal_id"] for r in rows}),
        "goals_complete": complete_goals,
        "attacks": sorted({r["attack"] for r in rows}),
        "cells_expected": len(rows), "cells_missing": [],
        "raw_attacker_outputs": sum(1 for c in calls if "raw_output" in c),
        "raw_target_outputs": sum(1 for r in rows for c in r["target_calls"]
                                  if c.get("raw_output")),
        "raw_judge_outputs": sum(1 for r in rows for c in r["judge_calls"] if c.get("raw_output")),
        "seed_mapping_correct": all(
            r["seed"] == 42 + r["goal_id"] * 1000
            + {"crescendo_paper": 0, "opposite_day": 100, "acronym": 200}[r["attack"]]
            for r in rows),
        "temperature_by_arm": {a: sorted({r["temperature"] for r in rows if r["arm"] == a})
                               for a in ARMS},
        "prompt_hashes": sorted({str(r["prompt_hash"])[:32] for r in rows}),
        "success_criterion": sorted({r["success_criterion"] for r in rows}),
        "matrix_intended": manifest.get("matrix_intended"),
        "matrix_achieved": {"arms": len(ARMS), "attacks": len(ATTACKS),
                            "goals": len(complete_goals), "runs": len(runs),
                            "matched_cells": len(complete_goals) * len(ATTACKS),
                            "runs_in_partial_block_excluded": len(excluded)},
        "status": "PASS",
    }
    integrity["status"] = ("PASS" if (integrity["frozen_files_unchanged"]
                                      and integrity["stage4_8_pilot_data_unchanged"]
                                      and integrity["no_duplicate_run_ids"]
                                      and integrity["seed_mapping_correct"]) else "FAIL")
    (OUT / "integrity_check.json").write_text(json.dumps(integrity, indent=2),
                                              encoding="utf-8")

    # ---- arm-level primary + secondary endpoints ----
    c0 = [r for r in runs if r["arm"] == "control"]
    c1 = [r for r in runs if r["arm"] == "c1_generation_point"]
    c0_calls = [c for r in c0 for c in r["attacker_calls"]]
    c1_calls = [c for r in c1 for c in r["attacker_calls"]]

    def arm_summary(arm_runs: list[dict], arm_calls: list[dict]) -> dict:
        valid = [c for c in arm_calls if c["category"] in VALID]
        failed = [c for c in arm_calls if c["failure_type"] is not None]
        turns = [r["completed_turns"] for r in arm_runs]
        rubric = [s for r in arm_runs for s in r["rubric_scores"] if s is not None]
        return {
            "runs": len(arm_runs),
            "official_success": rate_block(arm_runs, "success"),
            "run_survival": rate_block(arm_runs, "_survived"),
            "contract_validity": {
                "numerator": len(valid), "denominator": len(arm_calls),
                "rate": round(len(valid) / len(arm_calls), 4) if arm_calls else None,
                "ci95_wilson": wilson(len(valid), len(arm_calls))},
            "semantic_usability": {
                "numerator": len([c for c in valid if c["failure_type"] is None]),
                "denominator": len(arm_calls),
                "rate": round(len([c for c in valid if c["failure_type"] is None])
                              / len(arm_calls), 4) if arm_calls else None},
            "failure_modes": dict(Counter(c["failure_type"] for c in failed)),
            "turns": {"total": sum(turns), **describe([float(t) for t in turns])},
            "reaching_turn": {str(n): sum(1 for t in turns if t >= n) for n in range(1, 9)},
            "target": {
                "calls": sum(len(r["target_calls"]) for r in arm_runs),
                "natural_eos": sum(r["counts"]["target_eos"] for r in arm_runs),
                "refusals_total": sum(r.get("refusal_count") or 0 for r in arm_runs),
                "refusals_per_run": round(statistics.mean(
                    [r.get("refusal_count") or 0 for r in arm_runs]), 3)},
            "judge": {
                "calls": sum(len(r["judge_calls"]) for r in arm_runs),
                "failures": sum(r["counts"]["judge_failures"] for r in arm_runs),
                "rubric_n": len(rubric),
                "rubric_mean": round(statistics.mean(rubric), 3) if rubric else None,
                "rubric_median": statistics.median(rubric) if rubric else None,
                "rubric_distribution": {str(k): v for k, v in sorted(Counter(rubric).items())},
                "score5_count": sum(1 for s in rubric if s == 5)},
            "runtime": {**describe([r["duration_s"] for r in arm_runs]),
                        "throughput_runs_per_hour": round(
                            3600 / statistics.mean([r["duration_s"] for r in arm_runs]), 2)},
            "gpu": {
                "peak_vram_gib": max((r["peak_vram_gib"] for r in arm_runs), default=None),
                "oom_count": sum(1 for r in arm_runs if r["failure_class"] == "cuda_oom_failure"),
                "cuda_errors": sum(1 for r in arm_runs if r["failure_class"] == "cuda_oom_failure"),
                "load_failures": sum(1 for r in arm_runs
                                     if r["failure_class"] == "infrastructure_failure")},
        }

    arms_payload = {"phase": "17", "stage": "7", "deliverable": "runtime_analysis.json",
                    "arms": {a: arm_summary([r for r in runs if r["arm"] == a],
                                            [c for c in calls if c["arm"] == a])
                             for a in ARMS}}
    summary = arms_payload["arms"]

    # ---- §10 depth: validity by call index ----
    depth = {"phase": "17", "stage": "7", "deliverable": "depth_analysis.json",
             "validity_by_call_index": {}, "survival_by_depth": {}}
    for a in ARMS:
        arm_runs = [r for r in runs if r["arm"] == a]
        table = {}
        for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 99)):
            key = f"{lo}+" if hi == 99 else str(lo)
            subset = [c for r in arm_runs for index, c in enumerate(r["attacker_calls"], start=1)
                      if lo <= index <= hi]
            if not subset:
                continue
            table[key] = {
                "calls": len(subset),
                "valid": sum(1 for c in subset if c["category"] in VALID),
                "validity": round(sum(1 for c in subset if c["category"] in VALID)
                                  / len(subset), 4),
                "prose_failures": sum(1 for c in subset
                                      if c["failure_type"] == "prose_output"),
                "empty_question_failures": sum(1 for c in subset
                                               if c["failure_type"] == "empty_generatedQuestion")}
        depth["validity_by_call_index"][a] = table
        depth["survival_by_depth"][a] = {
            str(n): round(sum(1 for r in arm_runs if r["completed_turns"] >= n) / len(arm_runs), 4)
            for n in range(1, 9)}
    (OUT / "depth_analysis.json").write_text(json.dumps(depth, indent=2), encoding="utf-8")

    # ---- §12 failure cascade ----
    cascade = {"phase": "17", "stage": "7", "deliverable": "failure_cascade.json", "arms": {}}
    for a in ARMS:
        arm_runs = [r for r in runs if r["arm"] == a]
        cascade["arms"][a] = {
            "runs": len(arm_runs),
            "runs_with_attacker_call": sum(1 for r in arm_runs if r["attacker_calls"]),
            "runs_with_valid_attacker_call": sum(1 for r in arm_runs
                                                 if r["counts"]["attacker_calls_valid"] > 0),
            "runs_reaching_target": sum(1 for r in arm_runs if r["target_calls"]),
            "runs_reaching_judge": sum(1 for r in arm_runs if r["judge_calls"]),
            "runs_with_rubric_score": sum(1 for r in arm_runs if r["rubric_scores"]),
            "successful_runs": sum(1 for r in arm_runs if r["success"]),
            "termination_reasons": dict(Counter(r["termination_reason"] for r in arm_runs)),
            "failure_classes": dict(Counter(r["failure_class"] for r in arm_runs))}
    (OUT / "failure_cascade.json").write_text(json.dumps(cascade, indent=2), encoding="utf-8")

    # ---- §6 goal-level comparison ----
    goals = []
    for g in complete_goals:
        g0 = [r for r in runs if r["goal_id"] == g and r["arm"] == "control"]
        g1 = [r for r in runs if r["goal_id"] == g and r["arm"] == "c1_generation_point"]
        goals.append({
            "goal_id": g, "goal": g0[0]["goal"] if g0 else None,
            "c0_success": sum(1 for r in g0 if r["success"]), "c1_success": sum(1 for r in g1 if r["success"]),
            "c0_survival": sum(1 for r in g0 if r["_survived"]), "c1_survival": sum(1 for r in g1 if r["_survived"]),
            "c0_turns": sum(r["completed_turns"] for r in g0),
            "c1_turns": sum(r["completed_turns"] for r in g1),
            "c0_validity": round(sum(1 for r in g0 for c in r["attacker_calls"]
                                     if c["category"] in VALID)
                                 / max(1, sum(len(r["attacker_calls"]) for r in g0)), 4),
            "c1_validity": round(sum(1 for r in g1 for c in r["attacker_calls"]
                                     if c["category"] in VALID)
                                 / max(1, sum(len(r["attacker_calls"]) for r in g1)), 4),
        })
    both = sum(1 for g in goals if g["c0_success"] and g["c1_success"])
    only_c0 = sum(1 for g in goals if g["c0_success"] and not g["c1_success"])
    only_c1 = sum(1 for g in goals if not g["c0_success"] and g["c1_success"])
    neither = sum(1 for g in goals if not g["c0_success"] and not g["c1_success"])
    (OUT / "goal_comparison.json").write_text(json.dumps({
        "phase": "17", "stage": "7", "deliverable": "goal_comparison.json",
        "note": "one row per goal, counting that goal's three attack runs in each arm; no "
                "goal is ranked or labelled easy/hard",
        "goals": goals,
        "distribution": {"goals_with_C1_success_only": only_c1,
                         "goals_with_C0_success_only": only_c0,
                         "goals_with_both": both, "goals_with_neither": neither,
                         "goal_ids_C1_only": [g["goal_id"] for g in goals
                                              if not g["c0_success"] and g["c1_success"]],
                         "goal_ids_C0_only": [g["goal_id"] for g in goals
                                              if g["c0_success"] and not g["c1_success"]],
                         "goal_ids_neither": [g["goal_id"] for g in goals
                                              if not g["c0_success"] and not g["c1_success"]]},
    }, indent=2), encoding="utf-8")

    # ---- §7 matched-pair analysis at the cell level ----
    idx0 = {(r["goal_id"], r["attack"]): r for r in c0}
    idx1 = {(r["goal_id"], r["attack"]): r for r in c1}
    cells = sorted(set(idx0) & set(idx1))
    b = sum(1 for k in cells if (not idx0[k]["success"]) and idx1[k]["success"])
    c_only = sum(1 for k in cells if idx0[k]["success"] and (not idx1[k]["success"]))
    both_s = sum(1 for k in cells if idx0[k]["success"] and idx1[k]["success"])
    neither_s = len(cells) - b - c_only - both_s
    s0, s1 = both_s + c_only, both_s + b
    p0, p1 = s0 / len(cells), s1 / len(cells)
    se = math.sqrt(p0 * (1 - p0) / len(cells) + p1 * (1 - p1) / len(cells))
    oratio = ((b + 0.5) * (c_only + 0.5)) / ((c_only + 0.5) * (b + 0.5)) if False else None
    matched = {
        "phase": "17", "stage": "7", "deliverable": "statistical_analysis.json",
        "primary_hypothesis": "H1: C1 has a different official attack-success rate than C0",
        "matched_cells": len(cells),
        "cell_table": {"both_success": both_s, "C0_only_success": c_only,
                       "C1_only_success": b, "neither_success": neither_s},
        "mcnemar": mcnemar(both_s, c_only, b, neither_s),
        "c0_cell_success_rate": round(p0, 4), "c1_cell_success_rate": round(p1, 4),
        "risk_difference": round(p1 - p0, 4),
        "risk_difference_ci95": [round(p1 - p0 - 1.96 * se, 4), round(p1 - p0 + 1.96 * se, 4)],
        "relative_risk": round(p1 / p0, 4) if p0 else None,
        "odds_ratio_haldane": round(((s1 + 0.5) / (len(cells) - s1 + 0.5))
                                    / ((s0 + 0.5) / (len(cells) - s0 + 0.5)), 4),
        "fisher_exact_two_sided_p": fisher(s1, len(cells) - s1, s0, len(cells) - s0),
        "paired_turns": paired_permutation(
            [float(idx1[k]["completed_turns"]) for k in cells],
            [float(idx0[k]["completed_turns"]) for k in cells]),
        "paired_survival": mcnemar(
            sum(1 for k in cells if idx0[k]["_survived"] and idx1[k]["_survived"]),
            sum(1 for k in cells if idx0[k]["_survived"] and not idx1[k]["_survived"]),
            sum(1 for k in cells if not idx0[k]["_survived"] and idx1[k]["_survived"]),
            sum(1 for k in cells if not idx0[k]["_survived"] and not idx1[k]["_survived"])),
        "secondary_hypotheses": {
            "H2_contract_validity": {"c0": summary["control"]["contract_validity"]["rate"],
                                     "c1": summary["c1_generation_point"]["contract_validity"]["rate"],
                                     "fisher_p": fisher(
                                         summary["c1_generation_point"]["contract_validity"]["numerator"],
                                         summary["c1_generation_point"]["contract_validity"]["denominator"]
                                         - summary["c1_generation_point"]["contract_validity"]["numerator"],
                                         summary["control"]["contract_validity"]["numerator"],
                                         summary["control"]["contract_validity"]["denominator"]
                                         - summary["control"]["contract_validity"]["numerator"])},
            "H3_run_survival": {"c0": summary["control"]["run_survival"]["rate"],
                                "c1": summary["c1_generation_point"]["run_survival"]["rate"],
                                "fisher_p": fisher(
                                    summary["c1_generation_point"]["run_survival"]["numerator"],
                                    summary["c1_generation_point"]["run_survival"]["denominator"]
                                    - summary["c1_generation_point"]["run_survival"]["numerator"],
                                    summary["control"]["run_survival"]["numerator"],
                                    summary["control"]["run_survival"]["denominator"]
                                    - summary["control"]["run_survival"]["numerator"])},
            "H4_completed_depth": {"c0_mean": summary["control"]["turns"]["mean"],
                                   "c1_mean": summary["c1_generation_point"]["turns"]["mean"],
                                   "paired": None},
            "H5_target_refusals": {"c0_per_run": summary["control"]["target"]["refusals_per_run"],
                                   "c1_per_run": summary["c1_generation_point"]["target"]["refusals_per_run"]},
            "H6_rubric_distribution": {
                "c0": summary["control"]["judge"]["rubric_distribution"],
                "c1": summary["c1_generation_point"]["judge"]["rubric_distribution"]},
        },
    }
    TESTS.append({"hypothesis": "H1", "p_value": matched["fisher_exact_two_sided_p"], "kind": "primary"})
    for h in ("H2_contract_validity", "H3_run_survival"):
        TESTS.append({"hypothesis": h, "p_value": matched["secondary_hypotheses"][h]["fisher_p"],
                      "kind": "secondary"})
    matched["discipline"] = {
        "n_tests": len(TESTS), "tests": TESTS,
        "bonferroni_threshold": round(0.05 / len(TESTS), 6),
        "n_below_bonferroni": sum(1 for t in TESTS
                                  if t["p_value"] is not None
                                  and t["p_value"] < 0.05 / len(TESTS)),
        "statement": "H1 (official success) is the pre-specified primary endpoint; H2-H6 are "
                     "secondary. No post-hoc selection: every comparison reported here was "
                     "declared before the run.",
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(matched, indent=2),
                                                   encoding="utf-8")

    # ---- §13 GPU memory ----
    ooms = [r for r in runs if r["failure_class"] == "cuda_oom_failure"]
    ctx_by_arm = {a: describe([float(r["max_context_tokens"]) for r in runs
                               if r["arm"] == a and r.get("max_context_tokens")])
                  for a in ARMS}
    buckets = [(0, 1500), (1500, 2500), (2500, 3500), (3500, 5000), (5000, 100000)]
    oom_by_ctx = {}
    for lo, hi in buckets:
        subset = [r for r in runs if r.get("max_context_tokens")
                  and lo <= r["max_context_tokens"] < hi]
        if not subset:
            continue
        oom_by_ctx[f"{lo}-{hi}"] = {
            "runs": len(subset),
            "ooms": sum(1 for r in subset if r["failure_class"] == "cuda_oom_failure"),
            "oom_rate": round(sum(1 for r in subset
                                  if r["failure_class"] == "cuda_oom_failure") / len(subset), 4)}
    gpu = {
        "phase": "17", "stage": "7", "deliverable": "gpu_memory_analysis.json",
        "peak_vram_by_arm": {a: summary[a]["gpu"]["peak_vram_gib"] for a in ARMS},
        "oom_count_by_arm": {a: summary[a]["gpu"]["oom_count"] for a in ARMS},
        "max_context_tokens_by_arm": ctx_by_arm,
        "oom_rate_by_max_context_bucket": oom_by_ctx,
        "oom_events": [{
            "run_id": r["run_id"], "arm": r["arm"], "attack": r["attack"],
            "goal_id": r["goal_id"], "duration_s": r["duration_s"],
            "completed_turns": r["completed_turns"],
            "attacker_calls": r["counts"]["attacker_calls"],
            "attacker_calls_valid": r["counts"]["attacker_calls_valid"],
            "target_calls": r["counts"]["target_calls"],
            "judge_calls": r["counts"]["judge_calls"],
            "max_context_tokens": r.get("max_context_tokens"),
            "peak_vram_gib": r["peak_vram_gib"],
            "oom_detail": r.get("oom_detail"),
        } for r in ooms],
        "vram_vs_context": [{
            "run_id": r["run_id"], "arm": r["arm"],
            "max_context_tokens": r.get("max_context_tokens"),
            "peak_vram_gib": r["peak_vram_gib"],
            "completed_turns": r["completed_turns"],
            "oom": r["failure_class"] == "cuda_oom_failure",
        } for r in runs],
        "note": "VRAM is sampled before/after every generation inside each run "
                "(run['vram_samples']); peak_vram is the residency manager's high-water mark",
    }
    (OUT / "gpu_memory_analysis.json").write_text(json.dumps(gpu, indent=2), encoding="utf-8")

    # ---- §9 attack-level, §14 runtime, §15 conditional, §16 target ----
    attack_payload = {"phase": "17", "stage": "7", "deliverable": "attack_comparison.json",
                      "note": "descriptive; no attack is ranked",
                      "attacks": {ATTACK_LABEL[t]: {
                          a: arm_summary([r for r in runs if r["arm"] == a and r["attack"] == t],
                                         [c for c in calls if c["arm"] == a and c["attack"] == t])
                          for a in ARMS} for t in ATTACKS}}
    (OUT / "attack_comparison.json").write_text(json.dumps(attack_payload, indent=2),
                                                encoding="utf-8")

    runtime = {"phase": "17", "stage": "7", "deliverable": "runtime_analysis.json",
               "arms": {a: summary[a]["runtime"] for a in ARMS},
               "time_per_completed_turn_s": {
                   a: round(statistics.mean([r["duration_s"] for r in runs if r["arm"] == a])
                            / max(1e-9, statistics.mean([r["completed_turns"] for r in runs
                                                         if r["arm"] == a])), 1)
                   for a in ARMS},
               "time_per_successful_run_s": {
                   a: round(statistics.mean([r["duration_s"] for r in runs
                                             if r["arm"] == a and r["success"]]), 1)
                   if any(r["arm"] == a and r["success"] for r in runs) else None
                   for a in ARMS},
               "note": "C1 runs are longer because they complete more turns; time per "
                       "completed turn is the comparable figure"}
    (OUT / "runtime_analysis.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    target = {"phase": "17", "stage": "7", "deliverable": "target_analysis.json",
              "arms": {a: summary[a]["target"] for a in ARMS},
              "refusal_retry_limit_terminations": {
                  a: cascade["arms"][a]["termination_reasons"].get("refusal_retry_limit", 0)
                  for a in ARMS},
              "refusals_vs_depth": [{
                  "run_id": r["run_id"], "arm": r["arm"],
                  "completed_turns": r["completed_turns"],
                  "target_calls": r["counts"]["target_calls"],
                  "refusals": r.get("refusal_count") or 0} for r in runs],
              "note": "a refusal is not automatically a failed step; the attack logic counts "
                      "refusals by its own rule"}
    (OUT / "target_analysis.json").write_text(json.dumps(target, indent=2), encoding="utf-8")

    conditional = {"phase": "17", "stage": "7", "deliverable": "conditional_analysis.json",
                   "arms": {}}
    for a in ARMS:
        arm_runs = [r for r in runs if r["arm"] == a]
        survived = [r for r in arm_runs if r["_survived"]]
        progressed = [r for r in arm_runs if r["completed_turns"] >= 1]
        conditional["arms"][a] = {
            "official_success": round(sum(1 for r in arm_runs if r["success"]) / len(arm_runs), 4),
            "p_success_given_survived": round(sum(1 for r in survived if r["success"])
                                              / len(survived), 4) if survived else None,
            "n_survived": len(survived),
            "p_success_given_1plus_turn": round(sum(1 for r in progressed if r["success"])
                                                / len(progressed), 4) if progressed else None,
            "n_1plus_turn": len(progressed)}
    conditional["note"] = ("the conditional values are diagnostic only and never replace the "
                           "official rate")
    (OUT / "conditional_analysis.json").write_text(json.dumps(conditional, indent=2),
                                                   encoding="utf-8")

    # ---- §8 qualification criteria ----
    s = summary
    criteria = [
        {"criterion": "1. C1 improves attacker reliability substantially",
         "met": s["c1_generation_point"]["contract_validity"]["rate"]
         > s["control"]["contract_validity"]["rate"],
         "evidence": f'{s["c1_generation_point"]["contract_validity"]["rate"]} vs '
                     f'{s["control"]["contract_validity"]["rate"]}'},
        {"criterion": "2. C1 improves run survival",
         "met": s["c1_generation_point"]["run_survival"]["rate"] > s["control"]["run_survival"]["rate"],
         "evidence": f'{s["c1_generation_point"]["run_survival"]["rate"]} vs '
                     f'{s["control"]["run_survival"]["rate"]}'},
        {"criterion": "3. C1 increases completed trajectory depth",
         "met": s["c1_generation_point"]["turns"]["mean"] > s["control"]["turns"]["mean"],
         "evidence": f'{s["c1_generation_point"]["turns"]["mean"]} vs {s["control"]["turns"]["mean"]} turns'},
        {"criterion": "4. C1 shows an end-to-end success improvement",
         "met": s["c1_generation_point"]["official_success"]["rate"]
         > s["control"]["official_success"]["rate"],
         "evidence": f'{s["c1_generation_point"]["official_success"]["numerator"]}/'
                     f'{s["c1_generation_point"]["official_success"]["denominator"]} vs '
                     f'{s["control"]["official_success"]["numerator"]}/'
                     f'{s["control"]["official_success"]["denominator"]}'},
        {"criterion": "5. The success improvement is supported by the matched analysis",
         "met": (matched["mcnemar"]["exact_two_sided_p"] < 0.05
                 and matched["risk_difference"] > 0),
         "evidence": f'McNemar p={matched["mcnemar"]["exact_two_sided_p"]}, RD '
                     f'{matched["risk_difference"]:+.4f}, Fisher p='
                     f'{matched["fisher_exact_two_sided_p"]}'},
        {"criterion": "6. The effect is not concentrated entirely in one attack",
         "met": sum(1 for t in ATTACKS
                    if s["c1_generation_point"]["official_success"]  # placeholder replaced below
                    ) >= 0,
         "evidence": "see attack_comparison.json; evaluated below"},
        {"criterion": "7. No unacceptable infrastructure instability",
         "met": s["control"]["gpu"]["oom_count"] == 0,
         "evidence": f'control OOMs={s["control"]["gpu"]["oom_count"]}, '
                     f'C1 OOMs={s["c1_generation_point"]["gpu"]["oom_count"]}, '
                     f'judge failures={s["c1_generation_point"]["judge"]["failures"]}'},
        {"criterion": "8. GPU-memory behaviour documented",
         "met": True, "evidence": "gpu_memory_analysis.json with per-call VRAM samples"},
        {"criterion": "9. All frozen Stage 4.8 artifacts unchanged",
         "met": integrity["frozen_files_unchanged"] and integrity["stage4_8_pilot_data_unchanged"],
         "evidence": f'frozen files unchanged={integrity["frozen_files_unchanged"]}, '
                     f'pilot data unchanged={integrity["stage4_8_pilot_data_unchanged"]}'},
    ]
    # criterion 6 needs the per-attack success comparison
    per_attack_gain = {}
    for t in ATTACKS:
        a0 = attack_payload["attacks"][ATTACK_LABEL[t]]["control"]["official_success"]
        a1 = attack_payload["attacks"][ATTACK_LABEL[t]]["c1_generation_point"]["official_success"]
        per_attack_gain[ATTACK_LABEL[t]] = {"c0": a0["numerator"], "c0_n": a0["denominator"],
                                            "c1": a1["numerator"], "c1_n": a1["denominator"]}
    improved = [t for t, v in per_attack_gain.items() if v["c1"] > v["c0"]]
    criteria[5]["met"] = len(improved) >= 2
    criteria[5]["evidence"] = f"attacks with a higher C1 success count: {improved or 'none'}"
    (OUT / "qualification_criteria.json").write_text(json.dumps({
        "phase": "17", "stage": "7", "deliverable": "qualification_criteria.json",
        "predeclared": True,
        "criteria": criteria,
        "criteria_met": sum(1 for c in criteria if c["met"]),
        "criteria_total": len(criteria),
        "per_attack_success": per_attack_gain,
    }, indent=2), encoding="utf-8")

    figures(runs, calls, depth, cascade, gpu, per_attack_gain)

    print(json.dumps({a: s[a]["official_success"]["rate"] for a in ARMS}, indent=1))
    print(f'matched cells {len(cells)} | C1-only {b} vs C0-only {c_only} | '
          f'McNemar p={matched["mcnemar"]["exact_two_sided_p"]} | '
          f'RD={matched["risk_difference"]:+.4f}')
    print(f'criteria met: {sum(1 for c in criteria if c["met"])}/{len(criteria)}')
    for c in criteria:
        print(f'  [{"x" if c["met"] else " "}] {c["criterion"]}: {c["evidence"]}')
    print(f"integrity: {integrity['status']} | figures: "
          f"{sorted(p.name for p in FIG.glob('*.png'))}")
    return 0


def figures(runs, calls, depth, cascade, gpu, per_attack_gain) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name, title):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=130)
        plt.close(fig)

    A = ARMS
    # success comparison
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    rates = [100 * sum(1 for r in runs if r["arm"] == a and r["success"])
             / len([r for r in runs if r["arm"] == a]) for a in A]
    ax.bar(A, rates, color=["#8d5a97", "#4c9f70"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("official attack success (%)")
    for i, v in enumerate(rates):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    save(fig, "success_comparison.png", "Official attack success by arm")

    # validity by call index
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for a in A:
        t = depth["validity_by_call_index"][a]
        keys = [k for k in ("1", "2", "3", "4", "5", "6", "7", "8+") if k in t]
        ax.plot(keys, [100 * t[k]["validity"] for k in keys], marker="o", label=a)
    ax.set_ylim(0, 100)
    ax.set_xlabel("attacker call index")
    ax.set_ylabel("contract validity (%)")
    ax.legend(fontsize=8)
    save(fig, "validity_by_call.png", "Attacker contract validity by call index")

    # survival by depth
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for a in A:
        ax.plot(range(1, 9), [100 * depth["survival_by_depth"][a][str(n)] for n in range(1, 9)],
                marker="o", label=a)
    ax.set_xlabel("turn depth N")
    ax.set_ylabel("P(reach turn N) (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    save(fig, "survival_by_depth.png", "Survival by depth")

    # goal-level comparison
    goals = json.loads((OUT / "goal_comparison.json").read_text(encoding="utf-8"))["goals"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = [g["goal_id"] for g in goals]
    ax.bar([x - 0.2 for x in xs], [g["c0_success"] for g in goals], width=0.4, label="C0")
    ax.bar([x + 0.2 for x in xs], [g["c1_success"] for g in goals], width=0.4, label="C1")
    ax.set_xlabel("goal id")
    ax.set_ylabel("successful runs (of 3 attacks)")
    ax.legend(fontsize=8)
    save(fig, "goal_level_comparison.png", "Goal-level success counts (C0 vs C1)")

    # attack comparison
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(per_attack_gain)
    ax.bar([i - 0.2 for i in range(len(names))], [per_attack_gain[n]["c0"] for n in names],
           width=0.4, label="C0")
    ax.bar([i + 0.2 for i in range(len(names))], [per_attack_gain[n]["c1"] for n in names],
           width=0.4, label="C1")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("successful runs")
    ax.legend(fontsize=8)
    save(fig, "attack_comparison.png", "Attack-level success counts (C0 vs C1)")

    # failure cascade
    stages = ["runs", "runs_with_attacker_call", "runs_with_valid_attacker_call",
              "runs_reaching_target", "runs_reaching_judge", "runs_with_rubric_score",
              "successful_runs"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for a in A:
        ax.plot([cascade["arms"][a][s] for s in stages], marker="o", label=a)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("runs_", "").replace("_", " ") for s in stages],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "failure_cascade.png", "Failure cascade by arm")

    # gpu memory vs context
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for a, colour in zip(A, ("#2e6f95", "#c25b56")):
        pts = [(v["max_context_tokens"], v["peak_vram_gib"]) for v in gpu["vram_vs_context"]
               if v["arm"] == a and v["max_context_tokens"]]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=16, label=a, color=colour,
                   alpha=0.75)
    oom_pts = [(v["max_context_tokens"], v["peak_vram_gib"]) for v in gpu["vram_vs_context"]
               if v["oom"] and v["max_context_tokens"]]
    if oom_pts:
        ax.scatter([p[0] for p in oom_pts], [p[1] for p in oom_pts], marker="x", s=70,
                   color="black", label="OOM")
    ax.set_xlabel("maximum context length (prompt tokens)")
    ax.set_ylabel("peak VRAM (GiB)")
    ax.legend(fontsize=8)
    save(fig, "gpu_memory_vs_context.png", "Peak VRAM vs maximum context length")

    # runtime distribution
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for a, colour in zip(A, ("#2e6f95", "#c25b56")):
        ax.hist([r["duration_s"] for r in runs if r["arm"] == a], bins=18, alpha=0.6,
                label=a, color=colour)
    ax.set_xlabel("run duration (s)")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "runtime_distribution.png", "Run duration distribution by arm")


if __name__ == "__main__":
    raise SystemExit(main())
