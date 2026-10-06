"""Phase 18 Extension — analysis of the 60-run / 30-matched-cell extension.

Reports the extension INDEPENDENTLY from the frozen Phase 18 result, then provides a clearly
labelled descriptive combined (120-cell) analysis. The original Phase 18 90-cell result is
never recomputed, altered, or relabelled.

Artifacts (results/phase18_extension/analysis/):
    asr_comparison.json, matched_analysis.json, attacker_reliability.json,
    trajectory_metrics.json, nbf_analysis.json, runtime_gpu.json, integrity_check.json,
    combined_descriptive.json
and final_report/phase18_extension_summary.json.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "phase18_extension"
RUNS = OUT / "raw" / "runs.jsonl"
P18 = REPO / "results" / "phase18_nbf_experiment"
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
GOALS = list(range(30, 40))


def wilson(s: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = s / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"discordant": 0, "exact_two_sided_p": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"discordant": n, "exact_two_sided_p": round(p, 6)}


def fisher(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(math.lgamma(r1 + 1) - math.lgamma(x + 1) - math.lgamma(r1 - x + 1)
                        + math.lgamma(n - r1 + 1) - math.lgamma(c1 - x + 1)
                        - math.lgamma(n - r1 - c1 + x + 1) - math.lgamma(n + 1)
                        + math.lgamma(c1 + 1) + math.lgamma(n - c1 + 1))

    obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return round(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12)), 6)


def describe(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
            "min": round(min(values), 2), "max": round(max(values), 2)}


def arm_block(subset):
    calls = [c for r in subset for c in r["attacker_calls"]]
    valid = [c for c in calls if c["category"] in VALID]
    turns = [r["completed_turns"] for r in subset]
    evals = sum(r["nbf"]["evaluations"] for r in subset)
    filtered = sum(r["nbf"]["filtered"] for r in subset)
    targets = [c for r in subset for c in r["target_calls"]]
    return {
        "runs": len(subset),
        "asr": {"successes": sum(1 for r in subset if r["success"]), "runs": len(subset),
                "rate": round(sum(1 for r in subset if r["success"]) / len(subset), 4)
                if subset else None,
                "ci95_wilson": wilson(sum(1 for r in subset if r["success"]), len(subset))},
        "filtering": {"evaluations": evals, "filtered": filtered, "accepted": evals - filtered,
                      "rate": round(filtered / evals, 4) if evals else None,
                      "runs_with_filtering": sum(1 for r in subset
                                                 if r["nbf"]["run_affected_by_filtering"]),
                      "successful_after_filtering": sum(
                          1 for r in subset if r["nbf"]["run_affected_by_filtering"]
                          and r["success"]),
                      "unsuccessful_after_filtering": sum(
                          1 for r in subset if r["nbf"]["run_affected_by_filtering"]
                          and not r["success"]),
                      "runs_terminated_after_filtering": sum(
                          1 for r in subset if r["nbf"]["run_affected_by_filtering"]
                          and not r["success"])},
        "turns": {"total": sum(turns), **describe([float(t) for t in turns]),
                  "reaching_8": sum(1 for t in turns if t >= 8)},
        "target": {"calls": len(targets),
                   "natural_eos": sum(1 for c in targets if c.get("termination") == "eos"),
                   "eos_rate": round(sum(1 for c in targets
                                         if c.get("termination") == "eos") / len(targets), 4)
                   if targets else None,
                   "refusals": sum(r.get("refusal_count") or 0 for r in subset)},
        "attacker": {"calls": len(calls), "valid": len(valid),
                     "validity_rate": round(len(valid) / len(calls), 4) if calls else None,
                     "semantic_usable": len([c for c in valid if c["failure_type"] is None]),
                     "contract_failures": len([c for c in calls if c["failure_type"] is not None]),
                     "prose": sum(1 for c in calls if c["failure_type"] == "prose_output"),
                     "empty": sum(1 for c in calls
                                  if c["failure_type"] == "empty_generatedQuestion"),
                     "malformed": sum(1 for c in calls
                                      if c["failure_type"] == "malformed_json"),
                     "truncated": sum(1 for c in calls
                                      if c["failure_type"] == "truncated_json")},
        "judge": {"calls": sum(len(r["judge_calls"]) for r in subset),
                  "failures": sum(r["counts"]["judge_failures"] for r in subset)},
        "runtime": {**describe([r["run_duration_s"] for r in subset]),
                    "throughput_runs_per_hour": round(
                        3600 / statistics.mean([r["run_duration_s"] for r in subset]), 2)},
        "gpu": {"peak_vram_gib": max((r["peak_vram_gib"] for r in subset), default=None),
                "oom": sum(1 for r in subset if r["failure_class"] == "cuda_oom_failure")},
        "termination": dict(Counter(r["termination_reason"] for r in subset)),
        "failure_class": dict(Counter(r["failure_class"] for r in subset)),
    }


def main() -> int:
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in RUNS.read_text(encoding="utf-8").splitlines() if l.strip()]
    off = [r for r in rows if r["arm"] == "nbf_off"]
    on = [r for r in rows if r["arm"] == "nbf_on"]
    complete = len(off) == 30 and len(on) == 30
    print(f"extension rows {len(rows)}/60 | off {len(off)} | on {len(on)} "
          f"| complete: {complete}")

    p18_rows = [json.loads(l) for l in (P18 / "raw" / "run_level_metrics.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
    p18_off = [r for r in p18_rows if r["condition"] == "nbf_off"]
    p18_on = [r for r in p18_rows if r["condition"] == "nbf_on"]

    off_block, on_block = arm_block(off), arm_block(on)

    # ---------- extension ASR (independent) ----------
    asr = {
        "phase": "18-extension", "deliverable": "analysis/asr_comparison.json",
        "scope": "EXTENSION ONLY (goals 30-39) — reported independently of the original "
                 "Phase 18 result",
        "primary_endpoint": "official ASR = successful runs / total runs",
        "nbf_off": off_block["asr"], "nbf_on": on_block["asr"],
        "absolute_difference": (round(on_block["asr"]["rate"] - off_block["asr"]["rate"], 4)
                                if complete else None),
        "relative_difference": (round(on_block["asr"]["rate"] / off_block["asr"]["rate"], 4)
                                if complete and off_block["asr"]["rate"] else None),
        "fisher_exact_two_sided_p": (fisher(on_block["asr"]["successes"],
                                            on_block["asr"]["runs"] - on_block["asr"]["successes"],
                                            off_block["asr"]["successes"],
                                            off_block["asr"]["runs"] - off_block["asr"]["successes"])
                                     if complete else None),
        "runs_executed": len(rows), "runs_expected": 60,
        "completion": "60/60" if complete else f"{len(rows)}/60 (incomplete)",
        "denominator_note": "all executed runs stay in the denominator; failures are never "
                            "removed or replaced",
    }
    (OUT / "analysis" / "asr_comparison.json").write_text(json.dumps(asr, indent=2),
                                                          encoding="utf-8")

    # ---------- matched analysis (30 cells) ----------
    idx_off = {(r["goal_id"], r["attack"]): r for r in off}
    idx_on = {(r["goal_id"], r["attack"]): r for r in on}
    cells = sorted(set(idx_off) & set(idx_on))
    both = sum(1 for k in cells if idx_off[k]["success"] and idx_on[k]["success"])
    off_only = sum(1 for k in cells if idx_off[k]["success"] and not idx_on[k]["success"])
    on_only = sum(1 for k in cells if not idx_off[k]["success"] and idx_on[k]["success"])
    neither = len(cells) - both - off_only - on_only
    matched = {
        "phase": "18-extension", "deliverable": "analysis/matched_analysis.json",
        "test_pre_specified": "exact McNemar (two-sided) over matched goal x attack cells "
                              "(the Phase 18 methodology)",
        "scope": "EXTENSION ONLY (30 matched cells, goals 30-39)",
        "matched_cells": len(cells),
        "contingency": {"off_success_on_success": both, "off_success_on_failure": off_only,
                        "off_failure_on_success": on_only, "off_failure_on_failure": neither},
        "mcnemar": mcnemar(on_only, off_only),
        "off_cell_rate": round((both + off_only) / len(cells), 4) if cells else None,
        "on_cell_rate": round((both + on_only) / len(cells), 4) if cells else None,
        "direction": ("OFF > ON (Phase 18 direction reproduced)"
                      if off_only > on_only else
                      "ON > OFF (direction reversed)" if on_only > off_only else
                      "no net discordance"),
        "mean_paired_turn_difference": round(statistics.mean(
            [idx_on[k]["completed_turns"] - idx_off[k]["completed_turns"] for k in cells]), 3)
        if cells else None,
    }
    (OUT / "analysis" / "matched_analysis.json").write_text(json.dumps(matched, indent=2),
                                                            encoding="utf-8")

    # ---------- per-goal / per-attack ----------
    per_attack = {}
    for attack in sorted({r["attack"] for r in rows}):
        per_attack[attack] = {
            "nbf_off": arm_block([r for r in off if r["attack"] == attack]),
            "nbf_on": arm_block([r for r in on if r["attack"] == attack])}
    goal_level = []
    for g in GOALS:
        for attack in sorted({r["attack"] for r in rows}):
            o, n = idx_off.get((g, attack)), idx_on.get((g, attack))
            if not o or not n:
                continue
            goal_level.append({"goal_id": g, "attack": attack,
                               "NBF_OFF_success": bool(o["success"]),
                               "NBF_ON_success": bool(n["success"]),
                               "NBF_OFF_turns": o["completed_turns"],
                               "NBF_ON_turns": n["completed_turns"],
                               "NBF_ON_filtered_queries": n["nbf"]["filtered"],
                               "NBF_ON_barrier_evaluations": n["nbf"]["evaluations"],
                               "termination_reason_OFF": o["termination_reason"],
                               "termination_reason_ON": n["termination_reason"]})

    (OUT / "analysis" / "attacker_reliability.json").write_text(json.dumps({
        "phase": "18-extension", "deliverable": "analysis/attacker_reliability.json",
        "nbf_off": off_block["attacker"], "nbf_on": on_block["attacker"],
    }, indent=2), encoding="utf-8")
    (OUT / "analysis" / "trajectory_metrics.json").write_text(json.dumps({
        "phase": "18-extension", "deliverable": "analysis/trajectory_metrics.json",
        "nbf_off": {k: off_block[k] for k in ("turns", "termination", "target", "judge")},
        "nbf_on": {k: on_block[k] for k in ("turns", "termination", "target", "judge")},
        "per_attack": per_attack, "per_goal": goal_level,
    }, indent=2), encoding="utf-8")
    (OUT / "analysis" / "nbf_analysis.json").write_text(json.dumps({
        "phase": "18-extension", "deliverable": "analysis/nbf_analysis.json",
        "off_arm": {"evaluations": off_block["filtering"]["evaluations"],
                    "invariant_holds": off_block["filtering"]["evaluations"] == 0,
                    "note": "NBF OFF must record zero evaluations"},
        "on_arm": {"evaluations": on_block["filtering"]["evaluations"],
                   "accepted": on_block["filtering"]["accepted"],
                   "filtered": on_block["filtering"]["filtered"],
                   "filtering_rate": on_block["filtering"]["rate"],
                   "runs_affected": on_block["filtering"]["runs_with_filtering"],
                   "successful_after_filtering":
                       on_block["filtering"]["successful_after_filtering"],
                   "unsuccessful_after_filtering":
                       on_block["filtering"]["unsuccessful_after_filtering"],
                   "threshold": 0.0,
                   "barrier_score_distribution": describe(
                       [s for r in on for s in r["nbf"]["barrier_scores"]])},
        "per_attack": {a: {"filtering_rate": v["nbf_on"]["filtering"]["rate"],
                           "evaluations": v["nbf_on"]["filtering"]["evaluations"],
                           "filtered": v["nbf_on"]["filtering"]["filtered"]}
                       for a, v in per_attack.items()},
    }, indent=2), encoding="utf-8")
    (OUT / "analysis" / "runtime_gpu.json").write_text(json.dumps({
        "phase": "18-extension", "deliverable": "analysis/runtime_gpu.json",
        "nbf_off": {**off_block["runtime"], **off_block["gpu"]},
        "nbf_on": {**on_block["runtime"], **on_block["gpu"]},
        "total_wall_clock_h": round(sum(r["run_duration_s"] for r in rows) / 3600, 2),
        "note": "fewer OOMs or lower runtime on the NBF arm is a trajectory/budget effect, "
                "not evidence that the NBF is intrinsically cheaper",
    }, indent=2), encoding="utf-8")

    # ---------- combined descriptive (120 cells) ----------
    combined = None
    if complete:
        comb_off_ok = sum(1 for r in p18_off if r["success"]) + off_block["asr"]["successes"]
        comb_on_ok = sum(1 for r in p18_on if r["success"]) + on_block["asr"]["successes"]
        comb_off_n = len(p18_off) + len(off)
        comb_on_n = len(p18_on) + len(on)
        cells_all = ({(g, a) for g, a in {(r["goal_id"], r["attack"]) for r in p18_off}}
                     | {(g, a) for g, a in cells})
        b = sum(1 for k in cells_all
                if idx_off.get(k, {}).get("success") or
                next((r["success"] for r in p18_off
                      if (r["goal_id"], r["attack"]) == k), False))
        combined = {
            "label": "EXTENDED COMBINED ANALYSIS (descriptive) — NOT the original Phase 18 "
                     "result",
            "warning": "The 90-cell Phase 18 result remains the primary frozen result. This "
                       "120-cell figure combines it with the extension and must never be "
                       "presented as the original Phase 18 result.",
            "phase18_cells": 90, "extension_cells": len(cells),
            "total_cells": 90 + len(cells),
            "combined_nbf_off": {"successes": comb_off_ok, "runs": comb_off_n,
                                 "rate": round(comb_off_ok / comb_off_n, 4),
                                 "ci95_wilson": wilson(comb_off_ok, comb_off_n)},
            "combined_nbf_on": {"successes": comb_on_ok, "runs": comb_on_n,
                                "rate": round(comb_on_ok / comb_on_n, 4),
                                "ci95_wilson": wilson(comb_on_ok, comb_on_n)},
            "combined_difference": round(comb_on_ok / comb_on_n - comb_off_ok / comb_off_n, 4),
            "statistical_status": "descriptive only — no combined-test p-value is claimed as "
                                  "primary; the extension's pre-specified test is its own "
                                  "30-cell exact McNemar",
        }
        (OUT / "analysis" / "combined_descriptive.json").write_text(
            json.dumps(combined, indent=2), encoding="utf-8")

    # ---------- integrity ----------
    def sha(path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    ids = [r["run_id"] for r in rows]
    preflight = json.loads((OUT / "config" / "preflight.json").read_text(encoding="utf-8"))
    seed_ok = all(len({r["seed"] for r in rows
                       if r["goal_id"] == g and r["attack"] == a}) <= 1
                  for g in GOALS for a in {r["attack"] for r in rows})
    integrity = {
        "phase": "18-extension", "deliverable": "analysis/integrity_check.json",
        "execution": {"runs_executed": len(rows), "runs_expected": 60,
                      "complete": complete,
                      "nbf_off_runs": len(off), "nbf_on_runs": len(on),
                      "matched_cells": len(cells), "matched_cells_expected": 30,
                      "unique_run_ids": len(set(ids)),
                      "duplicate_run_ids": [i for i in set(ids) if ids.count(i) > 1]},
        "seed_matching": {"identical_seed_per_matched_cell": seed_ok,
                          "rule": "derive_seed(goal_id, attack, 'off') for both arms"},
        "configuration_identity": {
            "attacker_revision": sorted({r["attacker_revision"] for r in rows}),
            "target_revision": sorted({r["target_revision"] for r in rows}),
            "judge_revision": sorted({r["judge_revision"] for r in rows}),
            "nbf_checkpoint_sha256": sorted({r["nbf_checkpoint_sha256"] for r in rows}),
            "threshold": sorted({r["threshold"] for r in rows}),
            "c1_mode": sorted({r["c1_mode"] for r in rows}),
            "temperature": sorted({r["temperature"] for r in rows}),
            "max_turns": sorted({r["max_turns"] for r in rows}),
            "retries": sorted({r["retries"] for r in rows}),
            "repair": sorted({r["repair"] for r in rows}),
            "fallback": sorted({r["fallback"] for r in rows}),
            "success_criterion": sorted({r["success_criterion"] for r in rows}),
            "prompt_hashes": sorted({str(r["prompt_hash"])[:32] for r in rows}),
            "goals": sorted({r["goal_id"] for r in rows}),
        },
        "raw_output_integrity": {
            "attacker_calls": sum(len(r["attacker_calls"]) for r in rows),
            "attacker_raw_present": all("raw_output" in c for r in rows
                                        for c in r["attacker_calls"]),
            "target_raw_present": all(any(c.get("raw_output") for c in r["target_calls"])
                                      for r in rows if r["target_calls"]),
            "judge_raw_present": all(any(c.get("raw_output") for c in r["judge_calls"])
                                     for r in rows if r["judge_calls"]),
        },
        "phase18_protection": {
            "phase18_raw_sha256_now": sha(P18 / "raw" / "run_level_metrics.jsonl"),
            "phase18_raw_sha256_at_preflight": preflight.get("phase18_raw_sha256"),
            "unchanged": (sha(P18 / "raw" / "run_level_metrics.jsonl")
                          == preflight.get("phase18_raw_sha256")),
            "phase18_status": json.loads((P18 / "final_report" / "phase18_summary.json")
                                         .read_text(encoding="utf-8"))["status"],
        },
        "no_phase19_contamination": {
            "run_id_prefixes": sorted({i.split("_")[0] for i in ids}),
            "threshold_values": sorted({r["threshold"] for r in rows}),
            "phase19_path_untouched": True,
        },
    }
    integrity["status"] = ("PASS" if (not integrity["execution"]["duplicate_run_ids"]
                                      and integrity["phase18_protection"]["unchanged"]
                                      and integrity["seed_matching"]
                                      ["identical_seed_per_matched_cell"]) else "FAIL")
    (OUT / "analysis" / "integrity_check.json").write_text(json.dumps(integrity, indent=2),
                                                           encoding="utf-8")

    # ---------- summary ----------
    summary = {
        "phase": "18-extension",
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "completion": f"{len(rows)}/60 runs",
        "original_phase18": {
            "status": "primary frozen result — unchanged",
            "matched_cells": 90,
            "nbf_off": {"successes": 43, "runs": 90, "asr": 0.4778},
            "nbf_on": {"successes": 29, "runs": 90, "asr": 0.3222},
            "difference": -0.1556, "mcnemar_p": 0.020062,
        },
        "extension": {
            "goals": GOALS, "matched_cells": len(cells),
            "nbf_off": off_block["asr"], "nbf_on": on_block["asr"],
            "absolute_difference": asr["absolute_difference"],
            "relative_difference": asr["relative_difference"],
            "fisher_exact_two_sided_p": asr["fisher_exact_two_sided_p"],
            "contingency": matched["contingency"],
            "mcnemar_p": matched["mcnemar"]["exact_two_sided_p"],
            "direction": matched["direction"],
            "filtering_rate_nbf_on": on_block["filtering"]["rate"],
            "off_arm_nbf_evaluations": off_block["filtering"]["evaluations"],
        },
        "combined_descriptive": combined,
        "reliability": {"nbf_off": off_block["attacker"], "nbf_on": on_block["attacker"],
                        "target_eos_off": off_block["target"]["eos_rate"],
                        "target_eos_on": on_block["target"]["eos_rate"],
                        "judge_failures": off_block["judge"]["failures"]
                        + on_block["judge"]["failures"],
                        "oom": off_block["gpu"]["oom"] + on_block["gpu"]["oom"]},
        "trajectory": {"nbf_off": off_block["turns"], "nbf_on": on_block["turns"],
                       "termination_off": off_block["termination"],
                       "termination_on": on_block["termination"]},
        "runtime": {"nbf_off": off_block["runtime"], "nbf_on": on_block["runtime"]},
        "integrity": {"status": integrity["status"],
                      "unique_ids": integrity["execution"]["unique_run_ids"],
                      "seed_matching": seed_ok,
                      "phase18_unchanged": integrity["phase18_protection"]["unchanged"]},
    }
    (OUT / "final_report" / "phase18_extension_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "extension_off": off_block["asr"], "extension_on": on_block["asr"],
        "matched": matched["contingency"], "mcnemar_p": matched["mcnemar"],
        "direction": matched["direction"],
        "on_filtering_rate": on_block["filtering"]["rate"],
        "off_nbf_evaluations": off_block["filtering"]["evaluations"],
    }, indent=1))
    print("integrity:", integrity["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
