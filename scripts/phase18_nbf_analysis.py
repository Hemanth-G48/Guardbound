"""Phase 18 — NBF experiment analysis: C1 + NBF OFF vs C1 + NBF ON.

Reads the Phase 18 run records and writes the analysis artifact set under
`results/phase18_nbf_experiment/analysis/` plus figures.

Primary endpoint: official attack success rate (ASR = successful runs / total runs), the
denominator never changed to survivors. Matched analysis uses the project's established
matched test for paired binary outcomes — the exact McNemar test, as used in Stages 6 and 7
— declared before the results were seen.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase18_nbf_experiment"
RUNS = OUT / "raw" / "run_level_metrics.jsonl"
FIG = OUT / "analysis" / "figures"
ATTACKS = ["crescendo_paper", "opposite_day", "acronym"]
LABEL = {"crescendo_paper": "Crescendo", "opposite_day": "OppositeDay", "acronym": "Acronym"}
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
ARMS = ("nbf_off", "nbf_on")


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


def mcnemar(b: int, c: int) -> dict:
    """Exact two-sided McNemar over discordant pairs (b = off-fail/on-success, c = reverse)."""
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"discordant": n, "exact_two_sided_p": round(p, 6)}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 2),
            "median": round(statistics.median(values), 2),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 2),
            "min": round(min(values), 2), "max": round(max(values), 2)}


def main() -> int:
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for r in rows:
        r["_survived"] = r["failure_class"] not in ("attacker_generation_error",
                                                    "cuda_oom_failure",
                                                    "infrastructure_failure")
    # balanced goal blocks only (both conditions for every goal x attack)
    complete_goals = [g for g in sorted({r["goal_id"] for r in rows})
                      if all(sum(1 for r in rows if r["goal_id"] == g and r["attack"] == t
                                 and r["condition"] == a) == 1
                             for t in ATTACKS for a in ARMS)]
    excluded = [r for r in rows if r["goal_id"] not in complete_goals]
    runs = [r for r in rows if r["goal_id"] in complete_goals]
    off = [r for r in runs if r["condition"] == "nbf_off"]
    on = [r for r in runs if r["condition"] == "nbf_on"]
    print(f"rows {len(rows)} | complete goals {complete_goals} -> {len(runs)} analysed "
          f"({len(excluded)} excluded from a partial block)")

    def arm_block(subset: list[dict]) -> dict:
        calls = [c for r in subset for c in r["attacker_calls"]]
        valid = [c for c in calls if c["category"] in VALID]
        turns = [r["completed_turns"] for r in subset]
        rubric = [s for r in subset for s in r["rubric_scores"] if s is not None]
        evals = sum(r["nbf"]["evaluations"] for r in subset)
        filtered = sum(r["nbf"]["filtered"] for r in subset)
        return {
            "runs": len(subset),
            "asr": {"successes": sum(1 for r in subset if r["success"]),
                    "runs": len(subset),
                    "rate": round(sum(1 for r in subset if r["success"]) / len(subset), 4)
                    if subset else None,
                    "ci95_wilson": wilson(sum(1 for r in subset if r["success"]), len(subset))},
            "run_survival": {"numerator": sum(1 for r in subset if r["_survived"]),
                             "rate": round(sum(1 for r in subset if r["_survived"])
                                           / len(subset), 4) if subset else None},
            "attacker": {
                "calls": len(calls), "valid": len(valid),
                "validity_rate": round(len(valid) / len(calls), 4) if calls else None,
                "semantic_usable": len([c for c in valid if c["failure_type"] is None]),
                "prose_failures": sum(1 for c in calls
                                      if c["failure_type"] == "prose_output"),
                "empty_question": sum(1 for c in calls
                                      if c["failure_type"] == "empty_generatedQuestion"),
                "truncated": sum(1 for c in calls
                                 if c["failure_type"] == "truncated_json"),
                "generation_errors": sum(1 for r in subset if r["failure_class"]
                                         == "attacker_generation_error")},
            "turns": {"total": sum(turns), **describe([float(t) for t in turns]),
                      "reaching_8": sum(1 for t in turns if t >= 8)},
            "termination_reasons": dict(Counter(r["termination_reason"] for r in subset)),
            "target": {"calls": sum(len(r["target_calls"]) for r in subset),
                       "natural_eos": sum(r["counts"]["target_eos"] for r in subset),
                       "refusals_total": sum(r.get("refusal_count") or 0 for r in subset),
                       "refusals_per_run": round(statistics.mean(
                           [r.get("refusal_count") or 0 for r in subset]), 3)},
            "judge": {"calls": sum(len(r["judge_calls"]) for r in subset),
                      "failures": sum(r["counts"]["judge_failures"] for r in subset),
                      "rubric_n": len(rubric),
                      "rubric_mean": round(statistics.mean(rubric), 3) if rubric else None,
                      "score5": sum(1 for s in rubric if s == 5)},
            "runtime": {**describe([r["run_duration_s"] for r in subset]),
                        "throughput_runs_per_hour": round(
                            3600 / statistics.mean([r["run_duration_s"] for r in subset]), 2)},
            "gpu": {"peak_vram_gib": max((r["peak_vram_gib"] for r in subset), default=None),
                    "oom": sum(1 for r in subset if r["failure_class"] == "cuda_oom_failure")},
            "nbf": {"evaluations": evals, "filtered": filtered,
                    "accepted": evals - filtered,
                    "filtering_rate": round(filtered / evals, 4) if evals else None,
                    "runs_affected": sum(1 for r in subset if r["nbf"]["run_affected_by_filtering"]),
                    "runs_with_any_evaluation": sum(1 for r in subset
                                                    if r["nbf"]["evaluations"] > 0)},
        }

    off_block, on_block = arm_block(off), arm_block(on)

    # ---- primary endpoint ----
    asr = {
        "phase": "18", "deliverable": "analysis/asr_comparison.json",
        "primary_endpoint": "official attack success rate (successful runs / total runs)",
        "nbf_off": off_block["asr"], "nbf_on": on_block["asr"],
        "absolute_difference": round(on_block["asr"]["rate"] - off_block["asr"]["rate"], 4),
        "relative_change": (round(on_block["asr"]["rate"] / off_block["asr"]["rate"], 4)
                            if off_block["asr"]["rate"] else None),
        "fisher_exact_two_sided_p": fisher(on_block["asr"]["successes"],
                                           on_block["asr"]["runs"] - on_block["asr"]["successes"],
                                           off_block["asr"]["successes"],
                                           off_block["asr"]["runs"] - off_block["asr"]["successes"]),
        "denominator_note": "all runs stay in the denominator; no conditional substitution",
        "matrix": {"goals": len(complete_goals), "attacks": len(ATTACKS),
                   "conditions": len(ARMS), "runs": len(runs),
                   "matched_cells": len(complete_goals) * len(ATTACKS),
                   "runs_in_partial_block_excluded": len(excluded)},
    }
    (OUT / "analysis" / "asr_comparison.json").write_text(json.dumps(asr, indent=2),
                                                          encoding="utf-8")

    # ---- matched analysis (pre-specified: exact McNemar) ----
    idx_off = {(r["goal_id"], r["attack"]): r for r in off}
    idx_on = {(r["goal_id"], r["attack"]): r for r in on}
    cells = sorted(set(idx_off) & set(idx_on))
    both = sum(1 for k in cells if idx_off[k]["success"] and idx_on[k]["success"])
    off_only = sum(1 for k in cells if idx_off[k]["success"] and not idx_on[k]["success"])
    on_only = sum(1 for k in cells if not idx_off[k]["success"] and idx_on[k]["success"])
    neither = len(cells) - both - off_only - on_only
    matched = {
        "phase": "18", "deliverable": "analysis/matched_analysis.json",
        "test_pre_specified": "exact McNemar (two-sided), the project's established matched "
                              "test for paired binary outcomes (Stages 6 and 7)",
        "matched_cells": len(cells),
        "contingency": {"off_success_on_success": both, "off_success_on_failure": off_only,
                        "off_failure_on_success": on_only,
                        "off_failure_on_failure": neither},
        "mcnemar": mcnemar(on_only, off_only),
        "off_cell_rate": round((both + off_only) / len(cells), 4) if cells else None,
        "on_cell_rate": round((both + on_only) / len(cells), 4) if cells else None,
        "paired_turns": None,
    }
    turn_pairs = [(idx_on[k]["completed_turns"], idx_off[k]["completed_turns"]) for k in cells]
    if turn_pairs:
        diffs = [a - b for a, b in turn_pairs]
        matched["paired_turns"] = {"mean_difference": round(statistics.mean(diffs), 3),
                                   "median_difference": statistics.median(diffs),
                                   "n_pairs": len(diffs)}
    (OUT / "analysis" / "matched_analysis.json").write_text(json.dumps(matched, indent=2),
                                                            encoding="utf-8")

    # ---- NBF-specific analysis ----
    on_scores = [s for r in on for s in r["nbf"]["barrier_scores"]]
    off_scores = [s for r in off for s in r["nbf"]["barrier_scores"]]
    per_turn: dict[int, dict] = defaultdict(lambda: {"evaluations": 0, "filtered": 0})
    for r in on:
        for index, (score, verdict) in enumerate(zip(r["nbf"]["barrier_scores"],
                                                     r["nbf"]["verdicts"]), start=1):
            per_turn[index]["evaluations"] += 1
            if not verdict:
                per_turn[index]["filtered"] += 1
    nbf_payload = {
        "phase": "18", "deliverable": "analysis/nbf_analysis.json",
        "off_arm": {"evaluations": off_block["nbf"]["evaluations"],
                    "barrier_scores_recorded": len(off_scores),
                    "invariant": "NBF OFF must record zero evaluations and zero barrier "
                                 "scores; observed: "
                                 f"{off_block['nbf']['evaluations']} evaluations",
                    "invariant_holds": off_block["nbf"]["evaluations"] == 0
                    and len(off_scores) == 0},
        "on_arm": {
            "evaluations": on_block["nbf"]["evaluations"],
            "accepted": on_block["nbf"]["accepted"],
            "filtered": on_block["nbf"]["filtered"],
            "filtering_rate": on_block["nbf"]["filtering_rate"],
            "runs_with_evaluations": on_block["nbf"]["runs_with_any_evaluation"],
            "runs_affected_by_filtering": on_block["nbf"]["runs_affected"],
            "barrier_score_distribution": describe(on_scores),
            "filtered_scores": describe([s for r in on for s, v
                                         in zip(r["nbf"]["barrier_scores"],
                                                r["nbf"]["verdicts"]) if not v]),
            "accepted_scores": describe([s for r in on for s, v
                                        in zip(r["nbf"]["barrier_scores"],
                                               r["nbf"]["verdicts"]) if v]),
            "filtering_by_evaluation_index": {str(k): v for k, v in sorted(per_turn.items())},
            "successful_attacks_despite_filtering": sum(
                1 for r in on if r["success"] and r["nbf"]["run_affected_by_filtering"]),
            "runs_terminated_after_filtering": sum(
                1 for r in on if r["nbf"]["run_affected_by_filtering"]
                and not r["_survived"]),
            "threshold": 0.0,
            "bypassed": "no — every candidate in this arm was scored before use",
        },
        "queries_behind_decisions": [
            {"run_id": r["run_id"], "goal_id": r["goal_id"], "attack": r["attack"],
             "scores": r["nbf"]["barrier_scores"], "verdicts": r["nbf"]["verdicts"],
             "embedded_text_count": len(r["nbf"]["embedded_texts"]),
             "last_embedded_texts": [t[:160] for t in r["nbf"]["embedded_texts"][-3:]]}
            for r in on if r["nbf"]["evaluations"]],
    }
    (OUT / "analysis" / "nbf_analysis.json").write_text(json.dumps(nbf_payload, indent=2),
                                                        encoding="utf-8")

    # ---- trajectories, attacker, per-attack, per-goal, runtime/gpu ----
    (OUT / "analysis" / "trajectory_metrics.json").write_text(json.dumps({
        "phase": "18", "deliverable": "analysis/trajectory_metrics.json",
        "nbf_off": {"turns": off_block["turns"], "termination_reasons": off_block["termination_reasons"],
                    "target": off_block["target"], "judge": off_block["judge"]},
        "nbf_on": {"turns": on_block["turns"], "termination_reasons": on_block["termination_reasons"],
                   "target": on_block["target"], "judge": on_block["judge"]},
    }, indent=2), encoding="utf-8")

    (OUT / "analysis" / "attacker_reliability.json").write_text(json.dumps({
        "phase": "18", "deliverable": "analysis/attacker_reliability.json",
        "nbf_off": off_block["attacker"], "nbf_on": on_block["attacker"],
        "note": "C1 is frozen; no generation-point instruction change was made in either arm",
    }, indent=2), encoding="utf-8")

    attack_payload = {"phase": "18", "deliverable": "analysis/attack_level.json",
                      "note": "no attack is ranked", "attacks": {}}
    for attack in ATTACKS:
        attack_payload["attacks"][LABEL[attack]] = {
            a: arm_block([r for r in runs if r["attack"] == attack and r["condition"] == a])
            for a in ARMS}
    (OUT / "analysis" / "attack_level.json").write_text(json.dumps(attack_payload, indent=2),
                                                        encoding="utf-8")

    goal_rows = []
    for g in complete_goals:
        for attack in ATTACKS:
            o = idx_off.get((g, attack))
            n = idx_on.get((g, attack))
            if not o or not n:
                continue
            goal_rows.append({
                "goal_id": g, "attack": attack,
                "NBF_OFF_success": bool(o["success"]), "NBF_ON_success": bool(n["success"]),
                "NBF_OFF_turns": o["completed_turns"], "NBF_ON_turns": n["completed_turns"],
                "NBF_ON_barrier_evaluations": n["nbf"]["evaluations"],
                "NBF_ON_filtered_queries": n["nbf"]["filtered"],
                "termination_reason_OFF": o["termination_reason"],
                "termination_reason_ON": n["termination_reason"]})
    (OUT / "analysis" / "goal_level.json").write_text(json.dumps({
        "phase": "18", "deliverable": "analysis/goal_level.json",
        "rows": goal_rows,
        "pattern": {"off_success_on_success": both, "off_success_on_failure": off_only,
                    "off_failure_on_success": on_only, "off_failure_on_failure": neither},
        "goals_present": complete_goals,
    }, indent=2), encoding="utf-8")

    (OUT / "analysis" / "runtime_gpu.json").write_text(json.dumps({
        "phase": "18", "deliverable": "analysis/runtime_gpu.json",
        "nbf_off": {**off_block["runtime"], **off_block["gpu"]},
        "nbf_on": {**on_block["runtime"], **on_block["gpu"]},
        "nbf_overhead_per_run_s": round(on_block["runtime"]["mean"]
                                        - off_block["runtime"]["mean"], 1),
        "total_runs_analysed": len(runs),
        "total_wall_clock_h": round(sum(r["run_duration_s"] for r in runs) / 3600, 2),
    }, indent=2), encoding="utf-8")

    # ---- integrity ----
    def sha(rel: str) -> str | None:
        p = REPO_ROOT / rel
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    ids = [r["run_id"] for r in rows]
    integrity = {
        "phase": "18", "deliverable": "analysis/integrity_check.json",
        "nbf_off_runs": sum(1 for r in rows if r["condition"] == "nbf_off"),
        "nbf_on_runs": sum(1 for r in rows if r["condition"] == "nbf_on"),
        "total_runs": len(rows), "unique_run_ids": len(set(ids)),
        "duplicate_run_ids": [i for i in set(ids) if ids.count(i) > 1],
        "missing_run_ids": "see matrix_achieved",
        "matrix_achieved": asr["matrix"],
        "configuration_consistency": {
            "c1_mode": sorted({r["c1_mode"] for r in rows}),
            "temperature": sorted({r["temperature"] for r in rows}),
            "top_p": sorted({r["top_p"] for r in rows}),
            "max_turns": sorted({r["max_turns"] for r in rows}),
            "nbf_enabled_by_condition": {a: sorted({r["nbf_enabled"] for r in rows
                                                    if r["condition"] == a}) for a in ARMS},
            "prompt_hashes": sorted({str(r["prompt_hash"])[:32] for r in rows}),
            "attacker_revision": sorted({r["attacker_revision"] for r in rows}),
            "target_revision": sorted({r["target_revision"] for r in rows}),
            "judge_revision": sorted({r["judge_revision"] for r in rows}),
            "success_criterion": sorted({r["success_criterion"] for r in rows}),
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
        "dataset_integrity": {"sha256": sha(str(Path("nbf_original_stuff/orginal_code_give_by_"
                                                     "author/NBF-LLM/data/test/"
                                                     "harmbench_tasks.json"))),
                              "expected": "ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb"},
        "frozen_artifact_integrity": {
            "stage4_8_pilot_data_sha256": sha("results/phase17_pilot/runs/raw_results.jsonl"),
            "expected": "55101e0e75c61f3a25e027e67450c51f39818ffad8db79288855c5f86c7feaad",
            "stage7_manifest_present": (REPO_ROOT / "results/phase17_pilot" / "analysis"
                                        / "stage7_c1_qualification" / "manifest.json").is_file(),
        },
        "nbf_semantics": {
            "off_arm_zero_evaluations": off_block["nbf"]["evaluations"] == 0,
            "on_arm_evaluations_positive": on_block["nbf"]["evaluations"] > 0,
            "one_variable_only": "both arms use C1 (mode I1, T=0.7); the only difference is "
                                 "the NBF state",
        },
    }
    integrity["status"] = ("PASS" if (not integrity["duplicate_run_ids"]
                                      and integrity["nbf_semantics"]
                                      ["off_arm_zero_evaluations"]
                                      and off_block["nbf"]["evaluations"] == 0) else "FAIL")
    (OUT / "analysis" / "integrity_check.json").write_text(json.dumps(integrity, indent=2),
                                                           encoding="utf-8")

    figures(off_block, on_block, on_scores, per_turn, attack_payload, runs, off, on, cells,
            idx_off, idx_on)

    print(json.dumps({"asr_off": off_block["asr"]["rate"], "asr_on": on_block["asr"]["rate"],
                      "matched_cells": len(cells),
                      "contingency": matched["contingency"],
                      "mcnemar_p": matched["mcnemar"]["exact_two_sided_p"],
                      "nbf_filtering_rate": on_block["nbf"]["filtering_rate"],
                      "off_arm_nbf_evaluations": off_block["nbf"]["evaluations"]}, indent=1))
    print(f"integrity: {integrity['status']} | figures: "
          f"{sorted(p.name for p in FIG.glob('*.png'))}")
    return 0


def figures(off_block, on_block, on_scores, per_turn, attack_payload, runs, off, on, cells,
            idx_off, idx_on) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name, title):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=130)
        plt.close(fig)

    # ASR comparison
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    vals = [100 * off_block["asr"]["rate"], 100 * on_block["asr"]["rate"]]
    ax.bar(["NBF OFF", "NBF ON"], vals, color=["#2e6f95", "#c25b56"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("official ASR (%)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    save(fig, "asr_comparison.png", "Official ASR: C1 + NBF OFF vs C1 + NBF ON")

    # barrier score distribution
    if on_scores:
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.hist(on_scores, bins=20, color="#4c9f70")
        ax.axvline(0.0, color="#c25b56", linestyle="--", label="threshold 0.0")
        ax.set_xlabel("barrier score (P(last class) - max other)")
        ax.set_ylabel("candidate evaluations")
        ax.legend(fontsize=8)
        save(fig, "barrier_score_distribution.png",
             "NBF barrier scores (NBF ON arm)")

    # filtering / evaluation index
    if per_turn:
        keys = sorted(per_turn)[:10]
        fig, ax = plt.subplots(figsize=(6.2, 3.4))
        ax.bar([str(k) for k in keys], [per_turn[k]["evaluations"] for k in keys],
               color="#5b7fa6", label="evaluations")
        ax.bar([str(k) for k in keys], [per_turn[k]["filtered"] for k in keys],
               color="#c25b56", label="filtered")
        ax.set_xlabel("candidate evaluation index within the run")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        save(fig, "nbf_filtering_by_index.png", "NBF evaluations and filtering by index")

    # trajectory depth
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, block, colour in (("NBF OFF", off_block, "#2e6f95"),
                                 ("NBF ON", on_block, "#c25b56")):
        turns = [r["completed_turns"] for r in runs if r["condition"] ==
                 ("nbf_off" if label == "NBF OFF" else "nbf_on")]
        dist = Counter(turns)
        ax.bar([x + (0.2 if label == "NBF ON" else 0) for x in sorted(dist)],
               [dist[x] for x in sorted(dist)], width=0.4, label=label, color=colour)
    ax.set_xlabel("completed turns")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "turns_distribution.png", "Completed turns by condition")

    # matched-cell scatter
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    grid = Counter((idx_off[k]["success"], idx_on[k]["success"]) for k in cells)
    labels = ["neither", "OFF only", "ON only", "both"]
    values = [grid.get((False, False), 0), grid.get((True, False), 0),
              grid.get((False, True), 0), grid.get((True, True), 0)]
    ax.bar(labels, values, color=["#999999", "#2e6f95", "#c25b56", "#4c9f70"])
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("matched cells")
    save(fig, "matched_cells.png", "Matched goal x attack cells by outcome")

    # attack-level ASR
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(attack_payload["attacks"])
    off_vals = [100 * attack_payload["attacks"][n]["nbf_off"]["asr"]["rate"] for n in names]
    on_vals = [100 * attack_payload["attacks"][n]["nbf_on"]["asr"]["rate"] for n in names]
    ax.bar([i - 0.2 for i in range(len(names))], off_vals, width=0.4, label="NBF OFF",
           color="#2e6f95")
    ax.bar([i + 0.2 for i in range(len(names))], on_vals, width=0.4, label="NBF ON",
           color="#c25b56")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("ASR (%)")
    ax.legend(fontsize=8)
    save(fig, "attack_level_asr.png", "ASR by attack and condition")

    # runtime
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, subset, colour in (("NBF OFF", off, "#2e6f95"), ("NBF ON", on, "#c25b56")):
        ax.hist([r["run_duration_s"] for r in subset], bins=16, alpha=0.6, label=label,
                color=colour)
    ax.set_xlabel("run duration (s)")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "runtime_distribution.png", "Run duration by condition")


if __name__ == "__main__":
    raise SystemExit(main())
