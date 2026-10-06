"""Phase 17 Stage 4.6 — B2 qualification analysis.

Reads the qualification runs and produces the artifacts the stage brief names:
reliability rates with confidence intervals and exact tests, the failure
taxonomy with raw outputs preserved, paired-cell analysis on genuinely matched
cells, attack-capability comparison against the control, and the pilot-budget
result.

Semantic metrics are computed by importing the Stage 4.5 functions, so both arms
of this qualification and the Stage 4.5 study are measured by identical code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase17_stage45_common", REPO_ROOT / "scripts" / "phase17_stage45_common.py")
common = importlib.util.module_from_spec(_spec)
sys.modules["phase17_stage45_common"] = common
_spec.loader.exec_module(common)

_spec45 = importlib.util.spec_from_file_location(
    "phase17_stage45_analysis", REPO_ROOT / "scripts" / "phase17_stage45_analysis.py")
a45 = importlib.util.module_from_spec(_spec45)
sys.modules["phase17_stage45_analysis"] = a45
_spec45.loader.exec_module(a45)

OUT = REPO_ROOT / "results" / "phase17_model_optimization" / "stage4_6_b2_qualification"
RESULTS = OUT / "multiturn_results.jsonl"
STAGE4 = REPO_ROOT / "results" / "phase17_model_qualification" / "stage4_qwen38_reliability"
STAGE45 = REPO_ROOT / "results" / "phase17_model_optimization" / "stage4_5_qwen38"
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
ARMS = ("control", "b2")
ARM_PROMPT_VARIANT = {"control": "A0_control", "b2": "A1_json_object_statement"}

wilson = a45.wilson_interval


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact two-sided Fisher exact test for the 2x2 table [[a, b], [c, d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(_log_comb(r1, x) + _log_comb(n - r1, c1 - x) - _log_comb(n, c1))

    observed = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    total = sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed + 1e-12)
    return round(min(1.0, total), 4)


fisher = fisher_two_sided


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def risk_difference(a_s: int, a_n: int, b_s: int, b_n: int) -> dict:
    if not a_n or not b_n:
        return {}
    pa, pb = a_s / a_n, b_s / b_n
    diff = pb - pa
    se = math.sqrt(pa * (1 - pa) / a_n + pb * (1 - pb) / b_n)
    # Haldane-Anscombe correction for the odds ratio when a cell is zero.
    aa, bb = a_s + 0.5, a_n - a_s + 0.5
    cc, dd = b_s + 0.5, b_n - b_s + 0.5
    odds = (bb * cc) / (aa * dd)
    return {
        "control_rate": round(pa, 4), "b2_rate": round(pb, 4),
        "absolute_difference": round(diff, 4),
        "difference_ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
        "odds_ratio": round(odds, 4),
        "cohens_h": round(2 * math.asin(math.sqrt(pb)) - 2 * math.asin(math.sqrt(pa)), 4),
        "fisher_exact_two_sided_p": fisher(a_s, a_n - a_s, b_s, b_n - b_s),
    }


def _judge_types(rows: list[dict]) -> dict:
    """Judge call outcomes: the refusal judge returns a dict, the rubric returns a dict too."""
    counts = Counter(c["returned_type"] for r in rows
                     for c in r.get("judge_calls_detail", []))
    unusable = sum(1 for r in rows for c in r.get("judge_calls_detail", [])
                   if c["returned_type"] != "dict")
    out = dict(counts)
    out["unusable_return_types"] = unusable
    return out


def rate_block(rows: list[dict], key: str) -> dict:
    n = len(rows)
    s = sum(1 for r in rows if r.get(key))
    return {"numerator": s, "denominator": n,
            "rate": round(s / n, 4) if n else None,
            "rate_pct": f"{100 * s / n:.1f}%" if n else None,
            "ci95_wilson": wilson(s, n)}


def arm_block(rows: list[dict]) -> dict:
    calls = [c for r in rows for c in r["attacker_calls"]]
    valid_calls = [c for c in calls if c["category"] in VALID]
    usable_calls = [c for c in valid_calls if c["failure_type"] is None]
    failed_calls = [c for c in calls if c["failure_type"] is not None]
    latencies = sorted(c["latency_s"] for c in calls if c.get("latency_s") is not None)
    events = [e for r in rows for e in r.get("rubric_scores", [])]
    scores = [e["score"] for e in events if e.get("valid")]
    return {
        "runs": len(rows),
        "run_survival": rate_block(rows, "run_survived"),
        "usable_run_strict": rate_block(rows, "usable_run_strict"),
        "usable_run_any_turn": rate_block(rows, "usable_run_any_turn"),
        "full_turn_budget": {
            "numerator": sum(1 for r in rows if r["termination_mode"] == "full_turn_budget"),
            "denominator": len(rows),
            "rate": round(sum(1 for r in rows if r["termination_mode"] == "full_turn_budget")
                          / len(rows), 4) if rows else None,
            "ci95_wilson": wilson(sum(1 for r in rows
                                      if r["termination_mode"] == "full_turn_budget"), len(rows)),
        },
        "attacker_call_validity": {
            "numerator": len(valid_calls), "denominator": len(calls),
            "rate": round(len(valid_calls) / len(calls), 4) if calls else None,
            "ci95_wilson": wilson(len(valid_calls), len(calls)),
            "direct_json": sum(1 for c in valid_calls if c["category"] == "VALID_DIRECT_JSON"),
            "frozen_extraction": sum(1 for c in valid_calls if c["category"] == "VALID_FROZEN_EXTRACTION"),
        },
        "attacker_call_semantic_usability": {
            "numerator": len(usable_calls), "denominator": len(calls),
            "rate": round(len(usable_calls) / len(calls), 4) if calls else None,
            "ci95_wilson": wilson(len(usable_calls), len(calls)),
        },
        "attacker_calls": len(calls),
        "attacker_calls_failed": len(failed_calls),
        "failure_types": dict(Counter(c["failure_type"] for c in failed_calls)),
        "termination_modes": dict(Counter(r["termination_mode"] for r in rows)),
        "turns": {
            "completed_total": sum(r["turns_completed"] for r in rows),
            "mean_completed": round(statistics.mean(r["turns_completed"] for r in rows), 3),
            "max": max(r["turns_completed"] for r in rows),
        },
        "target_calls": sum(r["target_calls"] for r in rows),
        "target_eos": sum(r["target_eos_count"] for r in rows),
        "judge_calls": sum(r["judge_calls"] for r in rows),
        "judge_failures": sum(r["judge_failures"] for r in rows),
        "judge_call_types": _judge_types(rows),
        "rubric_scores": {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "median": statistics.median(scores) if scores else None,
            "histogram": dict(sorted(Counter(scores).items())),
            "count_score_5": sum(1 for s in scores if s == 5),
            "count_score_1": sum(1 for s in scores if s == 1),
        },
        "latency_s": {
            "n": len(latencies),
            "mean": round(statistics.mean(latencies), 3) if latencies else None,
            "median": round(statistics.median(latencies), 3) if latencies else None,
            "p95": round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "wall_s": {
            "mean": round(statistics.mean(r["wall_s"] for r in rows), 1),
            "median": round(statistics.median(r["wall_s"] for r in rows), 1),
            "p95": round(sorted(r["wall_s"] for r in rows)[int(0.95 * (len(rows) - 1))], 1),
        },
        "peak_vram_gib": max(r["peak_vram_gib"] for r in rows),
        "load_seconds": rows[0]["load_seconds"],
        "generated_tokens": {
            "mean": round(statistics.mean(c["generated_tokens"] for c in calls
                                          if c.get("generated_tokens")), 1),
            "max": max(c["generated_tokens"] for c in calls if c.get("generated_tokens")),
        },
        "prompt_tokens": {
            "mean": round(statistics.mean(c["prompt_tokens"] for c in calls
                                          if c.get("prompt_tokens")), 1),
            "max": max(c["prompt_tokens"] for c in calls if c.get("prompt_tokens")),
        },
    }


def per_attack(rows: list[dict]) -> dict:
    out = {}
    for attack in common.ATTACKS:
        subset = [r for r in rows if r["attack"] == attack]
        if not subset:
            continue
        block = arm_block(subset)
        out[attack] = {
            "runs": block["runs"],
            "survived": block["run_survival"]["numerator"],
            "usable_strict": block["usable_run_strict"]["numerator"],
            "calls_valid": block["attacker_call_validity"]["numerator"],
            "calls": block["attacker_calls"],
            "failures": block["failure_types"],
            "mean_turns": block["turns"]["mean_completed"],
        }
    return out


def mcnemar(control: list[bool], b2: list[bool]) -> dict:
    """Exact McNemar on paired binary outcomes."""
    b = sum(1 for c, t in zip(control, b2) if (not c) and t)   # control fail, b2 pass
    c_ = sum(1 for c, t in zip(control, b2) if c and (not t))  # control pass, b2 fail
    n = b + c_
    if n == 0:
        p = 1.0
    else:
        # two-sided exact binomial
        k = min(b, c_)
        p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
        p = min(1.0, p)
    return {"control_fail_b2_pass": b, "control_pass_b2_fail": c_,
            "discordant_pairs": n, "exact_two_sided_p": round(p, 4)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(RESULTS)
    if not rows:
        print("no qualification runs yet")
        return 0
    phases = sorted({r["phase"] for r in rows})
    by_phase = {p: [r for r in rows if r["phase"] == p] for p in phases}
    print(f"phases: {phases} | runs: {len(rows)}")

    # ---------------- multiturn_summary.json ---------------- #
    summary = {
        "phase": "17", "stage": "4.6",
        "deliverable": "multiturn_summary.json",
        "definition": {
            "run_survival": "run reached its configured budget without an attacker-side "
                            "termination (invalid/unusable attacker output or generation failure)",
            "usable_run_strict": "brief §8.2 — at least one attacker step occurred AND every "
                                 "attacker step in the run was semantically usable",
            "usable_run_any_turn": "continuity metric used by Stages 4/4.5 — at least one usable turn",
            "semantically_usable_call": "valid JSON contract AND a non-empty string "
                                        "generatedQuestion AND a string lastResponseSummary",
        },
        "phases": {},
    }
    for phase, subset in by_phase.items():
        summary["phases"][phase] = {
            "max_turns": subset[0]["max_turns"],
            "arms": {arm: arm_block([r for r in subset if r["arm"] == arm]) for arm in ARMS
                     if any(r["arm"] == arm for r in subset)},
            "per_attack": {arm: per_attack([r for r in subset if r["arm"] == arm])
                           for arm in ARMS if any(r["arm"] == arm for r in subset)},
        }
    (OUT / "multiturn_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    primary_phase = min(phases, key=lambda p: by_phase[p][0]["max_turns"])
    p_rows = by_phase[primary_phase]
    control_rows = [r for r in p_rows if r["arm"] == "control"]
    b2_rows = [r for r in p_rows if r["arm"] == "b2"]
    if not control_rows or not b2_rows:
        summary_path = OUT / "multiturn_summary.json"
        print(f"only one arm present in phase {primary_phase} "
              f"(control {len(control_rows)}, b2 {len(b2_rows)}); wrote {summary_path.name} "
              f"and waiting for the other arm")
        return 0

    # ---------------- failure_analysis.json ---------------- #
    failures = []
    for row in p_rows:
        for index, call in enumerate(row["attacker_calls"]):
            if call["failure_type"] is None:
                continue
            failures.append({
                "candidate": row["arm"],
                "run_id": row["run_id"],
                "attack": row["attack"],
                "goal": row["goal"],
                "round": index + 1,
                "call_index": index,
                "seed": (row["attacker_seeds"][index]
                         if index < len(row["attacker_seeds"]) else None),
                "failure_type": call["failure_type"],
                "category": call["category"],
                "failure_shape": call["failure_shape"],
                "raw_output": call["raw_output"],
                "prompt_hash": call["prompt_hash"],
                "generation_parameters": {
                    "temperature": call["temperature"],
                    "top_p": row["configuration"]["top_p"],
                    "max_new_tokens": None,
                    "generated_tokens": call["generated_tokens"],
                    "termination": call["termination"],
                    "enable_thinking": False,
                },
            })
    (OUT / "failure_analysis.json").write_text(json.dumps({
        "phase": "17", "stage": "4.6", "deliverable": "failure_analysis.json",
        "phase_analysed": primary_phase,
        "taxonomy": ["prose_output", "malformed_json", "unquoted_json_key", "truncated_json",
                     "empty_output", "empty_generatedQuestion", "missing_generatedQuestion",
                     "missing_lastResponseSummary", "wrong_field_type", "other_reasoning_only"],
        "counts_by_arm": {arm: dict(Counter(f["failure_type"] for f in failures
                                            if f["candidate"] == arm)) for arm in ARMS},
        "counts_by_attack": {arm: dict(Counter(f["attack"] for f in failures
                                               if f["candidate"] == arm)) for arm in ARMS},
        "counts_by_round": {arm: dict(Counter(f["round"] for f in failures
                                              if f["candidate"] == arm)) for arm in ARMS},
        "total_failures": len(failures),
        "raw_outputs_preserved": True,
        "failures": failures,
    }, indent=2), encoding="utf-8")

    # ---------------- semantic_analysis.json ---------------- #
    semantics = {"phase": "17", "stage": "4.6", "deliverable": "semantic_analysis.json",
                 "method": ("identical code for both arms; figures are descriptive and no "
                            "quality score is derived from them"),
                 "phase_analysed": primary_phase, "arms": {}}
    for arm in ARMS:
        subset = [r for r in p_rows if r["arm"] == arm]
        questions = [c["generated_question"] for r in subset for c in r["attacker_calls"]
                     if isinstance(c.get("generated_question"), str) and c["generated_question"].strip()]
        flat_rows = [{"generated_question": c["generated_question"], "goal": r["goal"],
                      "attack": r["attack"], "round": index + 1}
                     for r in subset for index, c in enumerate(r["attacker_calls"])
                     if isinstance(c.get("generated_question"), str) and c["generated_question"].strip()]
        semantics["arms"][arm] = {
            "n_questions": len(questions),
            "question_metrics": a45.question_metrics(flat_rows),
            "diversity_metrics": a45.diversity_metrics(flat_rows),
            "per_attack_questions": {attack: len([q for q, r in
                                                  zip(questions, [x for x in flat_rows])
                                                  if r["attack"] == attack])
                                     for attack in common.ATTACKS},
            "target_progression": None,
            "inspection_sample": [],
        }
        # §9.6 target-response progression via the project's own rubric evaluator.
        events = [e for r in subset for e in r.get("rubric_scores", [])]
        by_turn: dict[int, list[int]] = {}
        turn = 0
        for row in subset:
            for index, score in enumerate([e["score"] for e in row.get("rubric_scores", [])
                                           if e.get("valid")]):
                by_turn.setdefault(index + 1, []).append(score)
        scores = [e["score"] for e in events if e.get("valid")]
        semantics["arms"][arm]["target_progression"] = {
            "evaluator": "project rubric evaluator (Qi et al. 2023 policy) via "
                         "guardbound.attacks.rubric_evaluation telemetry",
            "n_scores": len(scores),
            "mean": round(statistics.mean(scores), 3) if scores else None,
            "histogram": dict(sorted(Counter(scores).items())),
            "count_score_5": sum(1 for s in scores if s == 5),
            "per_turn_mean": {str(k): round(statistics.mean(v), 3)
                              for k, v in sorted(by_turn.items())},
            "note": "the rubric score measures the TARGET's response, not the attacker's "
                    "question; it is the project's existing criterion and no new one is added",
        }
        # deterministic inspection sample: first two rounds of each attack
        for attack in common.ATTACKS:
            taken = 0
            for row in subset:
                if row["attack"] != attack or taken >= 4:
                    continue
                for index, call in enumerate(row["attacker_calls"][:2]):
                    if isinstance(call.get("generated_question"), str) and call["generated_question"].strip():
                        semantics["arms"][arm]["inspection_sample"].append({
                            "run_id": row["run_id"], "round": index + 1,
                            "goal": row["goal"],
                            "question": call["generated_question"],
                            "lastResponseSummary": call.get("last_response_summary"),
                        })
                        taken += 1
    (OUT / "semantic_analysis.json").write_text(json.dumps(semantics, indent=2), encoding="utf-8")

    # ---------------- capability_comparison (paired questions) ---------------- #
    paired_cells = []
    for c_row, b_row in zip(sorted(control_rows, key=lambda r: r["run_index"]),
                            sorted(b2_rows, key=lambda r: r["run_index"])):
        if c_row["attack"] != b_row["attack"] or c_row["goal"] != b_row["goal"]:
            continue
        for index in range(min(len(c_row["attacker_calls"]), len(b_row["attacker_calls"]))):
            cc, bc = c_row["attacker_calls"][index], b_row["attacker_calls"][index]
            paired_cells.append({
                "run_index": c_row["run_index"], "attack": c_row["attack"],
                "round": index + 1,
                "seed_control": c_row["attacker_seeds"][index] if index < len(c_row["attacker_seeds"]) else None,
                "seed_b2": b_row["attacker_seeds"][index] if index < len(b_row["attacker_seeds"]) else None,
                "control_category": cc["category"], "b2_category": bc["category"],
                "control_failure_type": cc["failure_type"], "b2_failure_type": bc["failure_type"],
                "control_usable": cc["category"] in VALID and cc["failure_type"] is None,
                "b2_usable": bc["category"] in VALID and bc["failure_type"] is None,
                "control_question": cc.get("generated_question"),
                "b2_question": bc.get("generated_question"),
                "identical_question": (
                    isinstance(cc.get("generated_question"), str)
                    and cc.get("generated_question") == bc.get("generated_question")),
                "control_question_chars": len(cc["generated_question"]) if isinstance(cc.get("generated_question"), str) else None,
                "b2_question_chars": len(bc["generated_question"]) if isinstance(bc.get("generated_question"), str) else None,
            })
    matched = len(paired_cells)
    both_usable = sum(1 for c in paired_cells if c["control_usable"] and c["b2_usable"])
    only_b2 = sum(1 for c in paired_cells if not c["control_usable"] and c["b2_usable"])
    only_control = sum(1 for c in paired_cells if c["control_usable"] and not c["b2_usable"])
    neither = matched - both_usable - only_b2 - only_control
    lengths_c = [c["control_question_chars"] for c in paired_cells if c["control_question_chars"]]
    lengths_b = [c["b2_question_chars"] for c in paired_cells if c["b2_question_chars"]]
    capability = {
        "phase": "17", "stage": "4.6", "deliverable": "capability_comparison.json",
        "paired_cells": {
            "matched_cells": matched,
            "both_usable": both_usable, "only_b2_usable": only_b2,
            "only_control_usable": only_control, "neither_usable": neither,
            "identical_questions": sum(1 for c in paired_cells if c["identical_question"]),
            "mcnemar_exact_p": mcnemar([c["control_usable"] for c in paired_cells],
                                       [c["b2_usable"] for c in paired_cells]),
            "control_question_chars_mean": round(statistics.mean(lengths_c), 1) if lengths_c else None,
            "b2_question_chars_mean": round(statistics.mean(lengths_b), 1) if lengths_b else None,
            "note": ("cells are genuinely matched: identical attack, goal and seed schedule; the "
                     "two arms diverge once sampling assigns different tokens, so 'identical' is "
                     "not expected"),
        },
        "arms": {arm: {"question_metrics": semantics["arms"][arm]["question_metrics"],
                       "diversity_metrics": semantics["arms"][arm]["diversity_metrics"]}
                 for arm in ARMS},
    }
    (OUT / "analysis" / "capability_comparison.json").write_text(
        json.dumps(capability, indent=2), encoding="utf-8")

    # ---------------- statistical_analysis.json ---------------- #
    c_block, b_block = arm_block(control_rows), arm_block(b2_rows)
    stats = {
        "phase": "17", "stage": "4.6", "deliverable": "statistical_analysis.json",
        "phase_analysed": primary_phase,
        "interval_method": "Wilson score interval, 95 %",
        "test": "Fisher exact, two-sided (independent arms); McNemar exact (paired cells)",
        "primary_rates": {
            "run_survival": risk_difference(
                c_block["run_survival"]["numerator"], c_block["run_survival"]["denominator"],
                b_block["run_survival"]["numerator"], b_block["run_survival"]["denominator"]),
            "usable_run_strict": risk_difference(
                c_block["usable_run_strict"]["numerator"], c_block["usable_run_strict"]["denominator"],
                b_block["usable_run_strict"]["numerator"], b_block["usable_run_strict"]["denominator"]),
            "usable_run_any_turn": risk_difference(
                c_block["usable_run_any_turn"]["numerator"], c_block["usable_run_any_turn"]["denominator"],
                b_block["usable_run_any_turn"]["numerator"], b_block["usable_run_any_turn"]["denominator"]),
            "attacker_call_validity": risk_difference(
                c_block["attacker_call_validity"]["numerator"], c_block["attacker_call_validity"]["denominator"],
                b_block["attacker_call_validity"]["numerator"], b_block["attacker_call_validity"]["denominator"]),
            "attacker_call_semantic_usability": risk_difference(
                c_block["attacker_call_semantic_usability"]["numerator"],
                c_block["attacker_call_semantic_usability"]["denominator"],
                b_block["attacker_call_semantic_usability"]["numerator"],
                b_block["attacker_call_semantic_usability"]["denominator"]),
        },
        "arm_rates": {"control": c_block, "b2": b_block},
        "interpretation_rule": ("differences are read together with their intervals and effect "
                               "sizes; a p-value below 0.05 is not treated as proof of general "
                               "superiority, and a non-significant difference is not treated as "
                               "evidence of equivalence"),
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---------------- paired_results.json ---------------- #
    paired_runs = []
    for c_row, b_row in zip(sorted(control_rows, key=lambda r: r["run_index"]),
                            sorted(b2_rows, key=lambda r: r["run_index"])):
        if c_row["attack"] != b_row["attack"] or c_row["goal"] != b_row["goal"]:
            continue
        paired_runs.append({
            "run_index": c_row["run_index"], "attack": c_row["attack"], "goal": c_row["goal"],
            "control_survived": c_row["run_survived"], "b2_survived": b_row["run_survived"],
            "control_usable_strict": c_row["usable_run_strict"],
            "b2_usable_strict": b_row["usable_run_strict"],
            "control_turns": c_row["turns_completed"], "b2_turns": b_row["turns_completed"],
            "control_calls_valid": c_row["attacker_calls_valid"],
            "control_calls": c_row["attacker_calls_attempted"],
            "b2_calls_valid": b_row["attacker_calls_valid"],
            "b2_calls": b_row["attacker_calls_attempted"],
            "control_termination": c_row["termination_mode"],
            "b2_termination": b_row["termination_mode"],
            "paired_difference_turns": b_row["turns_completed"] - c_row["turns_completed"],
        })
    paired_payload = {
        "phase": "17", "stage": "4.6", "deliverable": "paired_results.json",
        "pairing_basis": ("identical run_index → identical attack, goal and attacker seed "
                          "schedule in both arms; genuinely matched by construction, not "
                          "reconstructed after the fact"),
        "pairs": len(paired_runs),
        "mcnemar": {
            "run_survival": mcnemar([p["control_survived"] for p in paired_runs],
                                    [p["b2_survived"] for p in paired_runs]),
            "usable_run_strict": mcnemar([p["control_usable_strict"] for p in paired_runs],
                                         [p["b2_usable_strict"] for p in paired_runs]),
        },
        "turn_difference": {
            "mean_paired_difference": round(statistics.mean(
                p["paired_difference_turns"] for p in paired_runs), 3),
            "b2_minus_control": sum(p["paired_difference_turns"] for p in paired_runs),
        },
        "per_run": paired_runs,
    }
    (OUT / "paired_results.json").write_text(json.dumps(paired_payload, indent=2), encoding="utf-8")
    (OUT / "analysis" / "paired_cells.json").write_text(json.dumps({
        "phase": "17", "stage": "4.6", "deliverable": "paired_cells.json",
        "cells": paired_cells,
        "summary": capability["paired_cells"],
    }, indent=2), encoding="utf-8")

    # ---------------- confidence_intervals.json ---------------- #
    (OUT / "analysis" / "confidence_intervals.json").write_text(json.dumps({
        "phase": "17", "stage": "4.6", "deliverable": "confidence_intervals.json",
        "interval_method": "Wilson score interval, 95 %",
        "phase_analysed": primary_phase,
        "intervals": {
            "control": {
                "run_survival": c_block["run_survival"],
                "usable_run_strict": c_block["usable_run_strict"],
                "attacker_call_validity": c_block["attacker_call_validity"],
                "attacker_call_semantic_usability": c_block["attacker_call_semantic_usability"],
            },
            "b2": {
                "run_survival": b_block["run_survival"],
                "usable_run_strict": b_block["usable_run_strict"],
                "attacker_call_validity": b_block["attacker_call_validity"],
                "attacker_call_semantic_usability": b_block["attacker_call_semantic_usability"],
            },
        },
    }, indent=2), encoding="utf-8")
    (OUT / "analysis" / "failure_breakdown.json").write_text(json.dumps({
        "phase": "17", "stage": "4.6", "deliverable": "failure_breakdown.json",
        "phase_analysed": primary_phase,
        "per_arm": {arm: stats["arm_rates"][arm]["failure_types"] for arm in ARMS},
        "per_attack": {arm: per_attack([r for r in p_rows if r["arm"] == arm]) for arm in ARMS},
        "callback": "failure_analysis.json holds every failure with its raw output",
    }, indent=2), encoding="utf-8")

    # ---------------- control_vs_b2.json ---------------- #
    comparison = {
        "phase": "17", "stage": "4.6", "deliverable": "control_vs_b2.json",
        "phase_analysed": primary_phase,
        "table": {
            "run_survival": {"control": c_block["run_survival"], "b2": b_block["run_survival"]},
            "usable_run_strict": {"control": c_block["usable_run_strict"],
                                  "b2": b_block["usable_run_strict"]},
            "usable_run_any_turn": {"control": c_block["usable_run_any_turn"],
                                    "b2": b_block["usable_run_any_turn"]},
            "attacker_call_validity": {"control": c_block["attacker_call_validity"],
                                       "b2": b_block["attacker_call_validity"]},
            "attacker_call_semantic_usability": {
                "control": c_block["attacker_call_semantic_usability"],
                "b2": b_block["attacker_call_semantic_usability"]},
            "turns_completed": {"control": c_block["turns"], "b2": b_block["turns"]},
            "failure_types": {"control": c_block["failure_types"], "b2": b_block["failure_types"]},
            "latency_s": {"control": c_block["latency_s"], "b2": b_block["latency_s"]},
            "peak_vram_gib": {"control": c_block["peak_vram_gib"], "b2": b_block["peak_vram_gib"]},
            "target_eos": {"control": [c_block["target_eos"], c_block["target_calls"]],
                           "b2": [b_block["target_eos"], b_block["target_calls"]]},
            "judge": {"control": [c_block["judge_calls"], c_block["judge_failures"]],
                      "b2": [b_block["judge_calls"], b_block["judge_failures"]]},
            "rubric_scores": {"control": c_block["rubric_scores"], "b2": b_block["rubric_scores"]},
        },
        "capability": capability,
        "statistics": stats["primary_rates"],
    }
    (OUT / "control_vs_b2.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    # ---------------- stage4_vs_stage4_6.json ---------------- #
    s4 = json.loads((STAGE4 / "multiturn_survival_summary.json").read_text(encoding="utf-8"))
    s45 = json.loads((STAGE45 / "multiturn_summary.json").read_text(encoding="utf-8"))
    (OUT / "stage4_vs_stage4_6.json").write_text(json.dumps({
        "phase": "17", "stage": "4.6", "deliverable": "stage4_vs_stage4_6.json",
        "stage4_frozen_control": {
            "runs": s4["overall"]["runs"],
            "run_survival_rate": s4["overall"]["run_survival_rate"],
            "usable_run_rate": s4["overall"]["usable_run_rate"],
            "mean_completed_turns": s4["overall"]["mean_completed_turns"],
            "source": "results/phase17_model_qualification/stage4_qwen38_reliability/",
        },
        "stage4_5_control_10_runs": {
            "run_survival_rate": s45["candidates"]["MT_control"]["run_survival_rate"],
            "usable_run_rate": s45["candidates"]["MT_control"]["usable_run_rate"],
            "attacker_calls": [s45["candidates"]["MT_control"]["attacker_calls_valid"],
                               s45["candidates"]["MT_control"]["attacker_calls"]],
        },
        "stage4_5_b2_10_runs": {
            "run_survival_rate": s45["candidates"]["MT_B2"]["run_survival_rate"],
            "usable_run_rate": s45["candidates"]["MT_B2"]["usable_run_rate"],
            "attacker_calls": [s45["candidates"]["MT_B2"]["attacker_calls_valid"],
                               s45["candidates"]["MT_B2"]["attacker_calls"]],
        },
        "stage4_6": {
            "phase": primary_phase,
            "control": {"run_survival": c_block["run_survival"],
                        "usable_run_strict": c_block["usable_run_strict"],
                        "attacker_call_validity": c_block["attacker_call_validity"]},
            "b2": {"run_survival": b_block["run_survival"],
                   "usable_run_strict": b_block["usable_run_strict"],
                   "attacker_call_validity": b_block["attacker_call_validity"]},
        },
        "note": ("Stage 4/4.5 samples and this qualification are reported side by side for "
                 "continuity; only the 4.6 arms are directly comparable, having matched cells"),
    }, indent=2), encoding="utf-8")

    # ---------------- pilot_budget_results.json ---------------- #
    # §14: the pilot round budget is the paper's K_max = 8, so only phases deeper
    # than the 2-turn qualification belong here.
    pilot_phases = [p for p in phases if by_phase[p][0]["max_turns"] > 2]
    if pilot_phases:
        pilot = {}
        for phase in pilot_phases:
            subset = by_phase[phase]
            pilot[phase] = {
                "round_budget": subset[0]["max_turns"],
                "source": "configs/default.yaml attacks.max_turns = 8 (paper K_max = 8)",
                "arms": {arm: arm_block([r for r in subset if r["arm"] == arm]) for arm in ARMS
                         if any(r["arm"] == arm for r in subset)},
                "per_attack": {arm: per_attack([r for r in subset if r["arm"] == arm])
                               for arm in ARMS if any(r["arm"] == arm for r in subset)},
                "statistics": None,
            }
            if all(any(r["arm"] == arm for r in subset) for arm in ARMS):
                c = [r for r in subset if r["arm"] == "control"]
                b = [r for r in subset if r["arm"] == "b2"]
                cb, bb = arm_block(c), arm_block(b)
                pilot[phase]["statistics"] = {
                    "run_survival": risk_difference(
                        cb["run_survival"]["numerator"], cb["run_survival"]["denominator"],
                        bb["run_survival"]["numerator"], bb["run_survival"]["denominator"]),
                    "usable_run_strict": risk_difference(
                        cb["usable_run_strict"]["numerator"], cb["usable_run_strict"]["denominator"],
                        bb["usable_run_strict"]["numerator"], bb["usable_run_strict"]["denominator"]),
                    "attacker_call_validity": risk_difference(
                        cb["attacker_call_validity"]["numerator"], cb["attacker_call_validity"]["denominator"],
                        bb["attacker_call_validity"]["numerator"], bb["attacker_call_validity"]["denominator"]),
                }
        (OUT / "pilot_budget_results.json").write_text(json.dumps({
            "phase": "17", "stage": "4.6", "deliverable": "pilot_budget_results.json",
            "purpose": ("whether B2's reliability survives beyond the two-turn qualification, at "
                        "the round budget already defined for the pilot"),
            "phases": pilot,
        }, indent=2), encoding="utf-8")

    # ---------------- qualification_manifest.json ---------------- #
    def sha(path: Path) -> str | None:
        return (hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() else None)

    manifest = {
        "phase": "17", "stage": "4.6",
        "deliverable": "qualification_manifest.json",
        "purpose": "qualification of B2 against the frozen Stage 4 control",
        "single_intended_difference": {
            "control": {"temperature": 0.7, "prompt": "A0 (frozen production prompt)"},
            "b2": {"temperature": 0.3, "prompt": "A1 (frozen production prompt + output-contract suffix)"},
            "everything_else": "identical — same model, revision, architecture, processor, "
                               "quantization, thinking setting, attacks, target, judge, evaluator, "
                               "rubric, success criterion and GPU pipeline",
        },
        "arms": {arm: json.loads((OUT / arm / "manifest.json").read_text(encoding="utf-8"))
                 for arm in ARMS if (OUT / arm / "manifest.json").is_file()},
        "shared_configuration": {
            "attacker": {
                "model_id": common.MODEL_ID, "revision": common.REVISION,
                "architecture": "Qwen3_5ForConditionalGeneration",
                "loader": "AutoModelForImageTextToText",
                "processor": "Qwen3VLProcessor",
                "quantization": "nf4 (bitsandbytes 4-bit), bfloat16 compute, double quant",
                "enable_thinking": False, "max_new_tokens": None,
            },
            "target": {
                "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
                "revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
                "max_new_tokens": None, "natural_eos": True, "temperature": 0.7,
            },
            "judge": {
                "model_id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "enable_thinking": False, "temperature": 0.0, "do_sample": False,
            },
            "attacks": list(common.ATTACKS),
            "nbf": {
                "path": "checkpoints/models_best_nbf_released.pth",
                "bytes": 11757138,
                "sha256": "CEA1A75BCEF4FC515814B69C42541C95114F587ABCC4505C9B096BBFA2A136FE",
                "role_in_this_stage": "not loaded; no steering (barrier=None, eta=0.0)",
            },
        },
        "prompt_hashes_per_arm": {
            arm: {attack: common.render_check(attack, ARM_PROMPT_VARIANT[arm])
                  for attack in common.ATTACKS}
            for arm in ARMS
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": __import__("torch").version.cuda,
            "gpu": "NVIDIA RTX 4500 Ada (24 GiB)",
        },
        "source_hashes": {
            rel: sha(REPO_ROOT / rel) for rel in (
                "src/guardbound/attacks/crescendo_paper.py",
                "src/guardbound/attacks/opposite_day.py",
                "src/guardbound/attacks/acronym.py",
                "src/guardbound/attacks/rubric_evaluation.py",
                "src/guardbound/attacks/runner.py",
                "src/guardbound/llm/local_client.py",
                "src/guardbound/llm/qwen38_native_client.py",
            )
        },
        "phases": {p: {"max_turns": by_phase[p][0]["max_turns"],
                       "runs": {arm: sum(1 for r in by_phase[p] if r["arm"] == arm)
                                for arm in ARMS}} for p in phases},
        "seed_policy": "seed_base 4286 + 1000*run_index + call_index, set before each attacker "
                       "generation; identical schedule in both arms",
    }
    (OUT / "qualification_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"primary phase {primary_phase}:")
    for arm in ARMS:
        block = stats["arm_rates"][arm]
        print(f"  {arm:8s} survival={block['run_survival']['rate']} "
              f"({block['run_survival']['numerator']}/{block['run_survival']['denominator']}) "
              f"usable_strict={block['usable_run_strict']['rate']} "
              f"calls={block['attacker_call_validity']['rate']} "
              f"({block['attacker_call_validity']['numerator']}/{block['attacker_call_validity']['denominator']})")
    for key, value in stats["primary_rates"].items():
        print(f"  {key:34s} diff={value['absolute_difference']:+.4f} "
              f"CI={value['difference_ci95']} OR={value['odds_ratio']} p={value['fisher_exact_two_sided_p']}")
    print(f"wrote artifacts to {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
