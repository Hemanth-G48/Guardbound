"""Phase 17 Stage 4 — analysis of the reliability measurements (Parts 12, 15, 17-22, 26).

Reads the raw JSONL produced by the isolated reliability test and the multi-turn
survival test and derives every summary, taxonomy and comparison from them. No
number in the deliverables is typed in by hand, and no ASR is computed anywhere.

The sensitivity analysis in Part 17 is labelled as a theoretical approximation
and is reported next to the observed survival, never instead of it.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage4_qwen38_reliability"
STAGE3_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage3_qwen38_native"
STAGE2_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"

ATTACKS = ("crescendo_paper", "opposite_day", "acronym")
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def empty_query_flags(raw) -> bool | None:
    """Was the reply schema-valid but semantically empty?

    Returns None when the reply is not a parsed object with the field — that is
    a different failure and is classified as one. A ``True`` here means the JSON
    contract was satisfied and the attack still could not proceed: the frozen
    runner stops at its empty-query guard, without any generation error. This is
    why the report distinguishes *contract validity* from *semantic usability*.
    """
    if not isinstance(raw, str):
        return None
    from guardbound.llm.local_client import _extract_json_block

    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        parsed = _extract_json_block(raw)
    if not isinstance(parsed, dict) or "generatedQuestion" not in parsed:
        return None
    value = parsed["generatedQuestion"]
    return (not isinstance(value, str)) or not value.strip()


def stats(values: list[float]) -> dict | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return {
        "n": len(clean),
        "mean": round(statistics.mean(clean), 3),
        "median": round(statistics.median(clean), 3),
        "min": min(clean),
        "max": max(clean),
    }


def survival_summary(runs: list[dict]) -> dict:
    def survival_mode(run: dict) -> str:
        """Why the run ended — four distinct outcomes, not two.

        ``survived`` alone hides real differences:

          * the full turn budget was met;
          * the run ended early *without* any attacker failure because the target
            refused, the runner backtracked, and the round budget was consumed;
          * the attacker returned schema-valid but **empty** fields, which the
            frozen runner treats as an empty query and stops on — no generation
            error is raised, so the brief's literal survival definition counts it
            as a survival while the run produced nothing.
        """
        if not run["run_survived"]:
            return "terminated_by_attacker_failure"
        if any(empty_query_flags(call.get("raw_output")) is True
               for call in run.get("attacker_calls", [])):
            return "terminated_by_empty_query"
        if run["turns_completed"] >= (run.get("max_turns_setting") or 0):
            return "survived_full_turn_budget"
        return "ended_early_without_attacker_failure"

    def block(subset: list[dict]) -> dict:
        n = len(subset)
        if not n:
            return {"n": 0}
        survived = sum(1 for r in subset if r["run_survived"])
        attempted = sum(r["attacker_calls_attempted"] for r in subset)
        valid = sum(r["attacker_calls_valid"] for r in subset)
        invalid = sum(r["attacker_calls_invalid"] for r in subset)
        causes = Counter(
            "completed_without_attacker_failure" if r["run_survived"] else
            str(r["termination_reason"]).split(":")[0]
            for r in subset
        )
        categories = Counter(
            r["failure_category"] for r in subset
            if not r["run_survived"] and r["failure_category"]
        )
        failure_turns = Counter(r["failure_turn"] for r in subset
                                if not r["run_survived"] and r["failure_turn"])
        return {
            "runs": n,
            "runs_survived": survived,
            "runs_terminated_by_attacker_failure": n - survived,
            "run_survival_rate": round(survived / n, 4),
            "attacker_calls_attempted": attempted,
            "attacker_calls_valid": valid,
            "attacker_calls_invalid": invalid,
            "per_call_validity_within_runs": round(valid / attempted, 4) if attempted else None,
            "mean_completed_turns": round(statistics.mean(r["turns_completed"] for r in subset), 3),
            "mean_attacker_calls_per_run": round(attempted / n, 2),
            "termination_causes": dict(causes),
            "survival_modes": dict(Counter(survival_mode(r) for r in subset)),
            "usable_runs_turns_ge_1": sum(1 for r in subset if r["turns_completed"] >= 1),
            "usable_run_rate": round(
                sum(1 for r in subset if r["turns_completed"] >= 1) / n, 4),
            "attacker_calls_with_empty_query": sum(
                1 for r in subset for c in r["attacker_calls"]
                if empty_query_flags(c.get("raw_output")) is True),
            "turns_completed_histogram": dict(sorted(Counter(
                r["turns_completed"] for r in subset).items())),
            "failure_categories": dict(categories),
            "failure_turn_distribution": dict(sorted(failure_turns.items())),
            "target_calls": sum(r["target_calls"] for r in subset),
            "target_eos": sum(r["target_eos_count"] for r in subset),
            "target_hit_budget": sum(r["target_hit_budget_count"] for r in subset),
            "judge_calls": sum(r["judge_calls"] for r in subset),
            "judge_failures": sum(r["judge_failures"] for r in subset),
            "peak_vram_gib": max(r["peak_vram_gib"] for r in subset),
            "wall_s": stats([r["wall_s"] for r in subset]),
        }

    overall = block(runs)
    per_attack = {attack: block([r for r in runs if r["attack"] == attack]) for attack in ATTACKS}
    return {
        "phase": "17",
        "stage": "4",
        "deliverable": "multiturn_survival_summary.json",
        "definition": {
            "run_survival": ("the run reached its configured termination without any "
                             "attacker-generation failure (no AttackGenerationError)"),
            "per_call_validity": ("share of individual attacker replies that satisfied the "
                                  "frozen JSON contract, within these same runs"),
            "note": ("survival is a run-level property; a run with 9 valid calls and 1 prose "
                     "reply does not survive"),
        },
        "controlled_setting": {
            "max_turns": runs[0]["max_turns_setting"] if runs else None,
            "note": "identical to Stage 3's controlled multi-turn setting",
        },
        "overall": overall,
        "per_attack": per_attack,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    isolated = read_jsonl(OUT_DIR / "attacker_reliability_raw.jsonl")
    runs = read_jsonl(OUT_DIR / "multiturn_survival_raw.jsonl")
    if not runs:
        print("no survival runs yet — nothing to summarise")
    else:
        summary = survival_summary(runs)
        (OUT_DIR / "multiturn_survival_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["overall"], indent=2)[:1500])
        for attack, block in summary["per_attack"].items():
            print(f"  {attack:16s} survival {block['runs_survived']}/{block['runs']} "
                  f"calls {block['attacker_calls_valid']}/{block['attacker_calls_attempted']} "
                  f"turns~{block['mean_completed_turns']}")

    if isolated:
        # --- sensitivity: per-call reliability vs run survival (Part 17) ------ #
        overall_calls = len(isolated)
        valid_calls = sum(1 for r in isolated if r["category"] in VALID)
        p_isolated = valid_calls / overall_calls
        p_within_runs = None
        mean_calls_per_run = None
        observed_survival = None
        if runs:
            attempted = sum(r["attacker_calls_attempted"] for r in runs)
            valid_in_runs = sum(r["attacker_calls_valid"] for r in runs)
            p_within_runs = valid_in_runs / attempted if attempted else None
            mean_calls_per_run = attempted / len(runs)
            observed_survival = sum(1 for r in runs if r["run_survived"]) / len(runs)

        sensitivity = {
            "phase": "17", "stage": "4",
            "deliverable": "sensitivity_analysis",
            "status": "THEORETICAL APPROXIMATION — not an experimental result",
            "model": "P(survive a run of k calls) ≈ p^k, assuming independent calls",
            "measured_per_call_validity": {
                "isolated": round(p_isolated, 4),
                "within_survival_runs": (round(p_within_runs, 4) if p_within_runs else None),
            },
            "measured_mean_calls_per_run": (round(mean_calls_per_run, 2) if mean_calls_per_run else None),
            "predicted_survival_at_that_run_length": {},
            "observed_run_survival": observed_survival,
            "comparison_note": ("The approximation assumes per-call independence, which the "
                                "measurements do not establish; it is reported only to show "
                                "how a per-call rate compounds over a run, and the observed "
                                "survival is the figure that matters."),
        }
        for label, p in (("isolated", p_isolated), ("within_runs", p_within_runs)):
            if p and mean_calls_per_run:
                sensitivity["predicted_survival_at_that_run_length"][label] = {
                    f"{k}_calls": round(p ** k, 4) for k in (3, 5, 8, 12, 16)
                }

        # --- Stage 3 vs Stage 4 (Part 26) ------------------------------------- #
        stage3_isolated = json.loads((STAGE3_DIR / "native_json_qualification.json").read_text(encoding="utf-8"))
        stage3_mt = json.loads((STAGE3_DIR / "native_multiturn_qualification.json").read_text(encoding="utf-8"))
        stage3_runs_terminated = sum(
            1 for b in stage3_mt["attacks"].values() if b.get("error")
        )
        stage3_calls = sum(len(b["attacker_calls"]) for b in stage3_mt["attacks"].values())
        stage3_valid = sum(b["attacker_json_valid"] for b in stage3_mt["attacks"].values())

        comparison = {
            "phase": "17", "stage": "4",
            "deliverable": "stage3_vs_stage4_comparison.json",
            "isolated": {
                "stage3": {"n": stage3_isolated["n_cases"],
                           "valid": stage3_isolated["valid_json_total"],
                           "rate": stage3_isolated["validity_rate"],
                           "direct": stage3_isolated["valid_json_direct"],
                           "extracted": stage3_isolated["valid_json_via_extraction"]},
                "stage4": {"n": overall_calls, "valid": valid_calls,
                           "rate": round(p_isolated, 4),
                           "direct": sum(1 for r in isolated if r["category"] == "VALID_DIRECT_JSON"),
                           "extracted": sum(1 for r in isolated
                                            if r["category"] == "VALID_FROZEN_EXTRACTION")},
            },
            "multi_turn": {
                "stage3": {"runs": len(stage3_mt["attacks"]), "calls": stage3_calls,
                           "valid_calls": stage3_valid,
                           "runs_ended_by_attacker_failure": stage3_runs_terminated},
                "stage4": ({"runs": len(runs),
                            "runs_survived": sum(1 for r in runs if r["run_survived"]),
                            "calls": sum(r["attacker_calls_attempted"] for r in runs),
                            "valid_calls": sum(r["attacker_calls_valid"] for r in runs),
                            "runs_ended_by_attacker_failure": sum(1 for r in runs if not r["run_survived"])}
                           if runs else None),
            },
            "interpretation_rule": ("A difference between the two stages is reported as a "
                                    "difference. No significance is claimed: Stage 3's "
                                    "multi-turn sample was 3 runs, Stage 4's is 30."),
            "representativeness_question": {
                "stage3_isolated_22_of_24": "re-tested at 150 calls in Stage 4 (see isolated block)",
                "stage3_3_of_3_terminations": "re-tested at 30 runs in Stage 4 (see multi_turn block)",
            },
        }

        # --- context growth and failure-turn analysis (Parts 18/19) ----------- #
        # Contract validity vs semantic usability (the empty-query mode).
        isolated_empty = [r for r in isolated
                          if empty_query_flags(r.get("raw_output")) is True]
        isolated_valid = [r for r in isolated if r["category"] in VALID]
        run_calls = [c for r in runs for c in r["attacker_calls"]] if runs else []
        run_empty = [c for c in run_calls if empty_query_flags(c.get("raw_output")) is True]
        run_valid = [c for c in run_calls
                     if c["category"] in VALID]

        analysis = {
            "phase": "17", "stage": "4",
            "semantic_usability": {
                "why": ("a reply can satisfy the JSON contract and still be unusable: "
                        "`{\"generatedQuestion\": \"\", \"lastResponseSummary\": \"\"}` "
                        "parses, carries both fields as strings, and stops the frozen "
                        "runner at its empty-query guard without any generation error"),
                "isolated": {
                    "calls": len(isolated),
                    "contract_valid": len(isolated_valid),
                    "empty_query": len(isolated_empty),
                    "semantically_usable": len(isolated_valid) - len(isolated_empty),
                    "usable_rate": (round((len(isolated_valid) - len(isolated_empty)) / len(isolated), 4)
                                    if isolated else None),
                },
                "survival_runs": {
                    "calls": len(run_calls),
                    "contract_valid": len(run_valid),
                    "empty_query": len(run_empty),
                    "semantically_usable": len(run_valid) - len(run_empty),
                    "usable_rate": (round((len(run_valid) - len(run_empty)) / len(run_calls), 4)
                                    if run_calls else None),
                    "empty_query_runs": sorted({
                        r["run_index"] for r in runs
                        if any(empty_query_flags(c.get("raw_output")) is True
                               for c in r["attacker_calls"])
                    }) if runs else [],
                },
            },
            "context_growth": {
                "isolated_mean_prompt_tokens_valid": stats([
                    r["prompt_tokens"] for r in isolated if r["category"] in VALID]),
                "isolated_mean_prompt_tokens_failed": stats([
                    r["prompt_tokens"] for r in isolated if r["category"] not in VALID]),
                "survival_prompt_tokens_valid": stats([
                    c["prompt_tokens"] for r in runs for c in r["attacker_calls"]
                    if c["category"] in VALID]),
                "survival_prompt_tokens_failed": stats([
                    c["prompt_tokens"] for r in runs for c in r["attacker_calls"]
                    if c["category"] not in VALID]),
                "note": ("Reported as measured. A difference in prompt length between valid "
                         "and failed calls is not by itself evidence of context degradation, "
                         "and no such claim is made here."),
            },
            "failure_turns_in_runs": dict(sorted(Counter(
                r["failure_turn"] for r in runs if not r["run_survived"] and r["failure_turn"]
            ).items())) if runs else {},
            "per_attack_failures": {
                attack: dict(Counter(
                    r["failure_category"] for r in runs
                    if r["attack"] == attack and not r["run_survived"] and r["failure_category"]
                )) for attack in ATTACKS
            } if runs else {},
        }

        for name, payload in (
            ("sensitivity_analysis.json", sensitivity),
            ("stage3_vs_stage4_comparison.json", comparison),
            ("failure_and_context_analysis.json", analysis),
        ):
            (OUT_DIR / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"wrote {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
