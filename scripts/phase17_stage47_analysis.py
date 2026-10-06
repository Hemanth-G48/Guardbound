"""Phase 17 Stage 4.7 — pilot analysis.

Reads `results/phase17_pilot/runs/raw_results.jsonl` and writes the analysis
artifacts the stage brief names:

    analysis/reliability.json        attacker output reliability
    analysis/trajectory.json         run-level trajectory behaviour
    analysis/attack_effectiveness.json  the existing success criterion
    analysis/failure_breakdown.json  the established failure taxonomy
    analysis/latency_vram.json       engineering measurements
    pilot_summary.json               everything, plus the readiness context

The three concepts are kept separate throughout (brief §20): output reliability,
attack capability/trajectory, and attack success are reported as distinct blocks and
never substituted for one another. Nothing is re-derived: `success`, `final_score`,
`rubric_scores` and `termination_reason` come from the frozen run records.

With no pilot runs present the script still writes every artifact, each marked
`"status": "pilot_not_run"`, so the frozen layout exists and cannot be mistaken for
results.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase17_pilot"
RUNS = OUT / "runs" / "raw_results.jsonl"
SMOKE = OUT / "smoke_test" / "results.jsonl"
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def wilson(s: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = s / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def block(runs: list[dict], key: str) -> dict:
    n = len(runs)
    s = sum(1 for r in runs if r.get(key))
    return {"numerator": s, "denominator": n, "rate": round(s / n, 4) if n else None,
            "ci95_wilson": wilson(s, n)}


def analyse(runs: list[dict]) -> dict:
    calls = [c for r in runs for c in r.get("attacker_calls", [])]
    valid = [c for c in calls if c["category"] in VALID]
    usable = [c for c in valid if c["failure_type"] is None]
    failed = [c for c in calls if c["failure_type"] is not None]

    reliability = {
        "attacker_calls": len(calls),
        "valid_calls": len(valid),
        "semantic_usable_calls": len(usable),
        "validity_rate": round(len(valid) / len(calls), 4) if calls else None,
        "validity_ci95": wilson(len(valid), len(calls)),
        "semantic_usability_rate": round(len(usable) / len(calls), 4) if calls else None,
        "semantic_usability_ci95": wilson(len(usable), len(calls)),
        "direct_json": sum(1 for c in valid if c["category"] == "VALID_DIRECT_JSON"),
        "frozen_extraction": sum(1 for c in valid if c["category"] == "VALID_FROZEN_EXTRACTION"),
        "prose_failures": sum(1 for c in failed if c["failure_type"] == "prose_output"),
        "empty_query_failures": sum(1 for c in failed
                                    if c["failure_type"] == "empty_generatedQuestion"),
        "prose_rate": round(sum(1 for c in failed if c["failure_type"] == "prose_output")
                            / len(calls), 4) if calls else None,
        "by_attack": {attack: {"calls": len([c for r in runs if r["attack"] == attack
                                             for c in r.get("attacker_calls", [])]),
                               "valid": len([c for r in runs if r["attack"] == attack
                                             for c in r.get("attacker_calls", [])
                                             if c["category"] in VALID])}
                      for attack in sorted({r["attack"] for r in runs})},
    }

    durations = [r["duration_s"] for r in runs]
    trajectory = {
        "runs": len(runs),
        "runs_completed": sum(1 for r in runs
                              if r["failure_class"] not in ("infrastructure_failure",
                                                            "cuda_oom_failure")),
        "run_survival": block(runs, "_survived"),
        "strict_usable_runs": {
            "definition": "at least one attacker call AND every attacker call in the run "
                          "semantically usable",
            "numerator": sum(1 for r in runs if r.get("attacker_calls")
                             and all(c["failure_type"] is None
                                     for c in r["attacker_calls"])),
            "denominator": len(runs),
            "rate": round(sum(1 for r in runs if r.get("attacker_calls")
                              and all(c["failure_type"] is None
                                      for c in r["attacker_calls"])) / len(runs), 4)
            if runs else None,
        },
        "completed_turns": {
            "total": sum(r["completed_turns"] for r in runs),
            "mean": round(statistics.mean(r["completed_turns"] for r in runs), 3) if runs else None,
            "median": statistics.median(r["completed_turns"] for r in runs) if runs else None,
            "max": max((r["completed_turns"] for r in runs), default=None),
        },
        "full_budget_runs": {
            "numerator": sum(1 for r in runs
                             if r["termination_reason"] == "max_turns_reached"),
            "denominator": len(runs),
            "note": "max_turns = 8 is a cap, not a required depth",
        },
        "termination_reasons": dict(Counter(r["termination_reason"] for r in runs)),
        "target_refusals": sum(r.get("refusal_count") or 0 for r in runs),
        "target_eos_rate": (
            round(sum(r["counts"]["target_eos"] for r in runs)
                  / max(1, sum(r["counts"]["target_calls"] for r in runs)), 4)
            if runs else None),
    }

    successes = sum(1 for r in runs if r.get("success"))
    effectiveness = {
        "criterion": "the existing frozen criterion — attack.was_successful() "
                     "(rubric score 5 = unsafe = success); no new definition is introduced",
        "successes": successes,
        "total_runs": len(runs),
        "success_rate": round(successes / len(runs), 4) if runs else None,
        "success_rate_ci95": wilson(successes, len(runs)),
        "by_attack": {attack: {"successes": sum(1 for r in runs
                                                if r["attack"] == attack and r.get("success")),
                               "runs": len([r for r in runs if r["attack"] == attack])}
                      for attack in sorted({r["attack"] for r in runs})},
        "final_scores": dict(Counter(str(r.get("final_score")) for r in runs)),
        "rubric_scores_all_turns": dict(Counter(
            s for r in runs for s in (r.get("rubric_scores") or []) if s is not None)),
        "separation_note": "attack success is NOT output reliability: a run can succeed "
                           "with an invalid attacker call earlier, and a run with perfect "
                           "JSON can fail to move the target. Both are reported, separately.",
    }

    failure_breakdown = {
        "taxonomy": ["prose_output", "malformed_json", "unquoted_json_key", "truncated_json",
                     "empty_output", "empty_generatedQuestion", "missing_generatedQuestion",
                     "missing_lastResponseSummary", "wrong_field_type", "other_reasoning_only",
                     "attacker_generation_error", "target_failure", "judge_failure",
                     "cuda_oom_failure", "infrastructure_failure", "natural_termination"],
        "attacker_call_failures": dict(Counter(c["failure_type"] for c in failed)),
        "run_level_classes": dict(Counter(r["failure_class"] for r in runs)),
        "judge_failures": sum(r["counts"]["judge_failures"] for r in runs),
        "infrastructure_errors": [{"run_id": r["run_id"], "error": r.get("infrastructure_error")}
                                  for r in runs if r.get("infrastructure_error")],
        "raw_outputs_preserved": True,
        "per_failure": ([{"run_id": r["run_id"], "attack": r["attack"], "goal_id": r["goal_id"],
                          "failure_type": c["failure_type"], "category": c["category"],
                          "raw_output": (c.get("raw_output") or "")[:500]}
                         for r in runs for c in r.get("attacker_calls", [])
                         if c["failure_type"]]),
    }

    latencies = [c["latency_s"] for c in calls if c.get("latency_s") is not None]
    generated = [c["generated_tokens"] for c in calls if c.get("generated_tokens") is not None]
    latency_vram = {
        "runs": len(runs),
        "wall_clock_seconds_total": round(sum(durations), 1) if durations else None,
        "run_duration_seconds": {
            "mean": round(statistics.mean(durations), 1) if durations else None,
            "median": round(statistics.median(durations), 1) if durations else None,
            "p95": round(sorted(durations)[int(0.95 * (len(durations) - 1))], 1)
            if durations else None,
            "max": max(durations) if durations else None,
        },
        "throughput_runs_per_hour": round(3600 / statistics.mean(durations), 2)
        if durations else None,
        "attacker_call_latency_s": {
            "mean": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2)
            if latencies else None,
        },
        "generated_tokens": {
            "mean": round(statistics.mean(generated), 1) if generated else None,
            "max": max(generated) if generated else None,
        },
        "peak_vram_gib": max((r["peak_vram_gib"] for r in runs), default=None),
        "oom_count": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
        "cuda_failures": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
        "infrastructure_failures": sum(1 for r in runs
                                       if r["failure_class"] == "infrastructure_failure"),
        "runs_with_attacker_errors": sum(1 for r in runs
                                         if r["failure_class"] == "attacker_generation_error"),
        "runs_with_any_recorded_error": sum(1 for r in runs if r.get("error")),
        "error_label_note": "'cuda_failures'/'infrastructure_failures' count runs whose "
                            "failure class is CUDA/OOM or infrastructure; attacker-side "
                            "generation errors are reported separately and are not "
                            "infrastructure failures",
    }
    return {"reliability": reliability, "trajectory": trajectory,
            "attack_effectiveness": effectiveness, "failure_breakdown": failure_breakdown,
            "latency_vram": latency_vram}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)

    runs = read_jsonl(RUNS)
    for run in runs:
        run["_survived"] = run.get("failure_class") not in (
            "cuda_oom_failure", "infrastructure_failure", "attacker_generation_error")

    if not runs:
        for name in ("reliability", "trajectory", "attack_effectiveness",
                     "failure_breakdown", "latency_vram"):
            (OUT / "analysis" / f"{name}.json").write_text(json.dumps({
                "phase": "17", "stage": "4.7", "deliverable": f"analysis/{name}.json",
                "status": "pilot_not_run",
                "note": "no pilot runs exist yet; this file is written by "
                        "scripts/phase17_stage47_analysis.py once runs/raw_results.jsonl "
                        "is populated by the frozen pilot runner",
            }, indent=2), encoding="utf-8")
        smoke_runs = read_jsonl(SMOKE)
        smoke_summary = json.loads((OUT / "smoke_test" / "summary.json").read_text(encoding="utf-8")) \
            if (OUT / "smoke_test" / "summary.json").is_file() else None
        (OUT / "pilot_summary.json").write_text(json.dumps({
            "phase": "17", "stage": "4.7",
            "deliverable": "pilot_summary.json",
            "status": "HELD — readiness certified, pilot not launched",
            "pilot_launch": {
                "status": "HOLD — readiness certified, pilot not launched",
                "command": "python scripts/phase17_stage47_pilot.py --mode pilot",
                "expected_runtime_hours": (
                    round(90 * statistics.mean([r["duration_s"] for r in smoke_runs]) / 3600, 1)
                    if smoke_runs else None),
                "note": "this file will be rewritten with the measured metrics once the "
                        "pilot runs; it currently reports the readiness state only",
            },
            "readiness": {
                "gate": "see pilot_manifest.json and the Stage 4.7 report",
                "frozen_configuration": "A0 / temperature 0.7 / top_p 1.0 / thinking OFF / "
                                        "NF4 + BF16 / Qwen3.8-27B native",
                "B2": "PROMISING BUT NOT QUALIFIED — archived, not used",
            },
            "smoke_test": {
                "runs": len(smoke_runs),
                "summary": smoke_summary,
            },
            "pilot_expectation": {
                "runs": 90,
                "matrix": "30 goals x 3 attacks x 1 condition (NBF OFF)",
                "max_turns": 8,
                "measured_seconds_per_run_from_smoke": (
                    round(statistics.mean([r["duration_s"] for r in smoke_runs]), 1)
                    if smoke_runs else None),
                "estimated_total_hours": (
                    round(90 * statistics.mean([r["duration_s"] for r in smoke_runs]) / 3600, 1)
                    if smoke_runs else None),
            },
            "note": "this file reports the readiness state; the analysis artifacts under "
                    "analysis/ fill in with real metrics when the pilot runs",
        }, indent=2), encoding="utf-8")
        print(f"no pilot runs; wrote status-only artifacts "
              f"(smoke runs: {len(smoke_runs)})")
        return 0

    results = analyse(runs)
    for name, payload in results.items():
        (OUT / "analysis" / f"{name}.json").write_text(json.dumps({
            "phase": "17", "stage": "4.7", "deliverable": f"analysis/{name}.json",
            "status": "measured", **payload}, indent=2), encoding="utf-8")

    summary = {
        "phase": "17", "stage": "4.7", "deliverable": "pilot_summary.json",
        "status": "measured",
        "runs": len(runs),
        "configuration": "A0 / temperature 0.7 / top_p 1.0 / thinking OFF / NF4 + BF16 / "
                         "Qwen3.8-27B native / target Llama-3-8B / judge Qwen3-8B / NBF disabled",
        **results,
    }
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"analysed {len(runs)} runs")
    print(f"  attacker-call validity: {results['reliability']['validity_rate']} "
          f"({results['reliability']['valid_calls']}/{results['reliability']['attacker_calls']})")
    print(f"  run survival: {results['trajectory']['run_survival']['rate']}")
    print(f"  attack success: {results['attack_effectiveness']['success_rate']} "
          f"({results['attack_effectiveness']['successes']}/"
          f"{results['attack_effectiveness']['total_runs']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
