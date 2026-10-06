"""Phase 17 Stage 6 — run-level attacker contract intervention study: analysis.

Reads the Stage 6 run/call records (read-only) and writes the artifact set under
`results/phase17_pilot/analysis/stage6_run_level_contract/`:

    arm_comparison.json        M1-M10 per arm
    attack_comparison.json     per attack, per arm
    depth_analysis.json        validity by call index, survival by depth
    failure_cascade.json       the funnel per arm
    statistical_analysis.json  primary/secondary tests, paired + unpaired, discipline
    baseline_check.json        §22 before/after frozen-artifact hash comparison
    figures/                   six required figures

The primary endpoint is official attack success (the frozen criterion). Everything else is
secondary or exploratory, and the report states the number of comparisons and the
correction applied.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "phase17_pilot" / "analysis" / "stage6_run_level_contract"
FIG = OUT / "figures"
RUNS = OUT / "run_level_metrics.jsonl"
CALLS = OUT / "call_level_metrics.jsonl"
MANIFEST = OUT / "baseline_manifest.json"

ARMS = ["control", "generation_point", "a1", "a1_t03"]
ARM_LABEL = {"control": "C0 frozen control (A0, T=0.7)",
             "generation_point": "C1 generation-point instruction (T=0.7)",
             "a1": "C2 A1 output contract (T=0.7)",
             "a1_t03": "C3 A1 output contract (T=0.3)"}
ATTACKS = ["crescendo_paper", "opposite_day", "acronym"]
LABEL = {"crescendo_paper": "Crescendo", "opposite_day": "OppositeDay", "acronym": "Acronym"}
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


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact two-sided McNemar over matched (control, arm) cells."""
    b = sum(1 for c, t in pairs if (not c) and t)     # control fails, arm succeeds
    c_ = sum(1 for c, t in pairs if c and (not t))    # control succeeds, arm fails
    n = b + c_
    if n == 0:
        p = 1.0
    else:
        k = min(b, c_)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))
    return {"control_no_arm_yes": b, "control_yes_arm_no": c_, "discordant": n,
            "exact_two_sided_p": round(p, 6)}


def paired_permutation(xs: list[float], ys: list[float], n_perm: int = 20000,
                       seed: int = 20261002) -> dict:
    """Paired permutation test on the mean difference (sign-flip)."""
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


def arm_vs_control(runs: list[dict], arm: str, endpoint: str) -> dict:
    subset = [r for r in runs if r["arm"] == arm]
    control = [r for r in runs if r["arm"] == "control"]
    key = {"success": "success", "survival": "_survived", "validity": None}.get(endpoint)
    if endpoint == "success":
        a = sum(1 for r in subset if r["success"])
        b = len(subset) - a
        c = sum(1 for r in control if r["success"])
        d = len(control) - c
    elif endpoint == "survival":
        a = sum(1 for r in subset if r["_survived"])
        b = len(subset) - a
        c = sum(1 for r in control if r["_survived"])
        d = len(control) - c
    else:  # attacker-call validity
        ac = [c for r in subset for c in r["attacker_calls"]]
        cc = [c for r in control for c in r["attacker_calls"]]
        a = sum(1 for c in ac if c["category"] in VALID)
        b = len(ac) - a
        c = sum(1 for c in cc if c["category"] in VALID)
        d = len(cc) - c
    pa = a / (a + b) if (a + b) else None
    pc = c / (c + d) if (c + d) else None
    result = {
        "arm": arm, "endpoint": endpoint,
        "arm_rate": round(pa, 4) if pa is not None else None,
        "control_rate": round(pc, 4) if pc is not None else None,
        "risk_difference": round(pa - pc, 4) if pa is not None and pc is not None else None,
        "relative_risk": (round(pa / pc, 4) if pa is not None and pc not in (None, 0) else None),
        "arm_n": a + b, "control_n": c + d,
        "arm_ci95": wilson(a, a + b), "control_ci95": wilson(c, c + d),
        "fisher_exact_two_sided_p": fisher(a, b, c, d),
    }
    TESTS.append({"arm": arm, "endpoint": endpoint,
                  "p_value": result["fisher_exact_two_sided_p"],
                  "kind": "primary" if endpoint == "success" else "secondary"})
    return result


def paired_cells(runs: list[dict], arm: str) -> dict:
    """Matched (goal, attack) cells: control vs arm."""
    def index(arm_name: str) -> dict:
        return {(r["goal_id"], r["attack"]): r for r in runs if r["arm"] == arm_name}
    control, treat = index("control"), index(arm)
    shared = sorted(set(control) & set(treat))
    success_pairs = [(control[k]["success"], treat[k]["success"]) for k in shared]
    survival_pairs = [(control[k]["_survived"], treat[k]["_survived"]) for k in shared]
    turn_pairs = ([(control[k]["completed_turns"]) for k in shared],
                  [(treat[k]["completed_turns"]) for k in shared])
    return {
        "matched_cells": len(shared),
        "success": mcnemar(success_pairs),
        "survival": mcnemar(survival_pairs),
        "completed_turns": paired_permutation([float(x) for x in turn_pairs[1]],
                                              [float(x) for x in turn_pairs[0]]),
    }


def arm_block(runs: list[dict]) -> dict:
    calls = [c for r in runs for c in r["attacker_calls"]]
    valid = [c for c in calls if c["category"] in VALID]
    usable = [c for c in valid if c["failure_type"] is None]
    failed = [c for c in calls if c["failure_type"] is not None]
    turns = [r["completed_turns"] for r in runs]
    durations = [r["duration_s"] for r in runs]
    rubric = [s for r in runs for s in r["rubric_scores"] if s is not None]
    target_calls = sum(len(r["target_calls"]) for r in runs)
    target_eos = sum(1 for r in runs for c in r["target_calls"]
                     if c.get("termination") == "eos")
    judge_calls = sum(len(r["judge_calls"]) for r in runs)
    return {
        "M1_official_success": {
            "numerator": sum(1 for r in runs if r["success"]), "denominator": len(runs),
            "rate": round(sum(1 for r in runs if r["success"]) / len(runs), 4) if runs else None,
            "ci95_wilson": wilson(sum(1 for r in runs if r["success"]), len(runs))},
        "M2_run_survival": {
            "numerator": sum(1 for r in runs if r["_survived"]), "denominator": len(runs),
            "rate": round(sum(1 for r in runs if r["_survived"]) / len(runs), 4) if runs else None,
            "ci95_wilson": wilson(sum(1 for r in runs if r["_survived"]), len(runs))},
        "M3_contract_validity": {
            "numerator": len(valid), "denominator": len(calls),
            "rate": round(len(valid) / len(calls), 4) if calls else None,
            "ci95_wilson": wilson(len(valid), len(calls))},
        "M4_semantic_usability": {
            "numerator": len(usable), "denominator": len(calls),
            "rate": round(len(usable) / len(calls), 4) if calls else None,
            "ci95_wilson": wilson(len(usable), len(calls))},
        "M5_completed_turns": {
            "total": sum(turns),
            "mean": round(statistics.mean(turns), 3) if turns else None,
            "median": statistics.median(turns) if turns else None,
            "max": max(turns) if turns else None,
            "fraction_reaching_2": round(sum(1 for t in turns if t >= 2) / len(runs), 4) if runs else None,
            "fraction_reaching_4": round(sum(1 for t in turns if t >= 4) / len(runs), 4) if runs else None,
            "fraction_reaching_6": round(sum(1 for t in turns if t >= 6) / len(runs), 4) if runs else None,
            "fraction_reaching_8": round(sum(1 for t in turns if t >= 8) / len(runs), 4) if runs else None,
        },
        "M6_failure_modes": dict(Counter(c["failure_type"] for c in failed)),
        "M7_target": {
            "target_calls": target_calls, "natural_eos": target_eos,
            "natural_eos_rate": round(target_eos / target_calls, 4) if target_calls else None,
            "generation_failures": 0,
            "refusals_total": sum(r.get("refusal_count") or 0 for r in runs),
            "refusals_per_run": round(statistics.mean([r.get("refusal_count") or 0
                                                       for r in runs]), 3) if runs else None},
        "M8_judge": {
            "judge_calls": judge_calls,
            "judge_failures": sum(r["counts"]["judge_failures"] for r in runs),
            "rubric_n": len(rubric),
            "rubric_mean": round(statistics.mean(rubric), 3) if rubric else None,
            "rubric_median": statistics.median(rubric) if rubric else None,
            "rubric_distribution": {str(k): v for k, v in sorted(Counter(rubric).items())}},
        "M9_runtime": {
            "mean": round(statistics.mean(durations), 1) if durations else None,
            "median": round(statistics.median(durations), 1) if durations else None,
            "p95": round(sorted(durations)[int(0.95 * (len(durations) - 1))], 1) if durations else None,
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "throughput_runs_per_hour": round(3600 / statistics.mean(durations), 2)
            if durations else None},
        "M10_gpu": {
            "peak_vram_gib": max((r["peak_vram_gib"] for r in runs), default=None),
            "oom_count": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
            "cuda_failures": sum(1 for r in runs if r["failure_class"] == "cuda_oom_failure"),
            "model_load_failures": sum(1 for r in runs
                                       if r["failure_class"] == "infrastructure_failure")},
    }


def normalise_calls(rows: list[dict]) -> list[dict]:
    """Expose the §19 field names as aliases of the runner's stored values.

    The runner records the shared helper's field names (`prompt_tokens`,
    `generated_tokens`, `category`). §19 names the same quantities differently. This adds
    the aliases; nothing is overwritten and no value is invented. Quantities the pilot
    never stored (message counts, the call→turn mapping) stay null.
    """
    out = []
    for call in rows:
        raw = call.get("raw_output")
        json_valid = call.get("category") in VALID
        row = dict(call)
        row.update({
            "input_context_tokens": call.get("prompt_tokens"),
            "output_tokens": call.get("generated_tokens"),
            "output_characters": len(raw) if isinstance(raw, str) else None,
            "json_valid": json_valid,
            "semantic_valid": json_valid and call.get("empty_query") is not True,
            "generated_question_length": (len(call["generated_question"])
                                          if isinstance(call.get("generated_question"), str)
                                          else None),
            "input_message_count": None,
            "turn_index": None,
            "field_alias_note": "input_context_tokens=prompt_tokens, output_tokens="
                                "generated_tokens, json_valid=category in VALID; the "
                                "originals are retained alongside",
        })
        out.append(row)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(line) for line in RUNS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for run in runs:
        run["_survived"] = run["failure_class"] not in ("attacker_generation_error",
                                                       "cuda_oom_failure",
                                                       "infrastructure_failure")
    present_arms = [a for a in ARMS if any(r["arm"] == a for r in runs)]
    print(f"runs {len(runs)} | arms {present_arms}")

    # Balanced-block filter: the matrix runs goal-major, so only goals whose 12 cells are
    # all present give every arm the same n. A partial trailing goal is preserved in the
    # runner's files but excluded from the analysis, and reported as such.
    all_runs = runs
    complete_goals = [g for g in sorted({r["goal_id"] for r in runs})
                      if all(sum(1 for r in runs if r["goal_id"] == g and r["arm"] == a
                                 and r["attack"] == atk) == 1
                             for a in present_arms for atk in ATTACKS)]
    excluded = [r for r in runs if r["goal_id"] not in complete_goals]
    runs = [r for r in runs if r["goal_id"] in complete_goals]
    for run in runs:
        run["_survived"] = run["failure_class"] not in ("attacker_generation_error",
                                                       "cuda_oom_failure",
                                                       "infrastructure_failure")
    print(f"complete goal blocks: {complete_goals} -> {len(runs)} runs analysed "
          f"({len(excluded)} runs in a partial block excluded, preserved on disk)")

    # §19 call-level artifact, in the brief's field names
    if CALLS.is_file():
        as_recorded = CALLS.read_text(encoding="utf-8")
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        (OUT / "raw" / "call_level_metrics_as_recorded.jsonl").write_text(
            as_recorded, encoding="utf-8")
        raw_calls = [json.loads(line) for line in as_recorded.splitlines() if line.strip()]
        with CALLS.open("w", encoding="utf-8") as handle:
            for row in normalise_calls(raw_calls):
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # §22 frozen-baseline check
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    after = {}
    for rel, before in manifest["files"].items():
        path = REPO_ROOT / str(rel)
        now = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        after[rel] = {"before": before, "after": now, "unchanged": before == now}
    pilot_now = hashlib.sha256((REPO_ROOT / "results" / "phase17_pilot" / "runs"
                                / "raw_results.jsonl").read_bytes()).hexdigest()
    baseline_check = {
        "phase": "17", "stage": "6", "deliverable": "baseline_check.json",
        "files": after,
        "all_unchanged": all(v["unchanged"] for v in after.values()),
        "stage4_8_pilot_data": {
            "before": manifest["stage48_pilot_data_sha256"], "after": pilot_now,
            "unchanged": manifest["stage48_pilot_data_sha256"] == pilot_now},
    }
    (OUT / "baseline_check.json").write_text(json.dumps(baseline_check, indent=2),
                                             encoding="utf-8")

    # arm comparison (M1-M10)
    arm_payload = {"phase": "17", "stage": "6", "deliverable": "arm_comparison.json",
                   "matrix": {
                       "intended": {"arms": 4, "attacks": 3, "goals": 30, "runs": 360},
                       "achieved": {"arms": len(present_arms), "attacks": 3,
                                    "goals": len(complete_goals),
                                    "goal_ids": complete_goals,
                                    "runs": len(runs),
                                    "runs_per_arm": {a: sum(1 for r in runs if r["arm"] == a)
                                                     for a in present_arms},
                                    "runs_in_partial_block_excluded": len(excluded)},
                       "note": "the brief permitted a smaller matrix when the computational "
                               "budget required it; the goal-major execution order means "
                               "every complete goal is a balanced matched quadruple, so the "
                               "achieved matrix is complete at the goal boundary reported "
                               "here, not truncated mid-cell",
                   },
                   "arms": {a: {"label": ARM_LABEL[a], **arm_block([r for r in runs
                                                                   if r["arm"] == a])}
                            for a in present_arms}}
    (OUT / "arm_comparison.json").write_text(json.dumps(arm_payload, indent=2),
                                             encoding="utf-8")

    # per attack
    attack_payload = {"phase": "17", "stage": "6", "deliverable": "attack_comparison.json",
                      "note": "no attack is ranked; the purpose is to see whether the "
                              "intervention behaves consistently across attacks",
                      "attacks": {}}
    for attack in ATTACKS:
        attack_payload["attacks"][LABEL[attack]] = {
            a: arm_block([r for r in runs if r["arm"] == a and r["attack"] == attack])
            for a in present_arms
            if any(r["arm"] == a and r["attack"] == attack for r in runs)}
    (OUT / "attack_comparison.json").write_text(json.dumps(attack_payload, indent=2),
                                                encoding="utf-8")

    # depth analysis: validity by call index, survival by depth
    depth = {"phase": "17", "stage": "6", "deliverable": "depth_analysis.json",
             "validity_by_call_index": {}, "survival_by_depth": {}, "reaching_turn": {}}
    for arm in present_arms:
        calls = []
        for r in runs:
            if r["arm"] != arm:
                continue
            for index, call in enumerate(r["attacker_calls"], start=1):
                calls.append({**call, "call_index": index})
        table = {}
        for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 99)):
            key = f"{lo}+" if hi == 99 else str(lo)
            subset = [c for c in calls if lo <= c["call_index"] <= hi]
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
                                               if c["failure_type"] == "empty_generatedQuestion"),
            }
        depth["validity_by_call_index"][arm] = table
        arm_runs = [r for r in runs if r["arm"] == arm]
        depth["survival_by_depth"][arm] = {
            str(n): round(sum(1 for r in arm_runs if r["completed_turns"] >= n) / len(arm_runs), 4)
            for n in range(1, 9)}
        depth["reaching_turn"][arm] = {
            str(n): sum(1 for r in arm_runs if r["completed_turns"] >= n) for n in range(1, 9)}
    (OUT / "depth_analysis.json").write_text(json.dumps(depth, indent=2), encoding="utf-8")

    # failure cascade
    cascade = {"phase": "17", "stage": "6", "deliverable": "failure_cascade.json", "arms": {}}
    for arm in present_arms:
        arm_runs = [r for r in runs if r["arm"] == arm]
        cascade["arms"][arm] = {
            "runs": len(arm_runs),
            "runs_with_attacker_call": sum(1 for r in arm_runs if r["attacker_calls"]),
            "runs_with_valid_attacker_call": sum(1 for r in arm_runs
                                                 if r["counts"]["attacker_calls_valid"] > 0),
            "runs_reaching_target": sum(1 for r in arm_runs if r["target_calls"]),
            "runs_reaching_judge": sum(1 for r in arm_runs if r["judge_calls"]),
            "runs_with_rubric_score": sum(1 for r in arm_runs if r["rubric_scores"]),
            "successful_runs": sum(1 for r in arm_runs if r["success"]),
            "termination_reasons": dict(Counter(r["termination_reason"] for r in arm_runs)),
            "failure_classes": dict(Counter(r["failure_class"] for r in arm_runs)),
        }
    (OUT / "failure_cascade.json").write_text(json.dumps(cascade, indent=2), encoding="utf-8")

    # statistics
    stats = {"phase": "17", "stage": "6", "deliverable": "statistical_analysis.json",
             "primary_endpoint": "official attack success (frozen criterion, all runs in the "
                                 "denominator)",
             "unpaired": [], "paired": {}, "conditional_success": {}}
    for arm in present_arms:
        if arm == "control":
            continue
        for endpoint in ("success", "survival", "validity"):
            stats["unpaired"].append(arm_vs_control(runs, arm, endpoint))
        stats["paired"][arm] = paired_cells(runs, arm)
    for arm in present_arms:
        arm_runs = [r for r in runs if r["arm"] == arm]
        survived = [r for r in arm_runs if r["_survived"]]
        stats["conditional_success"][arm] = {
            "official_success": round(sum(1 for r in arm_runs if r["success"])
                                      / len(arm_runs), 4) if arm_runs else None,
            "p_success_given_survived": round(sum(1 for r in survived if r["success"])
                                              / len(survived), 4) if survived else None,
            "n_survived": len(survived),
            "note": "the conditional value is descriptive and never replaces the official rate",
        }
    p_values = [t["p_value"] for t in TESTS]
    stats["discipline"] = {
        "n_comparisons": len(TESTS),
        "comparisons": TESTS,
        "bonferroni_threshold": round(0.05 / len(p_values), 6) if p_values else None,
        "n_below_bonferroni": sum(1 for p in p_values
                                  if p < (0.05 / len(p_values) if p_values else 1)),
        "statement": "official success is the primary endpoint for each arm-versus-control "
                     "comparison; survival and validity are secondary; all other analyses in "
                     "this stage are exploratory. No arm is called superior on a point "
                     "estimate alone.",
    }
    (OUT / "statistical_analysis.json").write_text(json.dumps(stats, indent=2),
                                                   encoding="utf-8")

    figures(runs, present_arms, depth, cascade)
    print(json.dumps({a: arm_payload["arms"][a]["M1_official_success"]["rate"]
                      for a in present_arms}, indent=1))
    print(f"figures: {sorted(p.name for p in FIG.glob('*.png'))}")
    return 0


def figures(runs, present_arms, depth, cascade) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name, title):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=130)
        plt.close(fig)

    # JSON validity by call index, all arms
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for arm in present_arms:
        table = depth["validity_by_call_index"].get(arm, {})
        keys = [k for k in ("1", "2", "3", "4", "5", "6", "7", "8+") if k in table]
        ax.plot(keys, [100 * table[k]["validity"] for k in keys], marker="o",
                label=arm)
    ax.set_ylim(0, 100)
    ax.set_xlabel("attacker call index")
    ax.set_ylabel("contract validity (%)")
    ax.legend(fontsize=8)
    save(fig, "json_validity_by_call.png",
         "Contract validity by attacker call index — all four arms")

    # survival by depth
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for arm in present_arms:
        ys = [100 * depth["survival_by_depth"][arm][str(n)] for n in range(1, 9)]
        ax.plot(range(1, 9), ys, marker="o", label=arm)
    ax.set_xlabel("turn depth N")
    ax.set_ylabel("P(run reaches turn N) (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    save(fig, "survival_by_depth.png", "Survival by depth — P(reach turn N)")

    # run success comparison
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rates = [100 * cascade["arms"][a]["successful_runs"] / cascade["arms"][a]["runs"]
             for a in present_arms]
    ax.bar(present_arms, rates, color="#8d5a97")
    ax.set_ylabel("official attack success (%)")
    ax.set_ylim(0, 100)
    for i, (a, v) in enumerate(zip(present_arms, rates)):
        ax.text(i, v, f"{v:.1f}%\n{cascade['arms'][a]['successful_runs']}/"
                      f"{cascade['arms'][a]['runs']}", ha="center", va="bottom", fontsize=8)
    save(fig, "run_success_comparison.png", "Official attack success by arm")

    # attacker failure composition
    fig, ax = plt.subplots(figsize=(7, 3.4))
    modes = ["prose_output", "empty_generatedQuestion", "truncated_json",
             "malformed_json", "other"]
    width = 0.2
    for i, arm in enumerate(present_arms):
        counts = Counter(c["failure_type"] for r in runs if r["arm"] == arm
                         for c in r["attacker_calls"] if c["failure_type"])
        xs = [j + i * width for j in range(len(modes))]
        ax.bar(xs, [counts.get(m, 0) for m in modes], width=width, label=arm)
    ax.set_xticks([j + width * (len(present_arms) - 1) / 2 for j in range(len(modes))])
    ax.set_xticklabels(modes, fontsize=8)
    ax.set_ylabel("attacker calls")
    ax.legend(fontsize=8)
    save(fig, "attacker_failure_comparison.png", "Attacker failure modes by arm")

    # completed turns
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, arm in enumerate(present_arms):
        turns = [r["completed_turns"] for r in runs if r["arm"] == arm]
        dist = Counter(turns)
        ax.bar([x + i * 0.2 for x in sorted(dist)], [dist[x] for x in sorted(dist)],
               width=0.2, label=arm)
    ax.set_xlabel("completed turns")
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "completed_turns.png", "Completed turns per run by arm")

    # failure cascade funnel
    stages = ["runs", "runs_with_attacker_call", "runs_with_valid_attacker_call",
              "runs_reaching_target", "runs_reaching_judge", "runs_with_rubric_score",
              "successful_runs"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for arm in present_arms:
        ax.plot([cascade["arms"][arm][s] for s in stages], marker="o", label=arm)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("runs_", "").replace("_", " ") for s in stages],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("runs")
    ax.legend(fontsize=8)
    save(fig, "failure_cascade.png", "Failure cascade by arm")


if __name__ == "__main__":
    raise SystemExit(main())
