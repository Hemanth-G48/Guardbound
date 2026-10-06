"""Phase 17 Stage 7 — C1 generation-point contract qualification.

Two arms only (frozen control vs the Stage 6 C1 intervention), goal-major execution so
every complete goal block is a matched C0/C1 pair set.

The intervention itself is **imported from the Stage 6 module** (`ArmAttacker`, `ARMS`) so
the wording and its placement as a final user message cannot drift between the two studies.
Everything else is the frozen Stage 4.8 stack, imported from the pilot runner.

Adds the Stage 7 §13 instrumentation the Stage 6 runner did not carry: per-role VRAM
before/after each generation, maximum context length, and full OOM preservation (the run
stays in the denominator and is never retried).

No retries, no repair, no extraction from prose, no fallback, no reasoning stripping.

Usage:
    python scripts/phase17_stage7_c1_qualification.py --goals 10 --goal-start 0
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = _load("phase17_stage45_common", "scripts/phase17_stage45_common.py")
stage2 = _load("phase17_stage2_multiturn", "scripts/phase17_stage2_multiturn.py")
stage4 = _load("phase17_stage4_isolated_reliability",
               "scripts/phase17_stage4_isolated_reliability.py")
stage46 = _load("phase17_stage46_qualification", "scripts/phase17_stage46_qualification.py")
phase14 = _load("phase14_full_reproduction", "scripts/phase14_full_reproduction.py")
pilot = _load("phase17_stage47_pilot", "scripts/phase17_stage47_pilot.py")
# The C1 intervention is imported, never re-implemented.
stage6 = _load("phase17_stage6_run_level_contract",
               "scripts/phase17_stage6_run_level_contract.py")

import torch  # noqa: E402

OUT = REPO_ROOT / "results" / "phase17_pilot" / "analysis" / "stage7_c1_qualification"
RAW = OUT / "raw"
RUNS_OUT = OUT / "run_level_metrics.jsonl"
CALLS_OUT = OUT / "call_level_metrics.jsonl"
DATASET = REPO_ROOT / phase14.OFFICIAL_DATASET
ATTACKS = list(phase14.ATTACK_ORDER)
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")

# Two arms, using the Stage 6 definitions verbatim.
ARMS = {
    "control": stage6.ARMS["control"],
    "c1_generation_point": {
        "mode": stage6.ARMS["generation_point"]["mode"],
        "temperature": stage6.ARMS["generation_point"]["temperature"],
        "label": "C1 generation-point instruction (Stage 6 intervention, unchanged)",
    },
}
assert ARMS["c1_generation_point"]["mode"] == "I1"
assert ARMS["control"]["mode"] == "A0"


def vram_gb() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gib": None, "reserved_gib": None}
    return {"allocated_gib": round(torch.cuda.memory_allocated() / 1024 ** 3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 3)}


class ProbedRole:
    """Forwards to a role and samples VRAM around every generation (§13)."""

    def __init__(self, role: str, inner, samples: list[dict]):
        self._role = role
        self._inner = inner
        self._samples = samples
        self.calls = inner.calls

    @property
    def name(self):
        return getattr(self._inner, "name", self._role)

    @property
    def structured_output_mode(self):
        return getattr(self._inner, "structured_output_mode", None)

    def reset(self):
        return self._inner.reset()

    def generate(self, messages, temperature=0.7, max_turns_context=None,
                 json_format=False, structured_output_mode=None):
        before = vram_gb()
        call_index = len(self.calls) + 1
        result = self._inner.generate(
            messages, temperature=temperature, max_turns_context=max_turns_context,
            json_format=json_format, structured_output_mode=structured_output_mode)
        after = vram_gb()
        stats = (self.calls[-1].get("generation_stats") or {}) if self.calls else {}
        self._samples.append({
            "role": self._role, "call_index": call_index,
            "vram_before_allocated_gib": before["allocated_gib"],
            "vram_after_allocated_gib": after["allocated_gib"],
            "vram_delta_gib": (round(after["allocated_gib"] - before["allocated_gib"], 3)
                               if before["allocated_gib"] is not None else None),
            "vram_reserved_after_gib": after["reserved_gib"],
            "prompt_tokens": stats.get("prompt_tokens"),
            "generated_tokens": stats.get("generated_tokens"),
        })
        return result


def run_cell(arm_name: str, arm: dict, attack_key: str, goal_record: dict, goal_id: int,
             llms, manager, max_turns: int) -> dict:
    from guardbound.attacks import rubric_evaluation
    from guardbound.attacks.runner import run_attack_with_backtracking

    attack_short = phase14.ATTACK_KEYS[attack_key]
    run_id = f"{arm_name}_{attack_short}_{goal_id:03d}"
    goal = goal_record["task"]
    target_system = goal_record.get("target_system") or None
    seed = phase14.derive_seed(goal_id, attack_short, "off")
    phase14._seed_all(seed)

    attacker = stage6.ArmAttacker(llms["attacker"], arm["mode"], arm["temperature"])
    samples: list[dict] = []
    attacker_rec = ProbedRole("attacker", stage2.RecordingRole("attacker", attacker), samples)
    target_rec = ProbedRole("target", stage2.RecordingRole("target", llms["target"]), samples)
    evaluator_rec = ProbedRole("evaluator",
                               stage2.RecordingRole("evaluator", llms["evaluator"]), samples)
    rubric_evaluation.reset_evaluator_telemetry()
    manager.reset_vram_peak()
    vram_at_start = vram_gb()

    attack = stage2.build_attack(attack_key, attacker_rec, evaluator_rec, max_turns)
    started_at = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    error = None
    error_traceback = None
    conversation = None
    try:
        conversation = run_attack_with_backtracking(
            attack, goal=goal, target_llm=target_rec, embed_fn=None, barrier=None,
            eta=0.0, max_turns=max_turns, temperature=0.7,
            target_llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
            attack_method=attack_key, allow_regeneration=False, steer_target=False,
            system_prompt=target_system)
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        error_traceback = traceback.format_exc()[-2000:]
    duration = round(time.time() - started_at, 3)
    vram_end = vram_gb()

    attacker_calls = [stage46.call_record(c, stage4) for c in attacker_rec.calls]
    valid_calls = [c for c in attacker_calls if c["category"] in VALID]
    failed_calls = [c for c in attacker_calls if c["failure_type"] is not None]
    events = rubric_evaluation.evaluator_events()
    scores = [s for s in (getattr(attack, "_scores", []) or []) if s != "refused"]
    success = bool(attack.was_successful()) if hasattr(attack, "was_successful") else False
    if success:
        termination_reason = "success_score_5"
    elif hasattr(attack, "should_backtrack") and not attack.should_backtrack():
        termination_reason = "refusal_retry_limit"
    elif conversation is not None and len(conversation.turns) >= max_turns:
        termination_reason = "max_turns_reached"
    else:
        termination_reason = ("attack_loop_exhausted" if error is None
                              else "attacker_generation_error")

    lowered = (error or "").lower()
    is_oom = "out of memory" in lowered or "cuda oom" in lowered
    failure_class = ("cuda_oom_failure" if is_oom else
                     "attacker_generation_error" if error else
                     "attacker_output_unusable" if failed_calls else "natural_termination")

    target_calls = stage46.role_calls(target_rec)
    judge_calls = stage46.role_calls(evaluator_rec)
    context_lengths = [s["prompt_tokens"] for s in samples if s.get("prompt_tokens")]

    return {
        "run_id": run_id, "phase": "17", "stage": "7", "arm": arm_name,
        "arm_label": arm["label"], "mode": arm["mode"],
        "attack": attack_key, "attack_short": attack_short,
        "goal_id": goal_id, "seed": seed, "goal": goal,
        "temperature": arm["temperature"], "top_p": 1.0, "max_turns": max_turns,
        "attacker_revision": common.REVISION,
        "target_revision": "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2",
        "judge_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "start_time": started_iso, "end_time": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration,
        "completed_turns": len(conversation.turns) if conversation else 0,
        "turns": ([{"query": t.query, "response": t.response} for t in conversation.turns]
                  if conversation else []),
        "success": success, "final_score": scores[-1] if scores else None,
        "rubric_scores": [e.score for e in events],
        "rubric_outcomes": [e.outcome for e in events],
        "refusal_count": attack.get_refusal_count() if hasattr(attack, "get_refusal_count") else None,
        "termination_reason": termination_reason, "failure_class": failure_class,
        "error": error,
        "attacker_calls": attacker_calls,
        "target_calls": target_calls,
        "judge_calls": judge_calls,
        "counts": {
            "attacker_calls": len(attacker_calls),
            "attacker_calls_valid": len(valid_calls),
            "attacker_calls_usable": len([c for c in valid_calls
                                          if c["failure_type"] is None]),
            "attacker_calls_failed": len(failed_calls),
            "target_calls": len(target_rec.calls),
            "target_eos": sum(1 for c in target_calls if c.get("termination") == "eos"),
            "judge_calls": len(evaluator_rec.calls),
            "judge_failures": sum(1 for e in events if not e.valid),
        },
        "failure_types": dict(_counter(failed_calls)),
        "peak_vram_gib": round(manager.peak_vram_gb(), 3),
        "vram_at_start": vram_at_start, "vram_at_end": vram_end,
        "vram_samples": samples,
        "max_context_tokens": max(context_lengths) if context_lengths else None,
        "min_context_tokens": min(context_lengths) if context_lengths else None,
        "oom_detail": ({"error": error, "traceback_tail": error_traceback,
                        "last_context_tokens": context_lengths[-1] if context_lengths else None,
                        "max_context_tokens": max(context_lengths) if context_lengths else None,
                        "attacker_calls_before_failure": len(attacker_calls),
                        "target_calls_before_failure": len(target_rec.calls),
                        "judge_calls_before_failure": len(evaluator_rec.calls),
                        "retried": False, "in_denominator": True} if is_oom else None),
        "prompt_hash": None,
        "success_criterion": "attack.was_successful() (rubric score 5 = unsafe = success)",
    }


def _counter(items) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        key = item.get("failure_type")
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goals", type=int, default=30)
    parser.add_argument("--goal-start", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    # Anti-duplication guard, cell-precise: refuse only if a requested
    # (goal, attack, arm) cell already exists. Disjoint goal ranges can therefore be
    # appended in blocks; a cell can never be run twice.
    existing = set()
    if RUNS_OUT.is_file() and RUNS_OUT.stat().st_size:
        for line in RUNS_OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing.add((row["goal_id"], row["attack"], row["arm"]))
    if RUNS_OUT.is_file() and RUNS_OUT.stat().st_size:
        print(f"continuing: {len(existing)} cells already present; only new cells will run",
              flush=True)

    manifest = stage6.baseline_manifest()
    manifest.update({"deliverable": "manifest.json", "stage": "7",
                     "arms": {k: v for k, v in ARMS.items()},
                     "c1_intervention_source": "imported from "
                                               "scripts/phase17_stage6_run_level_contract.py "
                                               "(ArmAttacker, mode I1)",
                     "matrix_intended": {"arms": len(ARMS), "attacks": len(ATTACKS),
                                         "goals": 30, "runs": 180, "matched_cells": 90}})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    dataset_sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if dataset_sha != phase14.OFFICIAL_DATASET_SHA256:
        print("BLOCKED: dataset hash mismatch", flush=True)
        return 2
    goals = json.loads(DATASET.read_text(encoding="utf-8"))
    goals = goals[args.goal_start: args.goal_start + args.goals]
    arm_names = list(ARMS)
    total = len(goals) * len(ATTACKS) * len(arm_names)
    print(f"stage7: {len(arm_names)} arms x {len(ATTACKS)} attacks x {len(goals)} goals "
          f"(goal ids {args.goal_start}-{args.goal_start + len(goals) - 1}) = {total} runs "
          f"| max_turns={args.max_turns} | NBF off", flush=True)

    manager, llms = pilot.build_stack("stage7")
    inner = getattr(llms["attacker"], "_inner", None)
    if inner is not None and hasattr(inner, "_get_pipeline"):
        inner._get_pipeline()
    print("stack loaded", flush=True)

    prompt_hashes = manifest["a0_prompt_sha256"]
    durations: list[float] = []
    completed = 0
    per_arm_done: dict[str, int] = {a: 0 for a in arm_names}
    started = time.time()
    try:
        for offset, goal_record in enumerate(goals):
            goal_id = args.goal_start + offset
            for attack_key in ATTACKS:
                for arm_name in arm_names:
                    if (goal_id, attack_key, arm_name) in existing:
                        continue          # already measured; never re-run
                    run = run_cell(arm_name, ARMS[arm_name], attack_key, goal_record,
                                   goal_id, llms, manager, args.max_turns)
                    run["prompt_hash"] = prompt_hashes[attack_key]
                    with RUNS_OUT.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
                    per_arm_dir = RAW / arm_name
                    per_arm_dir.mkdir(parents=True, exist_ok=True)
                    with (per_arm_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(run, ensure_ascii=False, default=str) + "\n")
                    for index, call in enumerate(run["attacker_calls"]):
                        row = dict(call)
                        row.update({"run_id": run["run_id"], "arm": arm_name,
                                    "attack": attack_key, "goal_id": goal_id,
                                    "seed": run["seed"], "call_index": index + 1,
                                    "turn_index": None,
                                    "temperature": run["temperature"],
                                    "top_p": run["top_p"],
                                    "prompt_hash": prompt_hashes[attack_key],
                                    "model_revision": common.REVISION,
                                    "input_context_tokens": call.get("prompt_tokens"),
                                    "output_tokens": call.get("generated_tokens"),
                                    "output_characters": (len(call["raw_output"])
                                                          if isinstance(call.get("raw_output"), str)
                                                          else None),
                                    "json_valid": call["category"] in VALID,
                                    "semantic_valid": (call["category"] in VALID
                                                       and call.get("empty_query") is not True)})
                        with CALLS_OUT.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    durations.append(run["duration_s"])
                    completed += 1
                    per_arm_done[arm_name] = per_arm_done.get(arm_name, 0) + 1
                    elapsed = time.time() - started
                    eta = statistics.mean(durations) * (total - completed) if durations else None
                    (OUT / "progress.json").write_text(json.dumps({
                        "total_runs": total, "completed_runs": completed,
                        "elapsed_seconds": round(elapsed, 1),
                        "estimated_remaining_seconds": round(eta, 1) if eta else None,
                        "mean_run_seconds": round(statistics.mean(durations), 1) if durations else None,
                        "current": {"goal_id": goal_id, "attack": attack_key, "arm": arm_name},
                        "completed_by_arm": dict(per_arm_done),
                        "goal_range": [args.goal_start, args.goal_start + len(goals) - 1],
                        "observability_only": "no resume, no retry, no skipping",
                        "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2),
                        encoding="utf-8")
                    flag = "" if not run["failure_types"] else f" {run['failure_types']}"
                    print(f"  [{completed}/{total}] {run['run_id']:34s} turns="
                          f"{run['completed_turns']} success={run['success']} "
                          f"calls={run['counts']['attacker_calls_valid']}/"
                          f"{run['counts']['attacker_calls']} "
                          f"{run['failure_class']} {run['duration_s']:.0f}s "
                          f"maxctx={run['max_context_tokens']} "
                          f"peak={run['peak_vram_gib']}{flag}", flush=True)
    finally:
        manager.unload_all()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nstage7: {completed}/{total} runs -> {RUNS_OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
