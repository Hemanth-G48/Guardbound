"""Phase 17 Stage 4.6 — B2 qualification runner.

Runs one arm of the qualification (control or B2) over the production attacks
with the frozen target and judge, and records everything the qualification needs:
paired seeds, per-call classification, the brief's failure taxonomy, raw outputs,
rubric telemetry from the project's own evaluator, latency/token/VRAM counters.

Nothing downstream is modified. The two arms differ in exactly two declared
values: the attacker's sampling temperature and the A1 output-contract suffix
appended to the frozen production prompt. Everything else is shared code.

  control : temperature 0.7, frozen prompt, structured decoding off
  b2      : temperature 0.3, frozen prompt + A1 suffix, structured decoding off

Usage:
  python scripts/phase17_stage46_qualification.py --arm control --runs 30 --max-turns 2
  python scripts/phase17_stage46_qualification.py --arm b2      --runs 30 --max-turns 2
  python scripts/phase17_stage46_qualification.py --arm control --runs 10 --max-turns 8
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_spec = importlib.util.spec_from_file_location(
    "phase17_stage45_common", REPO_ROOT / "scripts" / "phase17_stage45_common.py")
common = importlib.util.module_from_spec(_spec)
sys.modules["phase17_stage45_common"] = common
_spec.loader.exec_module(common)

import torch  # noqa: E402

OUT_ROOT = REPO_ROOT / "results" / "phase17_model_optimization" / "stage4_6_b2_qualification"
RAW_ROOT = OUT_ROOT / "raw"
RESULTS_PATH = OUT_ROOT / "multiturn_results.jsonl"

VALID_CATEGORIES = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
SEED_BASE = 4286

ARM_CONFIG = {
    "control": {
        "arm": "control",
        "temperature": 0.7,
        "prompt_variant": "A0_control",
        "prompt_suffix": None,
        "prompt_role": "frozen Stage 4 production prompt",
        "structured_output_mode": None,
        "top_p": 1.0,
    },
    "b2": {
        "arm": "b2",
        "temperature": 0.3,
        "prompt_variant": "A1_json_object_statement",
        "prompt_suffix": (
            "\n\n###Output Format (required)###\n"
            "Your entire response must be a single JSON object. "
            "Do not include any text before or after it."
        ),
        "prompt_role": "frozen Stage 4 production prompt + the A1 output-contract suffix",
        "structured_output_mode": None,
        "top_p": 1.0,
    },
}

# §11 taxonomy: one label per failed attacker call.
UNQUOTED_KEY_RE = re.compile(r"\{\s*[A-Za-z_][A-Za-z0-9_]*\s*:")


def failure_type(category: str, shape: str | None, verdict: dict, raw: str | None) -> str | None:
    """Map the project's own classification onto the brief's failure taxonomy.

    This is a *labelling* of an already-classified failure.  It never parses,
    repairs or rewrites the output: the unquoted-key pattern is a diagnostic on
    the raw text only, and the parse chain remains the production one.
    """
    if category in VALID_CATEGORIES:
        return "empty_generatedQuestion" if verdict.get("empty_query") is True else None
    if category == "EMPTY_OUTPUT":
        return "empty_output"
    if category == "TRUNCATED_OUTPUT":
        return "truncated_json"
    if category == "MISSING_GENERATED_QUESTION":
        return "missing_generatedQuestion"
    if category == "MISSING_LAST_RESPONSE_SUMMARY":
        return "missing_lastResponseSummary"
    if category == "INVALID_FIELD_TYPE":
        return "wrong_field_type"
    if category == "JSON_PARSE_ERROR":
        if shape == "prose":
            return "prose_output"
        if shape == "reasoning_only":
            return "other_reasoning_only"
        if shape == "malformed_json":
            return ("unquoted_json_key" if UNQUOTED_KEY_RE.search(raw or "")
                    else "malformed_json")
        return "other"
    return "other"


def load_stage4():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage4_isolated_reliability",
        REPO_ROOT / "scripts" / "phase17_stage4_isolated_reliability.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage4_isolated_reliability"] = module
    spec.loader.exec_module(module)
    return module


def load_stage2_harness():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage2_multiturn", REPO_ROOT / "scripts" / "phase17_stage2_multiturn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage2_multiturn"] = module
    spec.loader.exec_module(module)
    return module


class SeededAttacker:
    """Applies the arm's temperature and a per-(run, call) paired seed.

    The seed is set immediately before each attacker generation, so the same
    (run, call) cell draws from the same RNG state in both arms.  The target and
    judge are left exactly as production runs them — unseeded — because their
    inputs differ between arms as soon as the questions differ, so seeding them
    would add no pairing while changing how the frozen models are driven.
    """

    def __init__(self, managed, temperature_override: float | None, seed_base: int):
        self._managed = managed
        self._inner = getattr(managed, "_inner", None)
        self.temperature_override = temperature_override
        self.seed_base = seed_base
        self.run_index = 0
        self.call_index = 0
        self.seeds: list[int] = []

    @property
    def name(self) -> str:
        return f"seeded[{self._managed.name}]"

    @property
    def structured_output_mode(self):
        return getattr(self._managed, "structured_output_mode", None)

    def generate(self, messages, temperature=0.7, max_turns_context=None,
                 json_format=False, structured_output_mode=None):
        seed = self.seed_base + 1000 * self.run_index + self.call_index
        torch.manual_seed(seed)
        self.seeds.append(seed)
        self.call_index += 1
        effective = (self.temperature_override if self.temperature_override is not None
                     else temperature)
        return self._managed.generate(
            messages, temperature=effective, max_turns_context=max_turns_context,
            json_format=json_format, structured_output_mode=structured_output_mode,
        )


def build_stack(arm_config: dict):
    from guardbound.llm.model_manager import ModelManager
    from guardbound.llm.provider_factory import build_role_llm

    manager = ModelManager(device="cuda")
    configs = {
        "attacker": {
            "provider": "local", "backend": "qwen38_native", "model": common.MODEL_ID,
            "revision": common.REVISION, "device_map": {"": 0}, "max_new_tokens": None,
            "top_p": arm_config["top_p"], "quantization": "nf4", "residency": "sequential",
            "structured_output_mode": arm_config["structured_output_mode"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "target": {
            "provider": "local", "backend": "hf_local",
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "device_map": {"": 0}, "max_new_tokens": None, "temperature": 0.7,
            "top_p": 1.0, "do_sample": True, "residency": "sequential",
        },
        "evaluator": {
            "provider": "local", "backend": "hf_local", "model": "Qwen/Qwen3-8B",
            "device_map": {"": 0}, "max_new_tokens": None, "temperature": 0.0,
            "do_sample": False, "residency": "sequential",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    llms = {role: build_role_llm(cfg, role, manager=manager) for role, cfg in configs.items()}
    return manager, llms


def call_record(call: dict, stage4) -> dict:
    stats = call.get("generation_stats") or {}
    raw = stats.get("raw_reply")
    verdict = stage4.classify_reply(raw if raw is not None else "")
    return {
        "returned_type": call["returned_type"],
        "error": call.get("error"),
        "category": verdict["category"],
        "failure_shape": verdict["failure_shape"],
        "failure_type": failure_type(verdict["category"], verdict["failure_shape"], verdict, raw),
        "direct_json": verdict["direct_json"],
        "extracted_json": verdict["extracted_json"],
        "has_generatedQuestion": verdict["has_generatedQuestion"],
        "has_lastResponseSummary": verdict["has_lastResponseSummary"],
        "generatedQuestion_type": verdict["generatedQuestion_type"],
        "lastResponseSummary_type": verdict["lastResponseSummary_type"],
        "empty_query": verdict["empty_query"],
        "reasoning_leakage": verdict["reasoning_leakage"],
        "generated_question": (verdict.get("parsed") or {}).get("generatedQuestion"),
        "last_response_summary": (verdict.get("parsed") or {}).get("lastResponseSummary"),
        "raw_output": raw,
        "prompt_hash": call.get("system_prompt_sha256"),
        "prompt_chars": call.get("system_prompt_chars"),
        "temperature": call.get("temperature"),
        "temperature_requested_by_attack": call.get("temperature_requested_by_attack"),
        "json_format_requested": call.get("json_format_requested"),
        "prompt_tokens": stats.get("prompt_tokens"),
        "generated_tokens": stats.get("generated_tokens"),
        "termination": stats.get("termination"),
        "latency_s": call.get("latency_s") or call.get("wall_s"),
        "wall_s": call.get("wall_s"),
    }


def role_calls(role_rec, stage4=None) -> list[dict]:
    out = []
    for call in role_rec.calls:
        stats = call.get("generation_stats") or {}
        row = {
            "returned_type": call["returned_type"],
            "error": call.get("error"),
            "latency_s": call.get("latency_s") or call.get("wall_s"),
            "wall_s": call.get("wall_s"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "generated_tokens": stats.get("generated_tokens"),
            "termination": stats.get("termination"),
            "raw_output": stats.get("raw_reply"),
            "prompt_hash": call.get("system_prompt_sha256"),
        }
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["control", "b2"])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--phase", default=None, help="label; defaults to runs × turns")
    args = parser.parse_args()

    arm = ARM_CONFIG[args.arm]
    phase = args.phase or f"{args.runs}runs_{args.max_turns}turns"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / args.arm).mkdir(parents=True, exist_ok=True)
    (RAW_ROOT / args.arm).mkdir(parents=True, exist_ok=True)

    stage4 = load_stage4()
    stage2 = load_stage2_harness()
    from guardbound.attacks.runner import run_attack_with_backtracking
    from guardbound.attacks import rubric_evaluation

    config = {
        "arm": args.arm,
        "temperature": arm["temperature"],
        "top_p": arm["top_p"],
        "prompt": arm["prompt_role"],
        "prompt_variant": arm["prompt_variant"],
        "prompt_suffix": arm["prompt_suffix"],
        "structured_output_mode": arm["structured_output_mode"],
        "model_id": common.MODEL_ID,
        "revision": common.REVISION,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "loader": "AutoModelForImageTextToText",
        "processor": "Qwen3VLProcessor",
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
        "enable_thinking": False,
        "max_new_tokens": None,
        "max_turns": args.max_turns,
        "seed_base": args.seed_base,
        "seed_schedule": "seed_base + 1000*run_index + call_index, set before each "
                         "attacker generation; target and judge unseeded as in production",
        "runs": args.runs,
    }
    (OUT_ROOT / args.arm / "manifest.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(f"arm={args.arm} temperature={arm['temperature']} prompt={arm['prompt_variant']} "
          f"runs={args.runs} max_turns={args.max_turns}", flush=True)

    load_started = time.perf_counter()
    manager, llms = build_stack(arm)
    inner = getattr(llms["attacker"], "_inner", None)
    if inner is not None and hasattr(inner, "_get_pipeline"):
        inner._get_pipeline()          # warm the attacker so load time is measured here
    load_seconds = round(time.perf_counter() - load_started, 2)
    print(f"stack loaded in {load_seconds}s; "
          f"vram={torch.cuda.memory_allocated() / 1024 ** 3:.3f} GiB", flush=True)

    override = None if abs(arm["temperature"] - 0.7) < 1e-9 else arm["temperature"]
    attacker = SeededAttacker(llms["attacker"], temperature_override=override,
                              seed_base=args.seed_base)
    attacker_rec = stage2.RecordingRole("attacker", attacker)
    target_rec = stage2.RecordingRole("target", llms["target"])
    evaluator_rec = stage2.RecordingRole("evaluator", llms["evaluator"])

    attacks = list(common.ATTACKS)
    per_arm_path = OUT_ROOT / args.arm / f"runs_{phase}.jsonl"
    completed = 0
    try:
        with contextlib.ExitStack() as stack:
            for attack in attacks:
                stack.enter_context(
                    common.injected_prompt_variant(attack, arm["prompt_variant"]))
            for run_index in range(args.runs):
                attack_key = attacks[run_index % len(attacks)]
                goal = common.GOALS[run_index % len(common.GOALS)]
                goal_id = run_index % len(common.GOALS)
                run_id = f"{args.arm}_{attack_key}_{run_index + 1:02d}"
                attacker.run_index = run_index
                attacker.call_index = 0
                attacker.seeds = []
                attacker_rec.reset()
                target_rec.reset()
                evaluator_rec.reset()
                rubric_evaluation.reset_evaluator_telemetry()
                manager.reset_vram_peak()
                attack = stage2.build_attack(attack_key, attacker_rec, evaluator_rec,
                                             args.max_turns)
                started = time.perf_counter()
                error = None
                conversation = None
                try:
                    conversation = run_attack_with_backtracking(
                        attack, goal=goal, target_llm=target_rec, embed_fn=None,
                        barrier=None, eta=0.0, max_turns=args.max_turns, temperature=0.7,
                        target_llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
                        attack_method=attack_key, allow_regeneration=False,
                        steer_target=False, system_prompt=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                wall = round(time.perf_counter() - started, 3)
                peak_vram = round(manager.peak_vram_gb(), 3)

                attacker_calls = [call_record(c, stage4) for c in attacker_rec.calls]
                valid = [c for c in attacker_calls if c["category"] in VALID_CATEGORIES]
                unusable = [c for c in valid if c["failure_type"] is not None]
                failed = [c for c in attacker_calls if c["failure_type"] is not None]
                events = rubric_evaluation.evaluator_events()

                if error:
                    mode = "attacker_failure"
                elif any(c["failure_type"] == "empty_generatedQuestion" for c in attacker_calls):
                    mode = "empty_query_termination"
                elif conversation and len(conversation.turns) >= args.max_turns:
                    mode = "full_turn_budget"
                else:
                    mode = "ended_early_no_attacker_failure"

                row = {
                    "stage": "4.6",
                    "arm": args.arm,
                    "run_id": run_id,
                    "phase": phase,
                    "max_turns": args.max_turns,
                    "run_index": run_index + 1,
                    "attack": attack_key,
                    "goal": goal,
                    "goal_id": goal_id,
                    "seed_base": args.seed_base,
                    "attacker_seeds": attacker.seeds,
                    "configuration": config,
                    "launched": True,
                    "run_survived": error is None,
                    "attacker_termination": error is not None,
                    "termination_mode": mode,
                    "termination_reason": error or "completed_without_attacker_failure",
                    "usable_run_strict": (bool(attacker_calls) and not failed),
                    "usable_run_any_turn": any(c["category"] in VALID_CATEGORIES
                                               and c["failure_type"] is None
                                               for c in attacker_calls),
                    "turns_completed": len(conversation.turns) if conversation else 0,
                    "attacker_calls_attempted": len(attacker_calls),
                    "attacker_calls_valid": len(valid),
                    "attacker_calls_unusable": len(unusable),
                    "attacker_calls_failed": len(failed),
                    "target_calls": len(target_rec.calls),
                    "target_eos_count": sum(
                        1 for c in role_calls(target_rec)
                        if c.get("termination") == "eos"),
                    "judge_calls": len(evaluator_rec.calls),
                    "judge_failures": sum(1 for e in events if not e.valid),
                    "rubric_scores": [
                        {"score": e.score, "outcome": e.outcome, "valid": e.valid}
                        for e in events
                    ],
                    "peak_vram_gib": peak_vram,
                    "load_seconds": load_seconds,
                    "wall_s": wall,
                    "attacker_calls": attacker_calls,
                    "target_calls_detail": role_calls(target_rec),
                    "judge_calls_detail": role_calls(evaluator_rec),
                }
                for handle in (RESULTS_PATH, per_arm_path):
                    with handle.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                (RAW_ROOT / args.arm /
                 f"{run_id}_{args.max_turns}turns.json").write_text(
                    json.dumps(conversation.to_dict() if conversation else {"error": error},
                               indent=2), encoding="utf-8")
                completed += 1
                print(f"  [{completed}/{args.runs}] {run_id} survived={error is None} "
                      f"mode={mode} turns={row['turns_completed']} "
                      f"calls={row['attacker_calls_valid']}/{row['attacker_calls_attempted']} "
                      f"failed={row['attacker_calls_failed']} "
                      f"usable_strict={row['usable_run_strict']} wall={wall}s "
                      f"{'' if not failed else [f['failure_type'] for f in failed]}",
                      flush=True)
    finally:
        manager.unload_all()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"released; vram={torch.cuda.memory_allocated() / 1024 ** 3:.3f} GiB", flush=True)

    print(f"\narm {args.arm}: {completed} runs -> {per_arm_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
