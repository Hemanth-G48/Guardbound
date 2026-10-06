"""Phase 17 Stage 4.5 — multi-turn qualification of a candidate (Parts 22-24).

Same three production attacks, the same frozen target and judge, the same
`max_turns=2` as the Stage 4 control, and the same measurement extraction. Only
the attacker configuration varies, and it is applied by this harness:

  * the prompt variant is injected into the attack module's prompt constant for
    the duration of the run and restored afterwards;
  * the candidate's sampling temperature is substituted at the call boundary
    (the attack passes 0.7; the frozen control's override equals 0.7 exactly);
  * the structured-decoding mode is a backend setting.

No retries, no repair, no reasoning stripping, no change to the attacks.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import os
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

VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
RESULTS_PATH = common.OUT_ROOT / "multiturn_results.jsonl"
RAW_DIR = common.OUT_ROOT / "multiturn"


def load_stage4_multiturn():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage4_multiturn_survival",
        REPO_ROOT / "scripts" / "phase17_stage4_multiturn_survival.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage4_multiturn_survival"] = module
    spec.loader.exec_module(module)
    return module


def load_stage4_isolated():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage4_isolated_reliability",
        REPO_ROOT / "scripts" / "phase17_stage4_isolated_reliability.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage4_isolated_reliability"] = module
    spec.loader.exec_module(module)
    return module


class CandidateAttacker:
    """Applies the candidate's sampling temperature and forwards telemetry."""

    def __init__(self, managed, temperature_override: float | None):
        self._managed = managed
        self._inner = getattr(managed, "_inner", None)
        self.temperature_override = temperature_override

    @property
    def name(self) -> str:
        return f"candidate[{self._managed.name}]"

    @property
    def structured_output_mode(self):
        return getattr(self._managed, "structured_output_mode", None)

    def generate(self, messages, temperature=0.7, max_turns_context=None,
                 json_format=False, structured_output_mode=None):
        effective = (self.temperature_override if self.temperature_override is not None
                     else temperature)
        return self._managed.generate(
            messages, temperature=effective, max_turns_context=max_turns_context,
            json_format=json_format, structured_output_mode=structured_output_mode,
        )


def build_stack(candidate: dict):
    from guardbound.llm.model_manager import ModelManager
    from guardbound.llm.provider_factory import build_role_llm

    manager = ModelManager(device="cuda")
    configs = {
        "attacker": {
            "provider": "local", "backend": "qwen38_native", "model": common.MODEL_ID,
            "revision": common.REVISION, "device_map": {"": 0}, "max_new_tokens": None,
            "top_p": candidate["top_p"], "quantization": "nf4",
            "residency": "sequential",
            "structured_output_mode": candidate.get("structured_output_mode"),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-variant", default="A0_control",
                        choices=sorted(common.PROMPT_VARIANTS))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--mode", default=None, choices=[None, "constrained_json"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=2)
    args = parser.parse_args()

    common.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stage4_mt = load_stage4_multiturn()
    stage4_iso = load_stage4_isolated()
    from guardbound.attacks.runner import run_attack_with_backtracking

    candidate = {
        "candidate_id": args.config,
        "prompt_variant": args.prompt_variant,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "structured_output_mode": args.mode,
    }
    print(f"candidate {args.config}: prompt={args.prompt_variant} temp={args.temperature} "
          f"top_p={args.top_p} mode={args.mode} runs={args.runs} max_turns={args.max_turns}",
          flush=True)

    manager, llms = build_stack(candidate)
    override = None if abs(args.temperature - 0.7) < 1e-9 else args.temperature
    attacker = CandidateAttacker(llms["attacker"], temperature_override=override)

    # The recording wrapper, attack construction and GOALS come from the Stage 2/4
    # harness so this qualification measures exactly what those did.
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location(
        "phase17_stage2_multiturn", REPO_ROOT / "scripts" / "phase17_stage2_multiturn.py")
    stage2_mt = _ilu.module_from_spec(spec)
    sys.modules["phase17_stage2_multiturn"] = stage2_mt
    spec.loader.exec_module(stage2_mt)

    attacker_rec = stage2_mt.RecordingRole("attacker", attacker)
    target_rec = stage2_mt.RecordingRole("target", llms["target"])
    evaluator_rec = stage2_mt.RecordingRole("evaluator", llms["evaluator"])

    attacks = list(common.ATTACKS)
    completed = 0
    try:
        with contextlib.ExitStack() as stack:
            for attack in attacks:
                stack.enter_context(
                    common.injected_prompt_variant(attack, args.prompt_variant))
            for repetition in range(args.runs):
                attack_key = attacks[repetition % len(attacks)]
                goal = common.GOALS[repetition % len(common.GOALS)]
                attacker_rec.reset()
                target_rec.reset()
                evaluator_rec.reset()
                manager.reset_vram_peak()
                attack = stage2_mt.build_attack(
                    attack_key, attacker_rec, evaluator_rec, args.max_turns)
                started = time.perf_counter()
                error = None
                conversation = None
                try:
                    conversation = run_attack_with_backtracking(
                        attack, goal=goal, target_llm=target_rec,
                        embed_fn=None, barrier=None, eta=0.0,
                        max_turns=args.max_turns, temperature=0.7,
                        target_llm_name="meta-llama/Meta-Llama-3-8B-Instruct",
                        attack_method=attack_key, allow_regeneration=False,
                        steer_target=False, system_prompt=None,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                wall = round(time.perf_counter() - started, 3)

                call_rows = []
                for call in attacker_rec.calls:
                    stats = call.get("generation_stats") or {}
                    raw = stats.get("raw_reply")
                    verdict = stage4_iso.classify_reply(raw if raw is not None else "")
                    call_rows.append({
                        "returned_type": call["returned_type"],
                        "category": verdict["category"],
                        "failure_shape": verdict["failure_shape"],
                        "empty_query": verdict["empty_query"],
                        "prompt_tokens": stats.get("prompt_tokens"),
                        "generated_tokens": stats.get("generated_tokens"),
                        "termination": stats.get("termination"),
                        "latency_s": call["wall_s"],
                        "generated_question": (verdict.get("generatedQuestion_preview")),
                        "raw_output": raw,
                    })
                valid_calls = [c for c in call_rows if c["category"] in VALID]
                empty_calls = [c for c in call_rows if c["empty_query"] is True]
                target_stats = [(c.get("generation_stats") or {}) for c in target_rec.calls]

                if error:
                    mode = "terminated_by_attacker_failure"
                elif empty_calls:
                    mode = "terminated_by_empty_query"
                elif (conversation and len(conversation.turns) >= args.max_turns):
                    mode = "survived_full_turn_budget"
                else:
                    mode = "ended_early_without_attacker_failure"

                row = {
                    "stage": "4.5",
                    "candidate_id": args.config,
                    "prompt_variant": args.prompt_variant,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "structured_output_mode": args.mode,
                    "max_turns_setting": args.max_turns,
                    "run_index": repetition + 1,
                    "attack": attack_key,
                    "goal": goal,
                    "run_survived": error is None,
                    "termination_mode": mode,
                    "termination_reason": error or "completed_without_attacker_failure",
                    "turns_completed": len(conversation.turns) if conversation else 0,
                    "attacker_calls_attempted": len(call_rows),
                    "attacker_calls_valid": len(valid_calls),
                    "attacker_calls_invalid": len(call_rows) - len(valid_calls),
                    "attacker_calls_usable": len(valid_calls) - len(empty_calls),
                    "target_calls": len(target_rec.calls),
                    "target_eos_count": sum(1 for s in target_stats
                                            if s.get("termination") == "eos"),
                    "judge_calls": len(evaluator_rec.calls),
                    "judge_failures": sum(1 for c in evaluator_rec.calls
                                          if c["returned_type"] not in ("dict",)),
                    "peak_vram_gib": round(manager.peak_vram_gb(), 3),
                    "wall_s": wall,
                    "attacker_calls": call_rows,
                }
                with RESULTS_PATH.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                (RAW_DIR / f"{args.config}_{attack_key}_run{repetition + 1:02d}.json").write_text(
                    json.dumps(conversation.to_dict() if conversation else {"error": error},
                               indent=2), encoding="utf-8")
                completed += 1
                print(f"  [{completed}/{args.runs}] {attack_key} survived={error is None} "
                      f"mode={mode} turns={row['turns_completed']} "
                      f"calls={row['attacker_calls_valid']}/{row['attacker_calls_attempted']} "
                      f"usable={row['attacker_calls_usable']} wall={wall}s", flush=True)
    finally:
        manager.unload_all()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    print(f"\ncandidate {args.config}: {completed} runs recorded in "
          f"{RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
