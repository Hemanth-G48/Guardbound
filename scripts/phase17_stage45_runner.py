"""Phase 17 Stage 4.5 — optimization runner (Stages A / B / C).

One configuration per invocation, so every candidate is explicitly defined and
documented rather than searched for automatically. What varies is exactly one
dimension per stage:

  A  prompt / output-contract variant
  B  sampling parameters
  C  structured (grammar-constrained) decoding

Everything else stays at the frozen Stage 4 values: same model, revision,
architecture, processor, NF4 quantization, BF16 compute, `max_new_tokens` UNSET,
`enable_thinking=False`, and the same production attack step functions.

Per call it records the raw output, the classification produced by the Stage 4
classifier (imported, not re-implemented), the exact system-prompt hash, the seed,
latency, generated tokens and termination. No retries, no repair, no reasoning
stripping.
"""
from __future__ import annotations

import argparse
import hashlib
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
    "phase17_stage45_common", REPO_ROOT / "scripts" / "phase17_stage45_common.py"
)
common = importlib.util.module_from_spec(_spec)
sys.modules["phase17_stage45_common"] = common
_spec.loader.exec_module(common)

import torch  # noqa: E402

STAGE_PREFIX = {
    "A": "prompt_variant",
    "B": "sampling_variant",
    "C": "structured_decoding",
}
DEFAULT_SEED_BASE = 1729


def write_manifest(prefix: str, record: dict) -> Path:
    path = common.OUT_ROOT / f"{prefix}_manifest.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "phase": "17", "stage": "4.5", "deliverable": f"{prefix}_manifest.json",
        "control": {
            "description": "Stage 4 frozen configuration",
            "model_id": common.MODEL_ID, "revision": common.REVISION,
            "architecture": "Qwen3_5ForConditionalGeneration",
            "loader": "AutoModelForImageTextToText",
            "processor": "Qwen3VLProcessor",
            "quantization": "nf4", "compute_dtype": "bfloat16", "double_quant": True,
            "temperature": 0.7, "top_p": 1.0, "max_new_tokens": None,
            "enable_thinking": False,
        },
        "candidates": {},
    }
    existing["candidates"][record["candidate_id"]] = record
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["A", "B", "C"])
    parser.add_argument("--config", required=True, help="candidate id, e.g. A1 / B2 / C0")
    parser.add_argument("--prompt-variant", default="A0_control",
                        choices=sorted(common.PROMPT_VARIANTS))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--mode", default=None, choices=[None, "constrained_json"],
                        help="structured_decoding_mode; None is the frozen behaviour")
    parser.add_argument("--calls", type=int, default=45)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--no-seed", action="store_true",
                        help="leave sampling unseeded (the production behaviour)")
    args = parser.parse_args()

    prefix = STAGE_PREFIX[args.stage]
    stage_dir = common.OUT_ROOT / {
        "A": "prompt_variants", "B": "sampling_variants",
        "C": "structured_decoding",
    }[args.stage]
    stage_dir.mkdir(parents=True, exist_ok=True)
    results_path = common.OUT_ROOT / f"{prefix}_results.jsonl"

    classifier = common.load_stage4_classifier()
    client = common.build_client(args.temperature, args.top_p, args.mode)
    print(f"candidate {args.config}: prompt={args.prompt_variant} "
          f"temperature={args.temperature} top_p={args.top_p} mode={args.mode} "
          f"calls={args.calls} seeded={not args.no_seed}", flush=True)
    load_started = time.perf_counter()
    client._get_pipeline()
    load_seconds = round(time.perf_counter() - load_started, 2)
    print(f"loaded in {load_seconds}s | vram={torch.cuda.memory_allocated() / 1024**3:.3f} GiB",
          flush=True)

    # Prove every variant template formats cleanly before spending generations.
    render_checks = [common.render_check(attack, args.prompt_variant)
                     for attack in common.ATTACKS]

    rows: list[dict] = []
    for index in range(args.calls):
        attack = common.ATTACKS[index % len(common.ATTACKS)]
        within = index // len(common.ATTACKS)
        round_num = common.ROUNDS[within % len(common.ROUNDS)]
        goal = common.GOALS[within % len(common.GOALS)]
        history = common.build_history(attack, goal, round_num)
        seed = args.seed_base + index
        if not args.no_seed:
            torch.manual_seed(seed)

        step = common.load_step(attack)
        # The attack passes 0.7; a sampling candidate substitutes its own value
        # at this boundary (recorded per row and in the manifest).
        override = None if abs(args.temperature - 0.7) < 1e-9 else args.temperature
        proxy = common.StudyProxy(client, temperature_override=override)
        started = time.perf_counter()
        error = None
        question = summary = None
        with common.injected_prompt_variant(attack, args.prompt_variant):
            try:
                question, summary = step(
                    round_num=round_num, goal=goal,
                    history_attacker=history["history_attacker"],
                    history_target=history["history_target"],
                    scores=history["scores"],
                    last_response=history["last_response"],
                    attacker_llm=proxy, max_rounds=8,
                )
                outcome = "OK"
            except Exception as exc:  # noqa: BLE001
                outcome = type(exc).__name__
                error = str(exc)[:400]
        wall = round(time.perf_counter() - started, 3)

        call = proxy.calls[-1] if proxy.calls else {}
        raw = call.get("pre_parse_reply")
        verdict = classifier.classify_reply(raw if raw is not None else "")
        stats = call.get("generation_stats") or {}
        row = {
            "stage": args.stage,
            "candidate_id": args.config,
            "prompt_variant": args.prompt_variant,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "structured_output_mode": args.mode,
            "seed": None if args.no_seed else seed,
            "call_index": index,
            "attack": attack,
            "round": round_num,
            "goal": goal,
            "step_outcome": outcome,
            "step_error": error,
            "wall_s": wall,
            "latency_s": call.get("latency_s"),
            "prompt_tokens": stats.get("prompt_tokens"),
            "generated_tokens": stats.get("generated_tokens"),
            "termination": stats.get("termination"),
            "system_prompt_sha256": call.get("system_prompt_sha256"),
            "category": verdict["category"],
            "failure_shape": verdict["failure_shape"],
            "direct_json": verdict["direct_json"],
            "extracted_json": verdict["extracted_json"],
            "has_generatedQuestion": verdict["has_generatedQuestion"],
            "has_lastResponseSummary": verdict["has_lastResponseSummary"],
            "generatedQuestion_type": verdict["generatedQuestion_type"],
            "empty_query": verdict["empty_query"],
            "reasoning_leakage": verdict["reasoning_leakage"],
            "generated_question": question,
            "raw_output": raw,
        }
        rows.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (stage_dir / f"{args.config}_raw.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{index + 1}/{args.calls}] {attack} round={round_num} "
              f"{row['category']:<26} outcome={outcome:<20} gen={row['generated_tokens']} "
              f"term={row['termination']} {row['latency_s']}s", flush=True)

    summary = common.summarise(rows, args.config)
    (stage_dir / f"{args.config}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    manifest_record = {
        "candidate_id": args.config,
        "stage": args.stage,
        "model_id": common.MODEL_ID,
        "revision": common.REVISION,
        "architecture": "Qwen3_5ForConditionalGeneration",
        "processor": "Qwen3VLProcessor",
        "quantization": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": None,
        "enable_thinking": False,
        "structured_decoding": args.mode,
        "prompt_variant": args.prompt_variant,
        "prompt_variant_definition": {
            "difference_from_control": common.PROMPT_VARIANTS[args.prompt_variant][
                "difference_from_control"],
            "reason_for_variant": common.PROMPT_VARIANTS[args.prompt_variant][
                "reason_for_variant"],
            "suffix_template": common.PROMPT_VARIANTS[args.prompt_variant]["suffix_template"],
        },
        "prompt_render_checks": render_checks,
        "seed_policy": ("none (production behaviour)" if args.no_seed else
                        f"torch.manual_seed({args.seed_base} + call_index) before each "
                        f"generation — study-level pairing, not a production change"),
        "sample_size": args.calls,
        "attack_distribution": {attack: sum(1 for r in rows if r["attack"] == attack)
                                for attack in common.ATTACKS},
        "round_distribution": {str(round_num): sum(1 for r in rows if r["round"] == round_num)
                               for round_num in sorted({r["round"] for r in rows})},
        "load_seconds": load_seconds,
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "summary": summary["overall"],
    }
    manifest_path = write_manifest(prefix, manifest_record)
    print(f"\n{args.config}: contract-valid {summary['overall']['contract_valid']}/"
          f"{summary['overall']['n']} ({summary['overall']['contract_valid_rate']}), "
          f"usable {summary['overall']['semantically_usable']} "
          f"({summary['overall']['semantically_usable_rate']}), "
          f"empty-query {summary['overall']['empty_query']}, "
          f"prose {summary['overall']['prose_failure']}")
    print(f"wrote {results_path.relative_to(REPO_ROOT)} and {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
