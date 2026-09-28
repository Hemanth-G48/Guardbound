#!/usr/bin/env python3
"""Phase 14 pre-flight probe: attacker JSON-compliance survival rate.

MEASUREMENT ONLY — no experimental code is modified. This script reuses the
production Phase 14 code path (`run_one_phase14`, `make_models`,
`make_barrier_and_embed`, `build_metadata`, the same seed derivation and the
same result schema) and writes to its own output file so the batch 0 study data
(`results/phase14/batchNN.jsonl`) stays pristine.

It answers one question: how far into an 8-round Crescendo / OppositeDay /
Acronym run does the local attacker keep emitting the required JSON object
before it answers in prose (which, under the author-faithful single-call rule,
ends the run as JSON_PARSE_ERROR)?

Usage:
    python scripts/_phase14_json_probe.py --goals 10 --condition off
    python scripts/_phase14_json_probe.py --goals 3 --attacks crescendo_paper --condition on
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# Import the production Phase 14 module by path (it must not be imported as a
# package: it lives in scripts/ and does work at module import time only in
# main()).
_spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", _ROOT / "scripts" / "phase14_full_reproduction.py"
)
P14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P14)

import torch  # noqa: E402

logger = logging.getLogger("phase14_probe")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(P14.CONFIG_PATH))
    p.add_argument("--goals", type=int, default=10, help="number of goals from the top of the file")
    p.add_argument("--goal-offset", type=int, default=0)
    p.add_argument("--attack", choices=list(P14.ATTACK_KEYS) + ["all"], default="all")
    p.add_argument("--condition", choices=["on", "off"], default="off")
    p.add_argument("--out", default="results/phase14/probe_json_compliance.jsonl")
    return p.parse_args(argv)


async def run_probe(args) -> int:
    cfg = P14.load_config(args.config)
    with open(P14.OFFICIAL_DATASET, "r", encoding="utf-8") as f:
        all_goals = json.load(f)
    goals = all_goals[args.goal_offset : args.goal_offset + args.goals]

    dataset_sha = P14.sha256_of(P14.OFFICIAL_DATASET)
    ckpt_sha = P14.sha256_of(Path(cfg["nbf"]["checkpoint"]))
    assert dataset_sha == P14.OFFICIAL_DATASET_SHA256, "dataset SHA mismatch"
    metadata = P14.build_metadata(cfg, ckpt_sha, dataset_sha)
    eta = cfg["nbf"].get("threshold", 0.0)
    max_turns = cfg["attacks"].get("max_turns", 8)
    attacks = P14.ATTACK_ORDER if args.attack == "all" else [args.attack]

    out_path = Path(args.out)
    rows: list[dict] = []
    done_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(prev, dict) and prev.get("run_id"):
                    rows.append(prev)
                    done_ids.add(prev["run_id"])

    print(f"[probe] MEASUREMENT ONLY — no experiment code modified, no batch data touched", flush=True)
    if done_ids:
        print(f"[probe] resuming: {len(done_ids)} completed rows loaded from {out_path} "
              f"(skipped, included in the summary)", flush=True)
    print(f"[probe] goals={len(goals)} attacks={attacks} condition={args.condition} "
          f"max_turns={max_turns} dataset_sha={dataset_sha[:12]}", flush=True)
    print(f"[probe] output={out_path}", flush=True)

    attacker_llm, target_llm, evaluator_llm, _ = P14.make_models(cfg)
    barrier, embed_fn = P14.make_barrier_and_embed(cfg) if args.condition == "on" else (None, None)

    from guardbound.attacks.registry import get_attack

    t_start = time.time()
    for gi, goal_record in enumerate(goals):
        goal_id = args.goal_offset + gi
        for attack_key in attacks:
            attack_short = P14.ATTACK_KEYS[attack_key]
            seed = P14.derive_seed(goal_id, attack_short, args.condition)
            run_id = P14.make_run_id(args.condition, attack_short, goal_id)
            if run_id in done_ids:
                print(f"[probe] {run_id} SKIP (already recorded)", flush=True)
                continue

            attack = get_attack(attack_key)
            if hasattr(attack, "_max_refusal_retries"):
                attack._max_refusal_retries = P14.OFFICIAL_MAX_REFUSAL_RETRIES
            if hasattr(attack, "set_attacker_llm"):
                attack.set_attacker_llm(attacker_llm)
            if hasattr(attack, "set_evaluator_llm"):
                attack.set_evaluator_llm(evaluator_llm)
            for _llm in (attacker_llm, target_llm, evaluator_llm):
                _llm.reset_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start = time.time()
            try:
                # Per-run seeding belongs inside the guarded block: a CUDA-level
                # failure here is an infrastructure fault, not a run result.
                P14._seed_all(seed)
                row = await P14.run_one_phase14(
                    attack=attack,
                    goal_record=goal_record,
                    goal_id=goal_id,
                    attack_key=attack_key,
                    condition=args.condition,
                    target_llm=target_llm,
                    barrier=barrier,
                    embed_fn=embed_fn,
                    cfg=cfg,
                    eta=eta,
                    max_turns=max_turns,
                    seed=seed,
                    metadata=metadata,
                )
                row["failure_class"] = "SUCCESS" if row.get("success") else "ATTACK_FAILURE"
                row["error_type"] = None
                row["error_message"] = None
            except Exception as e:  # noqa: BLE001 — probe records, never hides
                logger.exception("probe run failed: %s", run_id)
                failure_class, error_type, error_message = P14.classify_exception(e)
                if failure_class == "INFRASTRUCTURE_ERROR":
                    # GPU-level faults poison the CUDA context for every
                    # subsequent run and would write meaningless rows. Stop the
                    # probe; the operator re-runs it (resume skips finished ids).
                    print(f"[probe] ABORTED: infrastructure fault in {run_id}: "
                          f"{error_type}: {error_message}", flush=True)
                    return 1
                row = {
                    "run_id": run_id, "goal_id": goal_id, "condition": args.condition,
                    "attack": attack_key, "seed": seed, "success": False,
                    "failure_class": failure_class, "error_type": error_type,
                    "error_message": error_message, "error": str(e),
                }
            row["runtime"] = round(time.time() - start, 3)
            row["runtime_seconds"] = row["runtime"]
            row["probe"] = True
            row["probe_purpose"] = "measure attacker JSON-compliance survival (measurement only)"
            if torch.cuda.is_available():
                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)

            gen_calls = (getattr(attacker_llm, "purpose_counts", {}) or {}).get("generation", 0)
            eval_calls = (getattr(evaluator_llm, "purpose_counts", {}) or {})
            row["probe_generations"] = gen_calls
            row["probe_target_calls"] = getattr(target_llm, "calls", None)
            row["probe_evaluator_calls"] = dict(eval_calls)
            row["llm_calls"] = {
                "attacker": getattr(attacker_llm, "calls", None),
                "target": getattr(target_llm, "calls", None),
                "evaluator": getattr(evaluator_llm, "calls", None),
            }
            if hasattr(attacker_llm, "parse_failure_summary"):
                row["attacker_parse_failures"] = attacker_llm.parse_failure_summary()
            if hasattr(evaluator_llm, "parse_failure_summary"):
                row["evaluator_parse_failures"] = evaluator_llm.parse_failure_summary()
            turns = row.get("num_turns", 0) or 0
            row["probe_turns"] = turns
            rows.append(row)
            P14.append_result(out_path, row)
            print(
                f"[probe] {run_id} class={row['failure_class']} turns={turns} "
                f"generations={gen_calls} runtime={row['runtime']:.0f}s",
                flush=True,
            )

    total = len(rows)
    completed = [r for r in rows if r["failure_class"] in ("SUCCESS", "ATTACK_FAILURE")]
    failed = [r for r in rows if r["failure_class"] not in ("SUCCESS", "ATTACK_FAILURE")]

    print("\n" + "=" * 78)
    print(f"[probe] SUMMARY  ({time.time() - t_start:.0f}s wall, {total} runs, "
          f"condition={args.condition}, max_turns={max_turns})")
    print(f"[probe]   completed runs (reached the round budget): {len(completed)}/{total}")
    print(f"[probe]   failed runs                             : {len(failed)}/{total}")
    by_class: dict[str, int] = {}
    for r in rows:
        by_class[r["failure_class"]] = by_class.get(r["failure_class"], 0) + 1
    for k, v in sorted(by_class.items()):
        print(f"[probe]     {k:22s} {v}")

    # Generation-level survival: how many runs survived the first r generations.
    print("[probe] generation-level survival (fraction of runs that produced >= r "
          "valid JSON generations):")
    for r_idx in range(1, max_turns + 3):
        alive = sum(
            1 for r in rows
            if (r.get("probe_generations") or 0) - (0 if r in completed else 1) >= r_idx
        )
        if alive == 0:
            break
        print(f"[probe]     round {r_idx}: {alive}/{total} = {alive / total:.0%}")

    for atk in attacks:
        sub = [r for r in rows if r["attack"] == atk]
        comp = [r for r in sub if r["failure_class"] in ("SUCCESS", "ATTACK_FAILURE")]
        turns = [r.get("probe_turns", 0) for r in sub]
        print(f"[probe]   {atk:16s} runs={len(sub)} completed={len(comp)} "
              f"mean_turns={sum(turns) / max(len(turns), 1):.2f}")
    print("=" * 78)
    return 0


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")
    args = parse_args(argv)
    sys.exit(asyncio.run(run_probe(args)))


if __name__ == "__main__":
    main()
