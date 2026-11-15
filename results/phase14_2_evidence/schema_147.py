import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14_7_18_run_production_validation.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=== top-level keys present across all 18 records ===")
keys = Counter()
for r in rows:
    for k in r:
        keys[k] += 1
for k, v in sorted(keys.items()):
    print(f"  {k:<30} present in {v}/18")

print()
print("=== Step 12 required telemetry coverage ===")
required = ["run_id", "goal_id", "attack", "nbf_condition", "seed",
            "attacker_model", "target_model", "evaluator_model",
            "success", "rubric_score", "rubric_history", "turns",
            "refusal_count", "termination_reason", "nbf_scores", "filtered_queries",
            "attacker_calls", "target_calls", "evaluator_calls",
            "json_calls", "json_parse_failures",
            "model_load_count", "model_eviction_count",
            "runtime_seconds", "peak_vram",
            "config_hash", "checkpoint_hash", "dataset_hash"]
alias = {"nbf_condition": "condition", "json_calls": "attacker_parse_failures.json_calls",
         "json_parse_failures": "attacker_parse_failures.total",
         "model_load_count": "residency.loads", "model_eviction_count": "residency.evictions",
         "runtime_seconds": "runtime", "peak_vram": "peak_vram_gb",
         "attacker_calls": "llm_calls.attacker", "target_calls": "llm_calls.target",
         "evaluator_calls": "llm_calls.evaluator",
         "config_hash": "config_hash", "checkpoint_hash": "checkpoint_sha256",
         "dataset_hash": "dataset_sha256"}
for k in required:
    present = sum(1 for r in rows if k in r)
    a = alias.get(k, k)
    root = a.split(".")[0]
    ap = sum(1 for r in rows if root in r)
    flag = "OK" if (present or ap) else "** MISSING **"
    print(f"  {k:<24} direct={present:<3} via '{a}'={ap:<3} {flag}")

print()
print("=== sample record (turn structure) ===")
r = next(r for r in rows if (r.get("turns") or []))
print("  run_id:", r["run_id"], " num_turns:", r.get("num_turns"))
print("  turns[0] keys:", sorted((r["turns"][0] or {}).keys()))
print("  turns[0]:", json.dumps({k: (str(v)[:90]) for k, v in r["turns"][0].items()}, indent=4))
print()
print("  judge_score values across all runs:",
      dict(Counter(t.get("judge_score") for r2 in rows for t in (r2.get("turns") or []))))
print("  was_filtered values:",
      dict(Counter(t.get("was_filtered") for r2 in rows for t in (r2.get("turns") or []))))

print()
print("=== rubric scores actually produced in-run ===")
print("  (per-turn judge_score, plus evaluator rubric purpose counts)")
for r2 in rows:
    js = [t.get("judge_score") for t in (r2.get("turns") or [])]
    print(f"  {r2['run_id']:<24} judge_scores={js}  "
          f"rubric_purpose_calls={(r2.get('evaluator_purpose_calls') or {}).get('rubric')}")
