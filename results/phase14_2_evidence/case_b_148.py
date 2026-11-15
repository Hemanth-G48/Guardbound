"""Analyse the LIVE production JSON_PARSE_ERROR record produced by the new path."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts/phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)
CANON = set(P14.RUN_RECORD_FIELDS)

r = json.loads((ROOT / "results/phase14/batch01.jsonl")
               .read_text(encoding="utf-8").splitlines()[0])

print("=" * 92)
print("LIVE PRODUCTION JSON_PARSE_ERROR RECORD (real attacker, unmocked)")
print("=" * 92)
print(f"  run_id          : {r['run_id']}")
print(f"  failure_class   : {r['failure_class']}")
print(f"  success         : {r['success']}")
print(f"  error_type      : {r['error_type']}")
print(f"  error_message   : {str(r['error_message'])[:150]}")
print(f"  runtime         : {r['runtime']} s")
print(f"  peak_vram_gb    : {r['peak_vram_gb']}")

print("\n=== STRUCTURAL COMPLETENESS ===")
missing = sorted(CANON - set(r.keys()))
extra = sorted(set(r.keys()) - CANON)
print(f"  keys present    : {len(r)} / {len(CANON)}")
print(f"  missing         : {missing or 'NONE - complete'}")
print(f"  extra           : {extra or 'none'}")

print("\n=== FAILURE VISIBILITY (must be unmistakable) ===")
for k in ("success", "failure_class", "error_type", "error_message", "error"):
    v = r.get(k)
    print(f"  {k:<16} {'<-- MISSING' if k not in r else repr(v)[:110]}")
print(f"  is a legitimate attack outcome? "
      f"{r['failure_class'] in ('SUCCESS', 'ATTACK_FAILURE')}  (must be False)")

print("\n=== NULL SEMANTICS (no fabricated measurements) ===")
for k in ("turns", "num_turns", "accepted_query_count", "final_response",
          "termination_reason"):
    v = r.get(k)
    print(f"  {k:<22} = {v!r:<8} {'OK (None, not 0/[])' if v is None else 'FABRICATED!'}")

print("\n=== RECOVERED TELEMETRY (was omitted before Phase 14.8) ===")
recovered = ["rubric_scores", "final_score", "refusal_count", "refused_records",
             "nbf_scores", "filtered_queries", "filter_count", "nbf_stats",
             "attacker_parse_failures", "target_purpose_calls", "goal_text",
             "target_system", "dtype", "quantization", "top_p",
             "generation_config", "filter_trials"]
for k in recovered:
    present = k in r
    print(f"  {'OK ' if present else 'FAIL'} {k:<26} = {repr(r.get(k))[:80]}")

print("\n=== NBF: presence reflects actual participation (OFF run) ===")
print(f"  nbf_enabled      : {r['nbf_enabled']}  (condition={r['condition']})")
print(f"  nbf_scores       : {r['nbf_scores']}   genuinely empty")
print(f"  filtered_queries : {r['filtered_queries']}   genuinely zero")

print("\n=== RAW ERROR EVIDENCE PRESERVED ===")
print(f"  attacker_raw_outputs entries: {len(r['attacker_raw_outputs'])}")
for entry in r["attacker_raw_outputs"][:3]:
    print(f"    purpose={entry.get('purpose')} type={entry.get('type')} "
          f"raw={str(entry.get('raw'))[:120]!r}")

print("\n=== EVALUATOR / CALL TELEMETRY ===")
print(f"  llm_calls        : {r['llm_calls']}")
print(f"  evaluator_purpose_calls: {r['evaluator_purpose_calls']}")
print(f"  evaluator_parse_failures: {r['evaluator_parse_failures']}")
print(f"  attacker_purpose_calls: {r['attacker_purpose_calls']}")

print("\n=== CONFIG / IDENTITY (all present) ===")
for k in ("attacker_model", "target_model", "evaluator_model", "embedding_model",
          "checkpoint_sha256", "dataset_sha256", "config_hash", "seed",
          "max_rounds", "threshold", "dtype", "quantization", "top_p"):
    print(f"  {k:<20} {repr(r.get(k))[:70]}")

print("\n=== record size ===")
blob = json.dumps(r, ensure_ascii=False)
print(f"  json bytes: {len(blob)}")
print(f"  attacker_raw_outputs bytes: "
      f"{len(json.dumps(r['attacker_raw_outputs'], ensure_ascii=False))}")
