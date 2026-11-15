"""Case A validation — one production-shaped normal run."""
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

p = ROOT / "results/phase14/batch01.jsonl"
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"records: {len(rows)}")
r = rows[-1]
print(f"\nrun_id          : {r['run_id']}")
print(f"failure_class   : {r['failure_class']}")
print(f"success         : {r['success']}")
print(f"num_turns       : {r['num_turns']}")
print(f"runtime         : {r['runtime']}")
print(f"peak_vram_gb    : {r['peak_vram_gb']}")
print(f"termination     : {r['termination_reason']}")

print("\n=== Phase 14.8 contract checks on this NORMAL run ===")
checks = {
    "key set == canonical (55)": set(r.keys()) == CANON,
    "error is explicitly None": r.get("error", "MISSING") is None and "error" in r,
    "error_type is None": r.get("error_type") is None,
    "error_message is None": r.get("error_message") is None,
    "attacker_raw_outputs structurally present": "attacker_raw_outputs" in r,
    "nbf_scores present": "nbf_scores" in r,
    "filtered_queries present": "filtered_queries" in r,
    "nbf_stats present": "nbf_stats" in r,
    "rubric_scores present": "rubric_scores" in r,
    "llm_calls present": "llm_calls" in r,
    "residency present": "residency" in r,
    "system present": "system" in r,
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'}  {k}")

print("\n=== values that must NOT have changed ===")
print(f"  nbf_scores        : {r['nbf_scores']}   (OFF run -> genuinely empty)")
print(f"  filtered_queries  : {r['filtered_queries']}   (OFF run -> genuinely zero)")
print(f"  nbf_enabled       : {r['nbf_enabled']}")
print(f"  rubric_scores     : {r['rubric_scores']}")
print(f"  attacker_raw_outputs entries: {len(r['attacker_raw_outputs'])}")
print(f"  turns count       : {len(r['turns'])}")

missing = sorted(CANON - set(r.keys()))
extra = sorted(set(r.keys()) - CANON)
print(f"\n  missing vs canonical: {missing or 'none'}")
print(f"  extra vs canonical  : {extra or 'none'}")

print("\n=== OFF invariant still holds (unchanged semantics) ===")
print(f"  nbf_scores == []        : {r['nbf_scores'] == []}")
print(f"  filtered_queries == 0   : {r['filtered_queries'] == 0}")

print("\n=== record size ===")
print(f"  json bytes: {len(json.dumps(r, ensure_ascii=False))}")
print(f"  attacker_raw_outputs bytes: "
      f"{len(json.dumps(r['attacker_raw_outputs'], ensure_ascii=False))}")
