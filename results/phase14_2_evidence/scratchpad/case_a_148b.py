"""Case A — validate the NORMAL completion record against the Phase 14.8 contract."""
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

rows = [json.loads(l) for l in (ROOT / "results/phase14/batch02.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
r = rows[-1]
print(f"run_id        : {r['run_id']}")
print(f"failure_class : {r['failure_class']}")
print(f"success       : {r['success']}")
print(f"num_turns     : {r['num_turns']}")
print(f"runtime       : {r['runtime']} s")
print(f"termination   : {r['termination_reason']}")

print("\n=== PHASE 14.8 CONTRACT ON THE NORMAL PATH ===")
checks = {
    "structured completeness": set(r.keys()) == CANON,
    "error explicitly None": r["error"] is None,
    "error_type None": r["error_type"] is None,
    "error_message None": r["error_message"] is None,
    "attacker_raw_outputs structurally present": "attacker_raw_outputs" in r,
    "num_turns is an int (not None)": isinstance(r["num_turns"], int),
    "turns is a list": isinstance(r["turns"], list),
    "termination_reason is a normal enum value": r["termination_reason"] in (
        "success_score_5", "refusal_retry_limit", "max_turns_reached",
        "attack_loop_exhausted"),
    "NBF OFF invariant intact": r["nbf_scores"] == [] and r["filtered_queries"] == 0,
}
for k, v in checks.items():
    print(f"  {'OK ' if v else 'FAIL'}  {k}")

print("\n=== values unchanged vs the pre-fix semantics ===")
print(f"  nbf_enabled       : {r['nbf_enabled']}")
print(f"  nbf_scores        : {r['nbf_scores']}")
print(f"  filtered_queries  : {r['filtered_queries']}")
print(f"  rubric_scores     : {r['rubric_scores']}")
print(f"  final_score       : {r['final_score']}")
print(f"  refusal_count     : {r['refusal_count']}")
print(f"  accepted_query_count: {r['accepted_query_count']}")
print(f"  filter_trials     : {r['filter_trials']}")
print(f"  attacker_raw_outputs entries: {len(r['attacker_raw_outputs'])}")

missing = sorted(CANON - set(r.keys()))
extra = sorted(set(r.keys()) - CANON)
print(f"\n  missing vs canonical: {missing or 'NONE'}")
print(f"  extra vs canonical  : {extra or 'none'}")

print("\n=== KEY-SET EQUALITY: normal vs live error ===")
err = json.loads((ROOT / "results/phase14_8_case_b_live_json_error.jsonl")
                 .read_text(encoding="utf-8").splitlines()[0])
print(f"  normal keys : {len(r)}")
print(f"  error keys  : {len(err)}")
print(f"  IDENTICAL   : {set(r.keys()) == set(err.keys())}")
if set(r.keys()) != set(err.keys()):
    print(f"    only-normal: {sorted(set(r) - set(err))}")
    print(f"    only-error : {sorted(set(err) - set(r))}")

print("\n=== FINAL GATE QUESTION ===")
print("  Could an analyst mistake the error record for a legitimate outcome?")
signals = {
    "success is False": err["success"] is False,
    "failure_class == JSON_PARSE_ERROR": err["failure_class"] == "JSON_PARSE_ERROR",
    "failure_class is NOT a legit outcome": err["failure_class"] not in (
        "SUCCESS", "ATTACK_FAILURE", "REFUSAL_LIMIT", "MAX_ROUNDS",
        "ATTACK_LOOP_EXHAUSTED"),
    "error_type populated": bool(err["error_type"]),
    "error_message populated": bool(err["error_message"]),
    "error populated": bool(err["error"]),
    "num_turns is None (not a plausible 0)": err["num_turns"] is None,
    "termination_reason is None (no fake enum)": err["termination_reason"] is None,
}
for k, v in signals.items():
    print(f"  {'OK ' if v else 'FAIL'}  {k}")
print(f"\n  ANSWER: "
      f"{'NO — unmistakably a failure' if all(signals.values()) else 'YES — AMBIGUOUS!'}")
