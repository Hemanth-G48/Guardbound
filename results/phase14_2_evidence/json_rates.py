import json
from collections import Counter
from pathlib import Path

NEW = [json.loads(l) for l in Path("results/phase14/batch00.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=== ATTACKER STRUCTURED-OUTPUT ACCOUNTING ===")
json_calls = Counter()
failures = Counter()
for r in NEW:
    pf = r.get("attacker_parse_failures") or {}
    for k, v in (pf.get("json_calls") or {}).items():
        json_calls[k] += v
    for k, v in (pf.get("by_purpose") or {}).items():
        failures[k] += v
total_json_calls = sum(json_calls.values())
total_failures = sum(failures.values())
print("attacker json_format=True calls :", total_json_calls)
print("attacker JSON parse failures    :", total_failures)
print("attacker purposes               :", dict(json_calls))
print("failures by purpose             :", dict(failures))
rate = (1 - total_failures / total_json_calls) * 100 if total_json_calls else 0.0
print(f"attacker structured-JSON success rate: {rate:.1f}%")

print()
print("=== EVALUATOR STRUCTURED-OUTPUT ACCOUNTING ===")
ev_json = Counter(); ev_fail = Counter()
for r in NEW:
    pf = r.get("evaluator_parse_failures") or {}
    for k, v in (pf.get("json_calls") or {}).items():
        ev_json[k] += v
    for k, v in (pf.get("by_purpose") or {}).items():
        ev_fail[k] += v
tot_ev = sum(ev_json.values()); tot_evf = sum(ev_fail.values())
print("evaluator json_format=True calls:", tot_ev, "failures:", tot_evf)
print(f"evaluator structured-JSON success rate: "
      f"{(1 - tot_evf / tot_ev) * 100 if tot_ev else 0.0:.1f}%")

print()
print("=== EMPTY-QUERY FAILURES (ATTACK_FAILURE, turns=0) ===")
for r in NEW:
    if r.get("failure_class") == "ATTACK_FAILURE":
        print(f"  {r['run_id']:<24} turns={r.get('num_turns')} final_response={str(r.get('final_response'))[:80]!r}")

print()
print("=== THE ONE SUCCESS ===")
s = [r for r in NEW if r.get("failure_class") == "SUCCESS"][0]
print("run_id:", s["run_id"], "| turns:", s["num_turns"], "| success:", s["success"])
print("nbf_scores:", s.get("nbf_scores"), "| filtered:", s.get("filtered_queries"))
print("llm_calls:", s.get("llm_calls"))
print("peak_vram_gb:", s.get("peak_vram_gb"))
print("residency snapshot:", (s.get("residency") or {}).get("snapshot"))
print("rotations:")
for rot in (s.get("residency") or {}).get("rotations") or []:
    print("   ", rot)

print()
print("=== REFUSAL / BACKTRACK ACTIVITY (semantics preserved) ===")
for r in NEW:
    pc = (r.get("attacker_purpose_calls") or {})
    if pc or (r.get("residency") or {}).get("loads"):
        print(f"  {r['run_id']:<24} attacker_purposes={pc} "
              f"evaluator_purposes={r.get('evaluator_purpose_calls')}")
