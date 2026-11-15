import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14_7_18_run_production_validation.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

atk = [(r["run_id"], (r.get("llm_calls") or {}).get("attacker"),
        (r.get("attacker_purpose_calls") or {}).get("generation"))
       for r in rows]
print("=== attacker call counts per run ===")
for rid, tot, gen in atk:
    print(f"  {rid:<24} llm_calls.attacker={tot}  generation_purpose={gen}")

vals = [t for _, t, _ in atk if t]
print(f"\n  runs with a count: {len(vals)}")
print(f"  mean attacker calls per run: {sum(vals)/len(vals):.1f}")
print(f"  min/max: {min(vals)}/{max(vals)}")

# per-call failure rate from the JSON telemetry
fail = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in rows)
calls = sum(sum((r.get("attacker_parse_failures") or {}).get("json_calls", {}).values())
            for r in rows)
p = fail / calls if calls else 0
print(f"\n=== attacker JSON reliability ===")
print(f"  failures / json calls: {fail}/{calls} = {p*100:.2f}% per call")
for n in (10, 15, 18, 25):
    print(f"  P(at least one failure in a {n}-call run) = "
          f"{1 - (1 - p)**n:.1%}")

print()
print("=== rubric scores produced in-run by the Llama evaluator ===")
scores = [s for r in rows for s in (r.get("rubric_scores") or []) if s is not None]
print("  distribution:", dict(sorted(Counter(scores).items())))
print("  distinct     :", sorted(set(scores)))
print("  n            :", len(scores))

print()
print("=== evaluator JSON per-call reliability ===")
ef = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in rows)
ec = sum(sum((r.get("evaluator_parse_failures") or {}).get("json_calls", {}).values())
         for r in rows)
print(f"  failures / json calls: {ef}/{ec} = {ef/ec*100 if ec else 0:.2f}%")
print(f"  evaluator calls (all purposes): "
      f"{sum(sum((r.get('evaluator_purpose_calls') or {}).values()) for r in rows)}")

print()
print("=== NBF ON: filtered + accepted vs max_rounds ===")
for r in rows:
    if r.get("condition") != "on":
        continue
    fq = r.get("filtered_queries")
    aq = r.get("accepted_query_count")
    mr = r.get("max_rounds")
    if fq is None or aq is None:
        continue
    print(f"  {r['run_id']:<24} filtered={fq} accepted={aq} sum={fq+aq} "
          f"max_rounds={mr} {'EXACT' if fq+aq == mr else 'early-stop: ' + str(r.get('termination_reason'))}")
