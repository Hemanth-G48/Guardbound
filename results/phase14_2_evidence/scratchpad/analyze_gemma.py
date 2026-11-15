import json
from collections import Counter, defaultdict
from pathlib import Path

P = Path("results/phase14_4_gemma_evaluator_qualification.jsonl")
rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
print("records:", len(rows))
print("by role:", dict(Counter(r["role"] for r in rows)))

ev = [r for r in rows if r["role"] == "evaluator"]
print("\n=== EVALUATOR: outcome by contract ===")
for c in sorted({r["contract"] for r in ev}):
    s = [r for r in ev if r["contract"] == c]
    print(f"  {c.split('.')[-1]:<34} n={len(s):<3} valid_json={sum(r['valid_json'] for r in s):<3} "
          f"accepted={sum(r['parser_accepted'] for r in s):<3} "
          f"crashes={sum(1 for r in s if r.get('parser_crash'))}")

print("\n=== raw_type distribution for the failing refusal-family calls ===")
fam = [r for r in ev if "check_refusal" in r["contract"] or "disclaimer" in r["contract"]]
print("  n =", len(fam))
print("  raw_type:", dict(Counter(r["raw_type"] for r in fam)))
print("  valid_json:", dict(Counter(r["valid_json"] for r in fam)))

print("\n=== does the raw output contain a JSON object anywhere? (diagnosis only, NOT acceptance) ===")
def has_json_obj(s):
    if not isinstance(s, str):
        return False
    i = s.find("{")
    return i != -1 and "}" in s[i:]
def fenced(s):
    return isinstance(s, str) and "```" in s
with_obj = sum(1 for r in fam if has_json_obj(r["raw"]))
fenced_n = sum(1 for r in fam if fenced(r["raw"]))
print(f"  contains a '{{...}}' somewhere : {with_obj}/{len(fam)}")
print(f"  contains a ``` fence            : {fenced_n}/{len(fam)}")
print("  NOTE: these are diagnostics only. The production contract rejects both,")
print("        and this gate does NOT count them as valid.")

print("\n=== raw output lengths ===")
lens = [len(r["raw"] or "") for r in fam]
print(f"  min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.0f}")

print("\n=== lengt/h profile: how many end with an unterminated string vs prose ===")
ends_brace = sum(1 for r in fam if (r["raw"] or "").rstrip().endswith("}"))
ends_fence = sum(1 for r in fam if (r["raw"] or "").rstrip().endswith("```"))
print(f"  ends with '}}'      : {ends_brace}")
print(f"  ends with ```       : {ends_fence}")

print("\n=== 6 representative raw refusal-family outputs (unmodified, truncated for display) ===")
shown = 0
for c in ("crescendo_paper.llm_check_refusal", "acronym.check_disclaimer"):
    for r in ev:
        if r["contract"] == c and not r["parser_accepted"] and shown < 6:
            print(f"\n--- contract={c.split('.')[-1]} case={r['case_id']} "
                  f"raw_type={r['raw_type']} len={len(r['raw'] or '')} "
                  f"crash={bool(r.get('parser_crash'))}")
            print(repr((r["raw"] or "")[:420]))
            shown += 1

print("\n=== what did the PASSING rubric calls look like? (contrast) ===")
for r in ev:
    if "rubric" in r["contract"] and r["parser_accepted"]:
        print(f"  {r['contract'].split('.')[-1]:<34} {r['case_id']:<9} "
              f"raw={repr((r['raw'] or '')[:90])} parser_return={r.get('parser_return')}")
        break
for r in ev:
    if "rubric" in r["contract"]:
        print(f"  rubric sample case={r['case_id']} raw={repr((r['raw'] or '')[:120])}")
        break
print("  rubric parser_return distribution:",
      dict(Counter(r.get("parser_return") for r in ev if "rubric" in r["contract"])))

print("\n=== 11 parser crashes ===")
for r in ev:
    if r.get("parser_crash"):
        print(f"  {r['contract']:<40} {r['case_id']:<9} {r['parser_crash'][:70]}")

print("\n=== per-case acceptance (refusal family) ===")
bycase = defaultdict(lambda: [0, 0])
for r in fam:
    bycase[r["case_id"]][1] += 1
    if r["parser_accepted"]:
        bycase[r["case_id"]][0] += 1
for k in sorted(bycase):
    a, n = bycase[k]
    print(f"  {k:<10} {a}/{n}")
