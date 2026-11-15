import json
from pathlib import Path

p = Path("results/phase14/phase14_3_evaluator_qualification.jsonl")
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r["case_id"] == "refusal0" and "llm_check_refusal" in r["contract"]:
        print("contract :", r["contract"])
        print("case_id  :", r["case_id"])
        print("raw_type :", r["raw_type"])
        print("valid_json:", r["valid_json"], "fields:", r["required_fields_present"])
        print("parser_return:", r.get("parser_return"))
        print("parser_crash :", r.get("parser_crash"))
        print("raw      :", repr(r["raw"])[:800])
        print()
        break

print("=== raw types per contract (all rows) ===")
from collections import Counter
c = Counter()
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    c[(r["contract"].split(".")[-1], r["raw_type"], r["valid_json"])] += 1
for k, v in sorted(c.items()):
    print(f"  {k[0]:<28} raw_type={k[1]:<6} valid_json={k[2]}  n={v}")

print()
print("=== sample raw for a refusal-style case (check_disclaimer) ===")
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r["case_id"] == "refusal0" and "check_disclaimer" in r["contract"]:
        print(repr(r["raw"])[:600])
        break
