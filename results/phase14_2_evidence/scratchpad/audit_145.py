"""Phase 14.5 — semantic audit of the 77 raw outputs.

Step 12 requires checking whether a structurally-valid result could still be
degenerate. A 100% structural pass is only meaningful if the values vary with
the input and are semantically sensible.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

P = Path("results/phase14_5_llama32_3b_evaluator_qualification.jsonl")
rows = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
print("records:", len(rows), "| roles:", dict(Counter(r["role"] for r in rows)))
ev = [r for r in rows if r["role"] == "evaluator"]

print("\n=== structural totals ===")
print("  valid_json          :", sum(r["valid_json"] for r in ev), "/", len(ev))
print("  required_fields     :", sum(r["required_fields_present"] for r in ev), "/", len(ev))
print("  parser_accepted     :", sum(r["parser_accepted"] for r in ev), "/", len(ev))
print("  parser_crashes      :", sum(1 for r in ev if r.get("parser_crash")))
print("  failure_types       :", dict(Counter(r["failure_type"] or "OK" for r in ev)))

print("\n=== Step 9 checklist: absence of malformed-output patterns ===")
def count(pred, label):
    n = sum(1 for r in ev if pred(r["raw"] or "", r))
    print(f"  {label:<38} {n}/{len(ev)}")

count(lambda s, r: "```" in s, "markdown fence present")
count(lambda s, r: "<think>" in s.lower() or "</think>" in s.lower(), "<think> leakage")
count(lambda s, r: "\n" in s and '"' in s, "raw newline anywhere")
count(lambda s, r: '"\n' in s or '\n"' in s, "raw newline immediately inside a quote")
count(lambda s, r: s.strip().startswith("{") and s.strip().endswith("}"), "starts '{' and ends '}'")
count(lambda s, r: s.count("{") > 1, "more than one '{' (possible multi-object)")
count(lambda s, r: s.strip() != s, "leading/trailing whitespace")
count(lambda s, r: chr(92) + "n" in s, "contains an escaped backslash-n")

print("\n=== value distributions (degeneracy check) ===")
def obj(r):
    try:
        return json.loads(r["raw"])
    except Exception:
        return None

ref = [r for r in ev if "check_refusal" in r["contract"]]
dis = [r for r in ev if "disclaimer" in r["contract"]]
rub = [r for r in ev if "rubric" in r["contract"]]

print("  refusal-family (33 calls) by case kind:")
for kind in ("target_response", "refusal_style"):
    s = [r for r in ref if r.get("case_kind") == kind]
    vals = Counter(str((obj(r) or {}).get("value")) for r in s)
    meta = Counter(str((obj(r) or {}).get("metadata")) for r in s)
    print(f"    {kind:<16} n={len(s):<3} value={dict(vals)}")
    print(f"                     metadata={dict(meta)}")

print("  disclaimer (11 calls):")
print("    value =", dict(Counter(str((obj(r) or {}).get("value")) for r in dis)))
print("    keys  =", dict(Counter(tuple(sorted((obj(r) or {}).keys())) for r in dis)))

print("  rubric (33 calls):")
print("    score =", dict(Counter(str((obj(r) or {}).get("score")) for r in rub)))
print("    keys  =", dict(Counter(tuple(sorted((obj(r) or {}).keys())) for r in rub)))

print("\n=== parser_return vs raw (does the sentinel hide a failure?) ===")
print("  refusal-family parser_return:", dict(Counter(str(r.get("parser_return")) for r in ref)))
print("  disclaimer parser_return    :", dict(Counter(str(r.get("parser_return")) for r in dis)))
print("  rubric parser_return        :", dict(Counter(str(r.get("parser_return")) for r in rub)))
# The refusal parser returns value AND metadata>90. A structurally valid object
# with metadata<=90 returns False legitimately -> distinguish from a sentinel.
confident = [r for r in ref if int((obj(r) or {}).get("metadata", -1)) > 90]
print(f"  refusal objects with metadata>90 : {len(confident)}/{len(ref)}"
      "  (only these produce a non-sentinel verdict)")

print("\n=== byte-identical output check (degeneracy) ===")
for name, s in (("refusal", ref), ("disclaimer", dis), ("rubric", rub)):
    distinct = len(set(r["raw"] for r in s))
    print(f"  {name:<12} distinct raw outputs: {distinct}/{len(s)}")

print("\n=== rubric score vs case (does it vary by input?) ===")
byr = defaultdict(list)
for r in rub:
    byr[r["case_id"]].append(str((obj(r) or {}).get("score")))
for k in sorted(byr):
    print(f"    {k:<10} scores={byr[k]}")

print("\n=== 8 representative raw outputs (verbatim) ===")
for c in ("crescendo_paper.llm_check_refusal", "acronym.check_disclaimer",
          "crescendo_paper.llm_evaluate_with_rubric"):
    for r in ev:
        if r["contract"] == c:
            print(f"  {c.split('.')[-1]:<30} case={r['case_id']:<9} {json.dumps(r['raw'])[:180]}")
            break

print("\n=== latency ===")
lat = sorted(r["elapsed_s"] for r in ev)
n = len(lat)
print(f"  n={n} mean={sum(lat)/n:.2f}s median={lat[n//2]:.2f}s "
      f"p95={lat[min(n-1, int(n*0.95))]:.2f}s max={max(lat):.2f}s min={min(lat):.2f}s")

print("\n=== provenance fields present per record (Step 7) ===")
required = ["case_id", "goal", "contract", "model_id", "model_revision", "temperature",
            "top_p", "max_new_tokens", "structured_output_mode", "raw", "parser_accepted",
            "failure_type", "elapsed_s"]
missing = {k: sum(1 for r in ev if k not in r) for k in required}
print("  fields missing across 77 records:", {k: v for k, v in missing.items() if v})
print("  model_id      :", dict(Counter(r["model_id"] for r in ev)))
print("  model_revision:", dict(Counter(r["model_revision"] for r in ev)))
print("  temperature   :", dict(Counter(str(r.get("temperature")) for r in ev)))
print("  seed          :", dict(Counter(str(r.get("seed")) for r in ev)))
print("  max_new_tokens:", dict(Counter(str(r.get("max_new_tokens")) for r in ev)))
print("  mode          :", dict(Counter(str(r.get("structured_output_mode")) for r in ev)))
print("  dtype         :", dict(Counter(str(r.get("dtype")) for r in ev)))
