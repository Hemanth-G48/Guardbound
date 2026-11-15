"""Exact evidence: why json.loads rejects Gemma's refusal-family output.

A literal newline inside a JSON string is illegal (RFC 8259 requires control
characters to be escaped). This prints the precise decoder error and proves the
offending character is a real newline, not the two-character escape \n.
"""
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in
        Path("results/phase14_4_gemma_evaluator_qualification.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
fam = [r for r in rows if "check_refusal" in r["contract"] or "disclaimer" in r["contract"]]

print("=== json.loads errors for the 44 failing refusal-family calls ===")
errs = Counter()
exact = {}
for r in fam:
    raw = r["raw"]
    try:
        json.loads(raw)
        errs["UNEXPECTEDLY PARSED"] += 1
    except Exception as e:
        errs[type(e).__name__] += 1
        exact.setdefault(str(e), raw)
for k, v in errs.most_common():
    print(f"  {k:<22} {v}")
print()
for msg, raw in list(exact.items())[:4]:
    print(f"  json.JSONDecodeError: {msg}")

print("\n=== proof the offending char is a REAL newline, not an escaped \\n ===")
sample = fam[0]
raw = sample["raw"]
print(f"  contract        : {sample['contract']}")
print(f"  raw repr        : {raw!r}")
print(f"  contains '\\\\n'   : {chr(92) + 'n' in raw}   <- escaped backslash-n present?")
print(f"  contains chr(10) : {chr(10) in raw}   <- literal newline present?")
first_nl = raw.find(chr(10))
print(f"  first literal newline at index {first_nl}, context:")
print(f"    ...{raw[max(0,first_nl-14):first_nl]!r} + <NEWLINE> + {raw[first_nl+1:first_nl+16]!r}...")

# Where do the newlines fall — inside strings or only as JSON formatting?
print("\n=== newline placement across all 44 ===")
inside = 0
for r in fam:
    s = r["raw"]
    # A newline directly after an opening quote, or directly before a closing
    # quote, is inside a string literal -> illegal.
    if '"\n' in s or '\n"' in s:
        inside += 1
print(f"  outputs with a newline immediately inside a quote ('\"\\n' or '\\n\"') : {inside}/{len(fam)}")

print("\n=== full raw output, verbatim, for the report (one call) ===")
print(json.dumps(sample["raw"]))
print("\n=== the one contract that PASSED, for contrast ===")
for r in rows:
    if "rubric" in r["contract"] and r["parser_accepted"]:
        print(f"  {r['contract']}")
        print(f"  raw = {json.dumps(r['raw'])}")
        print(f"  parser_return = {r.get('parser_return')}")
        break
print("\n=== rubric: all 33 raw outputs ===")
print("  distinct raw values:",
      dict(Counter(r["raw"] for r in rows if "rubric" in r["contract"])))
