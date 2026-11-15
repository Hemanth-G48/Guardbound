"""Phase 14.8 audit — authoritative field-set diff between the normal and error paths."""
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14_7_18_run_production_validation.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

err = [r for r in rows if r.get("failure_class") == "JSON_PARSE_ERROR"]
ok = [r for r in rows if r.get("failure_class") != "JSON_PARSE_ERROR"]
print(f"records: {len(rows)}  normal: {len(ok)}  error: {len(err)}")

# Key set of a NORMAL record = union over normal runs (some fields are conditional)
normal_keys = set()
for r in ok:
    normal_keys |= set(r.keys())
error_keys = set()
for r in err:
    error_keys |= set(r.keys())

print(f"\nnormal-path key count (union) : {len(normal_keys)}")
print(f"error-path  key count (union) : {len(error_keys)}")

only_normal = sorted(normal_keys - error_keys)
only_error = sorted(error_keys - normal_keys)
print(f"\nPRESENT ONLY ON NORMAL PATH ({len(only_normal)}):")
for k in only_normal:
    print(f"   {k}")

print(f"\nPRESENT ONLY ON ERROR PATH ({len(only_error)}):")
for k in only_error:
    print(f"   {k}")

print("\n=== fields that are CONDITIONAL within the normal path ===")
for k in sorted(normal_keys):
    n = sum(1 for r in ok if k in r)
    if 0 < n < len(ok):
        print(f"   {k:<28} present in {n}/{len(ok)} normal runs  <- conditional")

print("\n=== fields that are CONDITIONAL within the error path ===")
if err:
    for k in sorted(error_keys):
        n = sum(1 for r in err if k in r)
        if 0 < n < len(err):
            print(f"   {k:<28} present in {n}/{len(err)} error runs  <- conditional")
else:
    print("   (only one error record; no within-path variation detectable)")

print("\n=== union of all keys across all 18 (the target schema surface) ===")
allk = sorted(normal_keys | error_keys)
print(f"   {len(allk)} keys")

print("\n=== PER-FIELD availability matrix ===")
print(f"{'field':<30} {'normal':>8} {'error':>8}")
print("-" * 48)
for k in allk:
    n = sum(1 for r in ok if k in r)
    e = sum(1 for r in err if k in r)
    print(f"{k:<30} {n:>4}/{len(ok):<3} {e:>4}/{len(err):<3}")
