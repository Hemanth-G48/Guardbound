import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14_7_18_run_production_validation.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=== in-run rubric scores (the evaluator's actual verdicts) ===")
allscores = []
for r in rows:
    rs = r.get("rubric_scores")
    fs = r.get("final_score")
    tr = r.get("termination_reason")
    fc = r.get("filter_count")
    rc = r.get("refusal_count")
    if rs is not None:
        allscores.extend([s for s in rs if s is not None])
    print(f"  {r['run_id']:<24} rubric_scores={rs} final_score={fs} "
          f"term={tr} filter_count={fc} refusal_count={rc}")
print()
print("  all in-run rubric scores:", dict(Counter(allscores)))
print("  distinct in-run scores :", sorted({s for s in allscores}))
print("  n evaluated            :", len(allscores))

print()
print("=== termination reasons (Step 12) ===")
print(" ", dict(Counter(r.get("termination_reason") for r in rows)))

print()
print("=== NBF ON filter accounting (Step 7: filtered queries must not consume history) ===")
for r in rows:
    if r.get("condition") != "on":
        continue
    print(f"  {r['run_id']:<24} nbf_scores={len(r.get('nbf_scores') or [])} "
          f"filtered_queries={r.get('filtered_queries')} filter_count={r.get('filter_count')} "
          f"accepted_query_count={r.get('accepted_query_count')} "
          f"num_turns={r.get('num_turns')} max_rounds={r.get('max_rounds')}")
print()
print("  OFF runs for contrast:")
for r in rows:
    if r.get("condition") != "off":
        continue
    print(f"  {r['run_id']:<24} nbf_scores={r.get('nbf_scores')} "
          f"filtered_queries={r.get('filtered_queries')} "
          f"accepted_query_count={r.get('accepted_query_count')}")

print()
print("=== harness/identity hashes (Step 12) ===")
r = rows[0]
for k in ("config_hash", "checkpoint_sha256", "dataset_sha256", "code_commit",
          "worktree_sha256", "embedding_model", "threshold", "max_rounds",
          "dtype", "quantization", "top_p"):
    vals = {rr.get(k) for rr in rows if k in rr}
    print(f"  {k:<20} {list(vals)[:2]}{' ...' if len(vals) > 2 else ''}")

print()
print("=== error-path record completeness (the one JSON_PARSE_ERROR run) ===")
base = set(rows[0].keys())
for r in rows:
    missing = sorted(base - set(r.keys()))
    if missing:
        print(f"  {r['run_id']}: missing {len(missing)} fields")
        print("   ", missing)
