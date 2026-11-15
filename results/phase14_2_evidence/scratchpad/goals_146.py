import json
from pathlib import Path

p = Path("nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json")
recs = json.loads(p.read_text(encoding="utf-8"))
print("total goals:", len(recs))
print("keys       :", sorted(recs[0].keys()))
print("max_rounds range:", min(r.get("max_rounds", 0) for r in recs),
      "-", max(r.get("max_rounds", 0) for r in recs))
print()
for i, r in enumerate(recs[:20]):
    print(f"{i:3d} len={len(r.get('task','')):4d}  {r.get('task','')[:220]!r}")
