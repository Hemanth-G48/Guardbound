import json
from pathlib import Path

p = Path("results/phase14/batch00.jsonl")
if not p.exists():
    print("batch00.jsonl not created yet (first run still in progress)")
    raise SystemExit
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"records so far: {len(rows)}/18")
for r in rows:
    res = r.get("residency") or {}
    print("  %-24s %-20s turns=%-5s rt=%-8s peak=%-7s loads=%-3s evicts=%-3s %s" % (
        r.get("run_id"), str(r.get("failure_class"))[:20], r.get("num_turns"),
        round(r.get("runtime") or 0, 1), r.get("peak_vram_gb"),
        res.get("loads"), res.get("evictions"), (r.get("error_type") or "")))
print()
from collections import Counter
print("failure_class:", dict(Counter(r.get("failure_class") for r in rows)))
