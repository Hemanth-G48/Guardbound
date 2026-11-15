"""Inventory REAL target responses already captured by the project.

These are probe inputs requiring no authoring: they were produced by the real
target model (microsoft/Phi-4-mini-instruct) against real HarmBench goals
during the Phase 14.2 18-run matrix.
"""
import json
from pathlib import Path

cand = [
    "results/phase14/batch00.jsonl",
    "results/phase13/nbf_off.jsonl",
    "results/phase13/nbf_on.jsonl",
    "results/phase12/smoke_A_nbf_off.jsonl",
    "results/phase14/batch00_pre14_1.jsonl",
]

records = []
for p in cand:
    f = Path(p)
    if not f.exists():
        continue
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"{p}: {len(rows)} rows")
    records.extend(rows)

print(f"\ntotal rows across sources: {len(records)}")

pairs = []
for r in records:
    goal = r.get("goal") or (r.get("turns") or [{}])[0].get("query") or ""
    for t in (r.get("turns") or []):
        q = t.get("query") or ""
        resp = t.get("response")
        if resp and isinstance(resp, str):
            pairs.append({"run_id": r.get("run_id"), "attack": r.get("attack"),
                          "goal": (r.get("goal") or q)[:200], "response": resp})
    fr = r.get("final_response")
    if fr and isinstance(fr, str):
        pairs.append({"run_id": r.get("run_id"), "attack": r.get("attack"),
                      "goal": (r.get("goal") or "")[:200], "response": fr})

# dedupe by response text
seen = set()
uniq = []
for p in pairs:
    if p["response"] not in seen:
        seen.add(p["response"])
        uniq.append(p)

print(f"target-response pairs: {len(pairs)}   distinct: {len(uniq)}")
print()
REFUSAL_MARKERS = ["cannot", "can't", "won't", "will not", "unable", "not able",
                   "I'm sorry", "I am sorry", "sorry,", "apolog", "not appropriate",
                   "cannot assist", "can't assist", "cannot help", "can't help"]
for i, p in enumerate(uniq[:40]):
    r = p["response"]
    low = r.lower()
    refused = any(m.lower() in low for m in REFUSAL_MARKERS)
    print(f"{i:3d} len={len(r):5d} refusal_marker={str(refused):5s} "
          f"attack={str(p['attack'])[:14]:14s} {r[:130]!r}")
