"""Phase 15 pilot — diagnose the abort at 108/180."""
import json
from collections import Counter
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14/batch00.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"records: {len(rows)}  goals: {sorted({r['goal_id'] for r in rows})}")
print(f"classes: {dict(Counter(r['failure_class'] for r in rows))}")
print(f"last record: {rows[-1]['run_id']}")

print("\n=== VRAM / FREE-VRAM TREND (every record) ===")
print("%-24s %-6s %-9s %-9s %-9s %-8s %-6s %-6s %s" % (
    "run_id", "free_gb", "peak_gb", "res_peak", "reserved", "loads", "evicts", "refus", "classes"))
prev_free = None
drops = []
for r in rows:
    s = r.get("system") or {}
    res = r.get("residency") or {}
    snap = res.get("snapshot") or {}
    free = s.get("gpu_free_gb")
    print("%-24s %-6s %-9s %-9s %-9s %-8s %-6s %-6s %s" % (
        r["run_id"], free, r.get("peak_vram_gb"), res.get("peak_vram_gb"),
        s.get("gpu_reserved_gb"), res.get("loads"), res.get("evictions"),
        res.get("load_refusals"), r["failure_class"]))
    if free is not None and prev_free is not None:
        drops.append(round(free - prev_free, 3))
    prev_free = free

frees = [(r.get("system") or {}).get("gpu_free_gb") for r in rows]
frees = [f for f in frees if f is not None]
print(f"\nfree VRAM: first={frees[0]}  min={min(frees)}  max={max(frees)}  last={frees[-1]}")
print(f"monotonic decline? {all(a >= b - 0.05 for a, b in zip(frees, frees[1:]))}")

print("\n=== per-goal: free at end of goal, peaks, refusals ===")
for g in sorted({r["goal_id"] for r in rows}):
    sub = [r for r in rows if r["goal_id"] == g]
    fr = [(r.get("system") or {}).get("gpu_free_gb") for r in sub]
    pk = [r.get("peak_vram_gb") for r in sub if r.get("peak_vram_gb")]
    ld = sum((r.get("residency") or {}).get("loads") or 0 for r in sub)
    ev = sum((r.get("residency") or {}).get("evictions") or 0 for r in sub)
    rf = sum((r.get("residency") or {}).get("load_refusals") or 0 for r in sub)
    print(f"  goal {g:02d}: n={len(sub)} free_last={fr[-1] if fr else None} "
          f"min_free={min(f for f in fr if f is not None) if any(f for f in fr) else None} "
          f"peak_max={max(pk) if pk else None} loads={ld} evicts={ev} refusals={rf}")

print("\n=== totals ===")
ld = sum((r.get("residency") or {}).get("loads") or 0 for r in rows)
ev = sum((r.get("residency") or {}).get("evictions") or 0 for r in rows)
rf = sum((r.get("residency") or {}).get("load_refusals") or 0 for r in rows)
print(f"loads={ld} evictions={ev} refusals={rf}  (residency leak if loads-evicts grows)")
pk = [r.get("peak_vram_gb") for r in rows if r.get("peak_vram_gb")]
print(f"peak VRAM: max={max(pk)}  last={pk[-1]}")

print("\n=== json / schema integrity ===")
import importlib.util, sys
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location(
    "p", "scripts/phase14_full_reproduction.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
CANON = set(P.RUN_RECORD_FIELDS)
print(f"schema violations: {sum(1 for r in rows if set(r) != CANON)}")
print(f"filter_trials: {dict(Counter(str(r['filter_trials']) for r in rows))}")
off = [r for r in rows if r["condition"] == "off"]
print(f"NBF OFF contamination: "
      f"{sum(1 for r in off if r['nbf_enabled'] or r['nbf_scores'] != [] or r['filtered_queries'] != 0)}"
      f" / {len(off)}")
jpe = [r for r in rows if r["failure_class"] == "JSON_PARSE_ERROR"]
print(f"JSON_PARSE_ERROR: {len(jpe)}  raw kept: {sum(1 for r in jpe if r['attacker_raw_outputs'])}"
      f"  turns non-null: {sum(1 for r in jpe if r['turns'] is not None)}")

print("\n=== evaluator / attacker JSON reliability ===")
ep = sum(sum((r.get("evaluator_parse_failures") or {}).get("json_calls", {}).values()) for r in rows)
ef = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in rows)
ap = sum(sum((r.get("attacker_parse_failures") or {}).get("json_calls", {}).values()) for r in rows)
af = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in rows)
print(f"evaluator: {ep-ef}/{ep} = {((ep-ef)/ep*100) if ep else 0:.1f}%")
print(f"attacker : {ap-af}/{ap} = {((ap-af)/ap*100) if ap else 0:.1f}%")

print("\n=== successes + runtime ===")
for r in rows:
    if r["failure_class"] == "SUCCESS":
        print(f"  SUCCESS {r['run_id']} turns={r['num_turns']} rt={r.get('runtime',0):.0f}s "
              f"nbf={len(r['nbf_scores'])} filt={r['filtered_queries']}")
rt = [r.get("runtime") or 0 for r in rows]
print(f"total runtime {sum(rt)/3600:.2f} h  mean {sum(rt)/len(rt):.0f}s")
