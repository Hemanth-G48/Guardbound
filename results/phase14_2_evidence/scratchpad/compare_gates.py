"""Phase 14.2 Step 11 — compare the new rotation gate against the previous gate."""
import json
from pathlib import Path

def load(p, cond=None):
    recs = [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in recs:
        r.setdefault("condition", cond)
        r.setdefault("_source", Path(p).name)
    return recs

NEW = load("results/phase14/batch00.jsonl")
OLD = load("results/phase13/nbf_off.jsonl", "off") + load("results/phase13/nbf_on.jsonl", "on")

print("=" * 100)
print("STEP 11 - PREVIOUS 18-RUN GATE (2-model, phase13) vs NEW (3-model rotation)")
print("=" * 100)
print("%-17s %-4s | %-30s | %-30s" % ("attack", "cond", "PREVIOUS 2-model", "NEW 3-model rotation"))
print("%-17s %-4s | %-30s | %-30s" % ("", "", "n  rt_mean  turns_mean  succ", "n  rt_mean  turns_mean  succ"))
print("-" * 100)
for atk in ("crescendo_paper", "opposite_day", "acronym"):
    for cond in ("off", "on"):
        o = [r for r in OLD if r["attack"] == atk and r["condition"] == cond]
        n = [r for r in NEW if r["attack"] == atk and r["condition"] == cond]

        def fmt(rs):
            if not rs:
                return "        -          -     -"
            rt = sum((r.get("runtime") or 0) for r in rs) / len(rs)
            tn = sum((r.get("num_turns") or 0) for r in rs) / len(rs)
            sc = sum(1 for r in rs if r.get("success"))
            return f"{len(rs):>2} {rt:8.1f} {tn:11.1f} {sc:5}"
        print("%-17s %-4s | %-30s | %-30s" % (atk, cond, fmt(o), fmt(n)))

print()
print("--- totals ---")
print(f"previous 18: {round(sum(r.get('runtime') or 0 for r in OLD),1)} s, "
      f"{sum(1 for r in OLD if r.get('success'))} success, "
      f"{sum(1 for r in OLD if (r.get('num_turns') or 0) > 0)} runs with turns > 0")
print(f"new 18     : {round(sum(r.get('runtime') or 0 for r in NEW),1)} s, "
      f"{sum(1 for r in NEW if r.get('success'))} success, "
      f"{sum(1 for r in NEW if (r.get('num_turns') or 0) > 0)} runs with turns > 0")

print()
print("--- apples-to-apples: only runs that actually produced turns ---")
ot = [r for r in OLD if (r.get("num_turns") or 0) > 0]
nt = [r for r in NEW if (r.get("num_turns") or 0) > 0]
for label, rs in (("previous", ot), ("new", nt)):
    if not rs:
        print(f"{label:<10} no qualifying runs")
        continue
    rt = sum(r["runtime"] for r in rs) / len(rs)
    loads = sum((r.get("residency") or {}).get("loads") or 0 for r in rs)
    evicts = sum((r.get("residency") or {}).get("evictions") or 0 for r in rs)
    print(f"{label:<10} n={len(rs):<3} mean_rt={rt:7.1f}s  "
          f"mean_turns={sum(r['num_turns'] for r in rs)/len(rs):.1f}  "
          f"rotable loads={loads} evictions={evicts}")

print()
print("--- per-model call counts (behavioural consistency) ---")
for label, rs in (("previous", OLD), ("new", NEW)):
    tot = {"attacker": 0, "target": 0, "evaluator": 0}
    for r in rs:
        for k in tot:
            tot[k] += (r.get("llm_calls") or {}).get(k) or 0
    print(f"{label:<10} attacker={tot['attacker']:<5} target={tot['target']:<5} "
          f"evaluator={tot['evaluator']:<5}")

print()
print("--- load/eviction cost actually paid in the new gate ---")
ld, ed = [], []
for r in NEW:
    for rot in (r.get("residency") or {}).get("rotations") or []:
        if rot.get("action") == "load":
            ld.append(rot.get("duration_s") or 0.0)
        elif rot.get("action") == "evict":
            ed.append(rot.get("duration_s") or 0.0)
print(f"loads     : n={len(ld)} mean={sum(ld)/len(ld):.2f}s total={sum(ld):.1f}s")
print(f"evictions : n={len(ed)} mean={sum(ed)/len(ed):.3f}s total={sum(ed):.1f}s")
print(f"rotation overhead paid across 18 runs: {sum(ld)+sum(ed):.1f}s "
      f"({100*(sum(ld)+sum(ed))/sum(r.get('runtime') or 0 for r in NEW):.1f}% of wall time)")
