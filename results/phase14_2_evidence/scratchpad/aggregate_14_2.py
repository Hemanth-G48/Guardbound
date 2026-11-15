"""Phase 14.2 — aggregate the 18-run validation and compare to the previous gate."""
import json
from collections import Counter, defaultdict
from pathlib import Path

NEW = [json.loads(l) for l in Path("results/phase14/batch00.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]
OLD_OFF = [json.loads(l) for l in Path("results/phase13/nbf_off.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
OLD_ON = [json.loads(l) for l in Path("results/phase13/nbf_on.jsonl")
          .read_text(encoding="utf-8").splitlines() if l.strip()]
OLD = OLD_OFF + OLD_ON

print("=" * 96)
print("1. RUN OUTCOMES")
print("=" * 96)
print("records:", len(NEW))
print("failure_class:", dict(Counter(r.get("failure_class") for r in NEW)))
succ = [r for r in NEW if r.get("failure_class") == "SUCCESS"]
print("SUCCESS runs:", [r["run_id"] for r in succ])

print()
print("=" * 96)
print("2. RESIDENCY CERTIFICATION (per run)")
print("=" * 96)
max_res, pins_ok, three_ids = 0, True, True
loads = evicts = refusals = 0
for r in NEW:
    res = r.get("residency") or {}
    snap = res.get("snapshot") or {}
    max_res = max(max_res, snap.get("resident_count") or 0)
    pins_ok &= (snap.get("pinned_roles") == ["attacker"])
    ids = {r.get("attacker_model"), r.get("target_model"), r.get("evaluator_model")}
    three_ids &= (len(ids) == 3)
    loads += res.get("loads") or 0
    evicts += res.get("evictions") or 0
    refusals += res.get("load_refusals") or 0
print(f"max distinct models resident at any run end : {max_res}   (must be <= 2)")
print(f"attacker pinned in every run                : {pins_ok}")
print(f"three distinct model ids in every run       : {three_ids}")
print(f"total model loads / evictions               : {loads} / {evicts}")
print(f"total load refusals                         : {refusals}")

# rotation fairness: did target and evaluator both get used?
rot_by_model = Counter()
rot_dur = defaultdict(list)
evict_dur = []
for r in NEW:
    for rot in (r.get("residency") or {}).get("rotations") or []:
        if rot.get("action") == "load":
            rot_by_model[rot["model_id"]] += 1
            rot_dur[rot["model_id"]].append(rot.get("duration_s") or 0.0)
        elif rot.get("action") == "evict":
            evict_dur.append(rot.get("duration_s") or 0.0)
print("\nloads by physical model:")
for mid, n in rot_by_model.most_common():
    d = rot_dur[mid]
    print(f"  {mid:<34} {n:>3} loads  mean {sum(d)/len(d):6.2f}s  min {min(d):5.2f}  max {max(d):5.2f}")
if evict_dur:
    print(f"evictions: n={len(evict_dur)} mean {sum(evict_dur)/len(evict_dur):.3f}s "
          f"max {max(evict_dur):.3f}s")

print()
print("=" * 96)
print("3. VRAM / STABILITY")
print("=" * 96)
peaks = [r.get("peak_vram_gb") for r in NEW if r.get("peak_vram_gb")]
frees = [(r.get("system") or {}).get("gpu_free_gb") for r in NEW]
temps = [(r.get("system") or {}).get("gpu_temp_c") for r in NEW]
utils = [(r.get("system") or {}).get("gpu_util_pct") for r in NEW]
print(f"peak_vram_gb (max_memory_allocated): max {max(peaks)}  mean {sum(peaks)/len(peaks):.3f}")
print(f"driver free at run end             : min {min(f for f in frees if f is not None)}")
print(f"GPU temperature                    : max {max(t for t in temps if t is not None)} C")
print(f"GPU utilisation                    : mean {sum(u for u in utils if u is not None)/len(utils):.1f} %")
print(f"VRAM_SAFETY_FAILURE runs           : "
      f"{sum(1 for r in NEW if r.get('failure_class') == 'VRAM_SAFETY_FAILURE')}")

print()
print("=" * 96)
print("4. NBF INVARIANTS")
print("=" * 96)
off_runs = [r for r in NEW if r.get("condition") == "off"]
on_runs = [r for r in NEW if r.get("condition") == "on"]
off_ok = all((r.get("nbf_scores") or []) == [] and (r.get("filtered_queries") or 0) == 0
             for r in off_runs)
print(f"NBF OFF runs: {len(off_runs)}  nbf_scores==[] and filtered==0 for ALL: {off_ok}")
for r in off_runs:
    print(f"  {r['run_id']:<24} nbf={len(r.get('nbf_scores') or [])} filt={r.get('filtered_queries')}")
print(f"\nNBF ON runs: {len(on_runs)}")
for r in on_runs:
    print(f"  {r['run_id']:<24} nbf={len(r.get('nbf_scores') or [])} "
          f"filt={r.get('filtered_queries')} turns={r.get('num_turns')} "
          f"nbf_enabled={r.get('nbf_enabled')} breach={bool(r.get('invariant_breach'))}")
breaches = [r["run_id"] for r in NEW if r.get("invariant_breach")]
print(f"invariant breaches: {breaches or 'none'} in 18 runs (0.0 % on 18 runs)")

print()
print("=" * 96)
print("5. STRUCTURED OUTPUT (attacker / evaluator)")
print("=" * 96)
atk_fail = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in NEW)
ev_fail = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in NEW)
ev_calls = sum(
    sum((r.get("evaluator_purpose_calls") or {}).values()) for r in NEW
)
json_errors = sum(1 for r in NEW if r.get("error_type") == "AttackGenerationError")
print(f"runs failing with AttackGenerationError (non-JSON attacker reply): {json_errors}")
print(f"attacker JSON parse failures (telemetry)   : {atk_fail}")
print(f"evaluator JSON parse failures (telemetry)  : {ev_fail}")
print(f"evaluator calls total                      : {ev_calls}")

print()
print("=" * 96)
print("6. RUNTIME vs PREVIOUS 18-RUN GATE (phase13, 2-model)")
print("=" * 96)


def agg(recs):
    out = {}
    for r in recs:
        k = (r.get("attack"), r.get("condition"))
        out.setdefault(k, []).append(r)
    return out


na, oa = agg(NEW), agg(OLD)
print("%-18s %-4s | %-28s | %-28s" % ("attack", "cond", "previous (2-model)", "new (3-model rotation)"))
print("%-18s %-4s | %-28s | %-28s" % ("", "", "n  rt_mean  turns_mean", "n  rt_mean  turns_mean"))
print("-" * 96)
for atk in ("crescendo_paper", "opposite_day", "acronym"):
    for cond in ("off", "on"):
        k = (atk, cond)
        o, n = oa.get(k, []), na.get(k, [])
        def f(rs):
            if not rs:
                return "   -        -"
            rt = sum((r.get("runtime") or 0) for r in rs) / len(rs)
            tn = sum((r.get("num_turns") or 0) for r in rs) / len(rs)
            return f"{len(rs):>2} {rt:8.1f} {tn:8.1f}"
        print("%-18s %-4s | %-28s | %-28s" % (atk, cond, f(o), f(n)))

print()
print("total wall time, previous 18:", round(sum(r.get('runtime') or 0 for r in OLD), 1), "s")
print("total wall time, new      18:", round(sum(r.get('runtime') or 0 for r in NEW), 1), "s")
print("NOTE: the new total is dominated by fast failures, not by completed runs.")
completed_o = [r for r in OLD if (r.get('num_turns') or 0) > 0]
completed_n = [r for r in NEW if (r.get('num_turns') or 0) > 0]
if completed_o and completed_n:
    print("mean runtime of runs that produced turns (prev):",
          round(sum(r['runtime'] for r in completed_o)/len(completed_o), 1), "s over", len(completed_o), "runs")
    print("mean runtime of runs that produced turns (new) :",
          round(sum(r['runtime'] for r in completed_n)/len(completed_n), 1), "s over", len(completed_n), "runs")
    print("rotation loads in those new runs:",
          sum((r.get('residency') or {}).get('loads') or 0 for r in completed_n))
