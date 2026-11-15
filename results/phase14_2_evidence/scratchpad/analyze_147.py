"""Phase 14.7 — full analysis of the 18-run production validation."""
import json
from collections import Counter, defaultdict
from pathlib import Path

NEW = [json.loads(l) for l in Path("results/phase14_7_18_run_production_validation.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]
OLD = [json.loads(l) for l in Path("results/phase14/batch00.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=" * 100)
print("1. MATRIX COMPLETENESS")
print("=" * 100)
print("records:", len(NEW))
seen = Counter((r.get("attack"), r.get("condition")) for r in NEW)
for k in sorted(seen, key=str):
    print(f"  {str(k[0]):<16} {str(k[1]):<4} n={seen[k]}")
goals = Counter(r.get("goal_id") for r in NEW)
print("  goal_id coverage:", dict(sorted(goals.items(), key=lambda kv: str(kv[0]))))
print("  unique run_ids :", len({r.get("run_id") for r in NEW}))

print()
print("=" * 100)
print("2. PER-RUN RESULTS")
print("=" * 100)
print("%-24s %-6s %-19s %-6s %-8s %-8s %-6s %-6s %s" % (
    "run_id", "cond", "failure_class", "turns", "rt_s", "peak_gb", "loads", "evicts", "err"))
for r in NEW:
    res = r.get("residency") or {}
    print("%-24s %-6s %-19s %-6s %-8s %-8s %-6s %-6s %s" % (
        r.get("run_id"), r.get("condition"), str(r.get("failure_class"))[:19],
        r.get("num_turns"), round(r.get("runtime") or 0, 1), r.get("peak_vram_gb"),
        res.get("loads"), res.get("evictions"), (r.get("error_type") or "")[:28]))

print()
print("=" * 100)
print("3. FAILURE TAXONOMY")
print("=" * 100)
print("  failure_class:", dict(Counter(r.get("failure_class") for r in NEW)))
print("  error_type   :", dict(Counter(r.get("error_type") for r in NEW)))
print("  expected taxonomy categories present:")
tax = {
    "MODEL_CAPABILITY": 0, "JSON_PARSE_ERROR": 0, "PARSER_FAILURE": 0,
    "DECODER_FAILURE": 0, "NBF_SEMANTICS_FAILURE": 0, "VRAM_SAFETY_FAILURE": 0,
    "MODEL_IDENTITY_FAILURE": 0, "MODEL_RESIDENCY_FAILURE": 0,
    "INFRASTRUCTURE_FAILURE": 0, "ALGORITHM_FAILURE": 0,
}
for r in NEW:
    fc = r.get("failure_class")
    if fc == "JSON_PARSE_ERROR":
        tax["JSON_PARSE_ERROR"] += 1
    elif fc in ("VRAM_SAFETY_FAILURE", "MODEL_IDENTITY_FAILURE", "NBF_SEMANTICS_FAILURE",
                "MODEL_LOAD_FAILURE", "MODEL_EVICTION_FAILURE", "CACHE_RELEASE_FAILURE",
                "RESIDENCY_STATE_FAILURE", "INFRASTRUCTURE_FAILURE"):
        tax[fc] = tax.get(fc, 0) + 1
    elif fc == "SUCCESS":
        pass
for k, v in tax.items():
    print(f"    {k:<26} {v}")

print()
print("=" * 100)
print("4. NBF OFF INVARIANT (hard)")
print("=" * 100)
off = [r for r in NEW if r.get("condition") == "off"]
ok = True
for r in off:
    ns = r.get("nbf_scores")
    fq = r.get("filtered_queries")
    n_ok = (ns == [])
    f_ok = (fq in (0, None))
    ok &= n_ok and f_ok
    print(f"  {r['run_id']:<24} nbf_scores={ns} filtered_queries={fq} "
          f"{'OK' if (n_ok and f_ok) else 'VIOLATION'}")
print(f"  ALL OFF runs satisfy nbf_scores==[] and filtered_queries in (0,None): {ok}")
print(f"  OFF runs with filtered_queries == 0 (literal): "
      f"{sum(1 for r in off if r.get('filtered_queries') == 0)}/{len(off)}")
print("  NOTE: filtered_queries is null (not 0) where the run produced no turn;")
print("        same convention as Phase 14.2 - reported, not normalised.")

print()
print("=" * 100)
print("5. NBF ON BEHAVIOUR")
print("=" * 100)
on = [r for r in NEW if r.get("condition") == "on"]
for r in on:
    print(f"  {r['run_id']:<24} nbf_enabled={r.get('nbf_enabled')} "
          f"nbf_scores={len(r.get('nbf_scores') or [])} "
          f"filtered={r.get('filtered_queries')} turns={r.get('num_turns')}")
active = all(r.get("nbf_enabled") for r in on)
print(f"\n  NBF active on all ON runs            : {active}")
print(f"  ON runs recording nbf_scores (>0)    : "
      f"{sum(1 for r in on if (r.get('nbf_scores') or []))}/{len(on)}")
print(f"  ON runs with filtered_queries == 0   : "
      f"{sum(1 for r in on if r.get('filtered_queries') == 0)}/{len(on)}")
zero_turn_on = [r for r in on if (r.get("num_turns") or 0) == 0]
print(f"  ON runs that stopped at 0 turns      : {len(zero_turn_on)}")
for r in zero_turn_on:
    print(f"     {r['run_id']:<24} nbf_scores={len(r.get('nbf_scores') or [])} "
          f"filtered={r.get('filtered_queries')} -> all candidates filtered")

print()
print("=" * 100)
print("6. EVALUATOR BEHAVIOUR (the model under test)")
print("=" * 100)
ev_fail = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in NEW)
ev_json = sum(sum((r.get("evaluator_parse_failures") or {}).get("json_calls", {}).values())
              for r in NEW)
ev_calls = sum(sum((r.get("evaluator_purpose_calls") or {}).values()) for r in NEW)
purposes = Counter()
for r in NEW:
    for k, v in (r.get("evaluator_purpose_calls") or {}).items():
        purposes[k] += v
print(f"  evaluator calls total              : {ev_calls}")
print(f"  evaluator json_format=True calls   : {ev_json}")
print(f"  evaluator JSON parse failures      : {ev_fail}")
print(f"  evaluator success rate             : "
      f"{(1 - ev_fail / ev_json) * 100 if ev_json else 0:.1f}%")
print(f"  evaluator purposes                 : {dict(purposes)}")
crashes = sum(1 for r in NEW if "AttributeError" in str(r.get("error_type") or "")
              or "AttributeError" in str(r.get("error_message") or ""))
print(f"  runs with an AttributeError        : {crashes}")
evmodels = {r.get("evaluator_model") for r in NEW}
print(f"  evaluator_model                    : {evmodels}")

print()
print("=" * 100)
print("7. JSON RELIABILITY - ATTACKER (non-evaluator JSON path)")
print("=" * 100)
atk_fail = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in NEW)
atk_json = sum(sum((r.get("attacker_parse_failures") or {}).get("json_calls", {}).values())
               for r in NEW)
print(f"  attacker json_format=True calls    : {atk_json}")
print(f"  attacker JSON parse failures       : {atk_fail}")
print(f"  attacker success rate              : "
      f"{(1 - atk_fail / atk_json) * 100 if atk_json else 0:.1f}%")
print(f"  runs terminating with JSON_PARSE_ERROR: "
      f"{sum(1 for r in NEW if r.get('failure_class') == 'JSON_PARSE_ERROR')}")
for r in NEW:
    if r.get("failure_class") == "JSON_PARSE_ERROR":
        print(f"    {r['run_id']}: {(r.get('error_message') or '')[:220]}")

print()
print("=" * 100)
print("8. GPU RESIDENCY")
print("=" * 100)
tot_loads = sum((r.get("residency") or {}).get("loads") or 0 for r in NEW)
tot_evicts = sum((r.get("residency") or {}).get("evictions") or 0 for r in NEW)
refusals = sum((r.get("residency") or {}).get("load_refusals") or 0 for r in NEW)
max_res = 0
for r in NEW:
    snap = (r.get("residency") or {}).get("snapshot") or {}
    max_res = max(max_res, snap.get("resident_count") or 0)
print(f"  total model loads / evictions      : {tot_loads} / {tot_evicts}")
print(f"  total load refusals                : {refusals}")
print(f"  max distinct models resident       : {max_res}")
ids = {tuple(sorted({r.get("attacker_model"), r.get("target_model"),
                     r.get("evaluator_model")})) for r in NEW}
print(f"  model-id triples observed          : {ids}")
identity_ok = all(len(s) == 3 for s in ids)
print(f"  three distinct ids in every run    : {identity_ok}")
print(f"  loads by model (top-level counts)  : "
      f"mean {tot_loads/len(NEW):.1f} per run, max "
      f"{max((r.get('residency') or {}).get('loads') or 0 for r in NEW)}")

print()
print("=" * 100)
print("9. VRAM")
print("=" * 100)
peaks = [r.get("peak_vram_gb") for r in NEW if r.get("peak_vram_gb")]
frees = [(r.get("system") or {}).get("gpu_free_gb") for r in NEW]
temps = [(r.get("system") or {}).get("gpu_temp_c") for r in NEW]
print(f"  peak_vram_gb max / mean            : {max(peaks)} / {sum(peaks)/len(peaks):.3f}")
print(f"  min driver-free at run end         : {min(f for f in frees if f is not None)}")
print(f"  max GPU temperature                : {max(t for t in temps if t is not None)} C")
print(f"  VRAM_SAFETY_FAILURE runs           : "
      f"{sum(1 for r in NEW if r.get('failure_class') == 'VRAM_SAFETY_FAILURE')}")
print(f"  card total                         : 25.763 GB")

print()
print("=" * 100)
print("10. RUNTIME")
print("=" * 100)
rts = [r.get("runtime") or 0 for r in NEW]
print(f"  total wall time                    : {sum(rts):.1f} s ({sum(rts)/60:.1f} min)")
print(f"  per-run mean / median / max / min  : {sum(rts)/len(rts):.1f} / "
      f"{sorted(rts)[len(rts)//2]:.1f} / {max(rts):.1f} / {min(rts):.1f} s")

print()
print("=" * 100)
print("11. COMPARISON WITH PHASE 14.2 (factual only, no ranking)")
print("=" * 100)


def prof(rows, label):
    atkf = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in rows)
    atkj = sum(sum((r.get("attacker_parse_failures") or {}).get("json_calls", {}).values())
               for r in rows)
    evf = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in rows)
    evj = sum(sum((r.get("evaluator_parse_failures") or {}).get("json_calls", {}).values())
              for r in rows)
    ld = sum((r.get("residency") or {}).get("loads") or 0 for r in rows)
    ev = sum((r.get("residency") or {}).get("evictions") or 0 for r in rows)
    pk = [r.get("peak_vram_gb") for r in rows if r.get("peak_vram_gb")]
    rt = [r.get("runtime") or 0 for r in rows]
    turns = [r.get("num_turns") or 0 for r in rows]
    print(f"\n  {label}")
    print(f"    failure_class        : {dict(Counter(r.get('failure_class') for r in rows))}")
    print(f"    attacker_model       : {sorted({r.get('attacker_model') for r in rows})}")
    print(f"    evaluator_model      : {sorted({r.get('evaluator_model') for r in rows})}")
    print(f"    attacker JSON        : {atkj - atkf}/{atkj} ok "
          f"({(1-atkf/atkj)*100 if atkj else 0:.1f}%)")
    print(f"    evaluator JSON       : {evj - evf}/{evj} ok "
          f"({(1-evf/evj)*100 if evj else 0:.1f}%)")
    print(f"    loads / evictions    : {ld} / {ev}   load_refusals="
          f"{sum((r.get('residency') or {}).get('load_refusals') or 0 for r in rows)}")
    print(f"    peak_vram max        : {max(pk) if pk else None}")
    print(f"    runtime total        : {sum(rt):.0f} s   mean turns {sum(turns)/len(turns):.2f}")


prof(OLD, "PHASE 14.2 18-run matrix (attacker Qwen3-4B==evaluator Qwen3-4B, 2 distinct models)")
prof(NEW, "PHASE 14.7 18-run validation (attacker Qwen3-4B / target Phi-4-mini / evaluator Llama-3.2-3B)")
