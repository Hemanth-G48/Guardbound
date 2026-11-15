"""Phase 15 pilot — evidence snapshot for the interim report."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts/phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)

rows = [json.loads(l) for l in (ROOT / "results/phase14/batch00.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=" * 96)
print(f"PHASE 15 PILOT — EVIDENCE SNAPSHOT ({len(rows)}/180 records)")
print("=" * 96)

print("\n%-24s %-4s %-18s %-6s %-6s %-7s %-8s %s" % (
    "run_id", "cond", "failure_class", "trials", "turns", "filtered", "runtime", "nbf_scores"))
for r in rows:
    print("%-24s %-4s %-18s %-6s %-6s %-7s %-8s %s" % (
        r["run_id"], r["condition"], r["failure_class"], r["filter_trials"],
        r["num_turns"], r["filtered_queries"],
        f"{r.get('runtime', 0):.0f}s", len(r["nbf_scores"])))

jpe = [r for r in rows if r["failure_class"] == "JSON_PARSE_ERROR"]
print(f"\n=== THE JSON_PARSE_ERROR RECORD (validates Phase 14.8 + 14.8.x) ===")
for r in jpe:
    print(f"  run_id            : {r['run_id']}")
    print(f"  success           : {r['success']}")
    print(f"  failure_class     : {r['failure_class']}")
    print(f"  error_type        : {r['error_type']}")
    print(f"  error_message     : {str(r['error_message'])[:130]}")
    print(f"  turns             : {r['turns']!r}   <- must be null, NOT 0")
    print(f"  num_turns         : {r['num_turns']!r}   <- must be null, NOT 0")
    print(f"  accepted_query_count : {r['accepted_query_count']!r}")
    print(f"  termination_reason: {r['termination_reason']!r}")
    print(f"  filter_trials     : {r['filter_trials']}   <- the 14.8.x fix")
    print(f"  rubric_scores     : {r['rubric_scores']}")
    print(f"  nbf_scores        : {r['nbf_scores']}")
    print(f"  filtered_queries  : {r['filtered_queries']}")
    print(f"  nbf_enabled       : {r['nbf_enabled']}")
    print(f"  raw outputs kept  : {len(r['attacker_raw_outputs'])} entries")
    for e in r["attacker_raw_outputs"][:2]:
        print(f"    type={e.get('type')} raw={str(e.get('raw'))[:100]!r}")
    print(f"  keys              : {len(r)} (canonical {len(P14.RUN_RECORD_FIELDS)})")

print("\n=== per-condition aggregates so far ===")
for cond in ("on", "off"):
    sub = [r for r in rows if r["condition"] == cond]
    if not sub:
        continue
    turns = [r["num_turns"] for r in sub if r["num_turns"] is not None]
    print(f"  {cond.upper():<4} n={len(sub)}  "
          f"nbf_enabled={dict(Counter(str(r['nbf_enabled']) for r in sub))}  "
          f"nbf_scores_total={sum(len(r['nbf_scores']) for r in sub)}  "
          f"filtered_total={sum(r['filtered_queries'] or 0 for r in sub)}  "
          f"mean_turns={sum(turns)/len(turns):.2f}" if turns else "")

print("\n=== per-attack aggregates so far ===")
for atk in ("crescendo_paper", "opposite_day", "acronym"):
    sub = [r for r in rows if r["attack"] == atk]
    if not sub:
        continue
    rt = [r.get("runtime") or 0 for r in sub]
    print(f"  {atk:<16} n={len(sub)}  filter_trials={dict(Counter(str(r['filter_trials']) for r in sub))}"
          f"  mean_runtime={sum(rt)/len(rt):.0f}s")

print("\n=== runtime / VRAM ===")
rt = [r.get("runtime") or 0 for r in rows]
pk = [r.get("peak_vram_gb") for r in rows if r.get("peak_vram_gb")]
print(f"  total runtime   : {sum(rt)/60:.1f} min")
print(f"  mean per run    : {sum(rt)/len(rt):.0f} s")
print(f"  peak VRAM max   : {max(pk)} GB / 25.763")
loads = sum((r.get("residency") or {}).get("loads") or 0 for r in rows)
evicts = sum((r.get("residency") or {}).get("evictions") or 0 for r in rows)
refus = sum((r.get("residency") or {}).get("load_refusals") or 0 for r in rows)
print(f"  loads/evictions : {loads}/{evicts}   load_refusals={refus}")

print("\n=== call + evaluator telemetry totals ===")
lc = Counter()
for r in rows:
    for k, v in (r.get("llm_calls") or {}).items():
        lc[k] += v or 0
print(f"  llm_calls: {dict(lc)}")
ep = sum(sum((r.get("evaluator_parse_failures") or {}).get("json_calls", {}).values())
         for r in rows)
ef = sum((r.get("evaluator_parse_failures") or {}).get("total", 0) for r in rows)
ap = sum(sum((r.get("attacker_parse_failures") or {}).get("json_calls", {}).values())
         for r in rows)
af = sum((r.get("attacker_parse_failures") or {}).get("total", 0) for r in rows)
print(f"  evaluator JSON  : {ep - ef}/{ep} ok ({((ep-ef)/ep*100) if ep else 0:.1f}%)")
print(f"  attacker  JSON  : {ap - af}/{ap} ok ({((ap-af)/ap*100) if ap else 0:.1f}%)")
