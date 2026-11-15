"""Phase 15 pilot — live execution-integrity monitor.

Checks ONLY execution integrity (never ASR), per the pilot spec.
"""
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts" / "phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)
CANON = set(P14.RUN_RECORD_FIELDS)

p = ROOT / "results/phase14/batch00.jsonl"
if not p.exists():
    print("monitor: batch00.jsonl not created yet")
    raise SystemExit
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

print(f"records: {len(rows)}/180  ({len(rows)/180*100:.1f}%)")

ids = [r["run_id"] for r in rows]
dupes = [k for k, v in Counter(ids).items() if v > 1]
print(f"unique run_ids: {len(set(ids))}   duplicates: {dupes or 'none'}")

# expected run ids for the 30x3x2 matrix
expected = {f"{c}_{a}_{g:03d}"
            for g in range(30)
            for a in ("crescendo", "opposite_day", "acronym")
            for c in ("ON", "OFF")}
missing = sorted(expected - set(ids))
print(f"missing run_ids: {len(missing)}" + (f"  first: {missing[:4]}" if missing else ""))

schema_bad = [r["run_id"] for r in rows if set(r.keys()) != CANON]
print(f"schema violations: {len(schema_bad)}" + (f"  {schema_bad[:3]}" if schema_bad else ""))

off = [r for r in rows if r["condition"] == "off"]
contam = [r["run_id"] for r in off
          if r["nbf_enabled"] or r["nbf_scores"] != [] or r["filtered_queries"] != 0]
print(f"NBF OFF contamination: {len(contam)} / {len(off)} OFF runs"
      + (f"  {contam[:3]}" if contam else ""))

ft = Counter(str(r["filter_trials"]) for r in rows)
ft_nulls = [r["run_id"] for r in rows if r["filter_trials"] is None]
print(f"filter_trials: {dict(ft)}   nulls: {len(ft_nulls)}"
      + (f"  {ft_nulls[:3]}" if ft_nulls else ""))

fc = Counter(r["failure_class"] for r in rows)
print(f"failure_class: {dict(fc)}")
jpe = [r for r in rows if r["failure_class"] == "JSON_PARSE_ERROR"]
print(f"JSON_PARSE_ERROR: {len(jpe)}   raw outputs preserved: "
      f"{sum(1 for r in jpe if r['attacker_raw_outputs'])}/{len(jpe)}")
mis = [r["run_id"] for r in jpe if r["turns"] is not None]
print(f"JSON_PARSE_ERROR with non-null turns (must be 0): {len(mis)}")

models = defaultdict(set)
for r in rows:
    models["attacker"].add(r["attacker_model"])
    models["target"].add(r["target_model"])
    models["evaluator"].add(r["evaluator_model"])
for k, v in models.items():
    print(f"  {k:<10} {sorted(v)}")
print(f"dtype: {dict(Counter(str(r['dtype']) for r in rows))}   "
      f"quantization: {dict(Counter(str(r['quantization']) for r in rows))}")

rt = [r.get("runtime") or 0 for r in rows]
pk = [r.get("peak_vram_gb") for r in rows if r.get("peak_vram_gb")]
if rt:
    print(f"runtime total {sum(rt)/60:.1f} min  mean {sum(rt)/len(rt):.0f}s  "
          f"max {max(rt):.0f}s   peak VRAM max {max(pk) if pk else None}")

off_turns = [r["num_turns"] for r in off if r["num_turns"] is not None]
print(f"OFF turns: mean {sum(off_turns)/len(off_turns):.2f}" if off_turns else "")
on = [r for r in rows if r["condition"] == "on"]
on_scored = sum(1 for r in on if r["nbf_scores"])
print(f"NBF ON in {len(on)} records; with populated nbf_scores: {on_scored}")
print(f"NBF ON filtered>0: {sum(1 for r in on if (r['filtered_queries'] or 0) > 0)}")

by_att = Counter(r["attack"] for r in rows)
print(f"by attack: {dict(by_att)}")
print(f"goals covered: {sorted({r['goal_id'] for r in rows})[:8]}"
      f"{' ...' if len({r['goal_id'] for r in rows}) > 8 else ''} "
      f"(n={len({r['goal_id'] for r in rows})})")
