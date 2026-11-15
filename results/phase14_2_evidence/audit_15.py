"""Phase 15 §4 — read-only audit of the interrupted pilot."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts" / "phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)

OUT = ROOT / "results/phase14/batch00.jsonl"
rows = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
ids = [r["run_id"] for r in rows]

print("=" * 92)
print("PHASE 15 §4 — READ-ONLY AUDIT")
print("=" * 92)
print(f"1. pilot artifact        : {OUT.relative_to(ROOT)}")
print(f"   records               : {len(rows)}")

# 2. expected matrix
expected = [f"{c}_{a}_{g:03d}"
            for g in range(30)
            for a in ("crescendo", "opposite_day", "acronym")
            for c in ("ON", "OFF")]
print(f"\n2. expected matrix       : {len(expected)} run_ids (30 goals x 3 attacks x 2 cond)")

done = P14.existing_run_ids(OUT)
remaining = [r for r in expected if r not in done]
print(f"\n3. resume/dedup mechanism: P14.existing_run_ids(out_path) -> done_ids")
print(f"   completed ids parsed  : {len(done)}")
print(f"   remaining to run      : {len(remaining)}")
print(f"   first 4 remaining     : {remaining[:4]}")
print(f"   last remaining        : {remaining[-1]}")
rem_goals = sorted({int(r.split('_')[-1]) for r in remaining})
print(f"   remaining goals       : {rem_goals[0]}..{rem_goals[-1]} (n={len(rem_goals)})")

print(f"\n4. integrity of the 108 completed records")
print(f"   duplicates            : {len(ids) - len(set(ids))}")
print(f"   unexpected ids        : {len(set(ids) - set(expected))}")
print(f"   schema violations     : {sum(1 for r in rows if set(r) != set(P14.RUN_RECORD_FIELDS))}")
off = [r for r in rows if r["condition"] == "off"]
print(f"   NBF-OFF contamination : "
      f"{sum(1 for r in off if r['nbf_enabled'] or r['nbf_scores'] != [] or r['filtered_queries'] != 0)}/{len(off)}")
print(f"   filter_trials nulls   : {sum(1 for r in rows if r['filter_trials'] is None)}")
print(f"   failure_class         : {dict(Counter(r['failure_class'] for r in rows))}")

print(f"\n5. frozen config         : {P14.CONFIG_PATH}")
cfg = P14.load_config(str(P14.CONFIG_PATH))
print(f"   models                : "
      f"{ {r: cfg['models'][r]['model'] for r in ('attacker','target','evaluator')} }")
print(f"   max_turns={cfg['attacks']['max_turns']}  threshold={cfg['nbf']['threshold']}  "
      f"seed={cfg['experiment']['seed']}")

print(f"\n6. VRAM guard thresholds (must be unchanged by the fix)")
from guardbound.llm.model_manager import ModelManager
m = ModelManager()
print(f"   min_free_before_load_gb = {m.min_free_before_load_gb}")
print(f"   min_free_after_load_gb  = {m.min_free_after_load_gb}")
print(f"   min_free_per_run_gb     = {m.min_free_per_run_gb}   <- the gate that tripped")
print(f"   max_reserved_gb         = {m.max_reserved_gb}")

print(f"\n7. Phase 14.2 evidence")
bak = Path(r"C:\Users\CSLAB\AppData\Local\Temp\commandcode\C--Users-CSLAB-Desktop-hemanth-Guardbound\a39f1601-6d84-42e7-9588-72635c3681d6\scratchpad\batch00_phase14_2_backup.jsonl")
print(f"   scratchpad backup     : {bak.exists()} ({len(bak.read_text(encoding='utf-8').splitlines()) if bak.exists() else 0} records)")
pdir = ROOT / "results/phase15_pilot_30/trackA_frozen"
print(f"   pilot dir exists      : {pdir.exists()}")
print(f"   pilot dir contents    : {sorted(p.name for p in pdir.iterdir()) if pdir.exists() else 'n/a'}")
