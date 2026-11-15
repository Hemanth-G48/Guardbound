"""Phase 14.8 — verify the canonical schema covers exactly the observed surface."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", ROOT / "scripts" / "phase14_full_reproduction.py")
P14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P14)

canon = list(P14.RUN_RECORD_FIELDS)
print(f"RUN_RECORD_FIELDS length      : {len(canon)}")
print(f"unique                        : {len(set(canon))}")
print(f"ordered (no dupes preserved)  : {canon == list(dict.fromkeys(canon))}")

# observed surface from the Phase 14.7 artifact
rows = [json.loads(l) for l in
        (ROOT / "results/phase14_7_18_run_production_validation.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]
observed = set()
for r in rows:
    observed |= set(r.keys())
print(f"\nobserved 14.7 surface         : {len(observed)} keys")

missing = sorted(observed - set(canon))
extra = sorted(set(canon) - observed)
print(f"in observed but NOT canonical : {missing or 'none'}")
print(f"in canonical but NOT observed : {extra or 'none'}")

print("\n=== builder behaviour ===")
rec = P14.build_run_record(run_id="X", success=False)
print(f"  empty builder keys          : {len(rec)}")
print(f"  keys == canonical           : {set(rec) == set(canon)}")
print(f"  default values are all None : {all(v is None for v in rec.values())}")
rec2 = P14.build_run_record(run_id="Y", rubric_scores=[3, 1], nbf_scores=[])
print(f"  overrides applied           : run_id={rec2['run_id']} "
      f"rubric_scores={rec2['rubric_scores']} nbf_scores={rec2['nbf_scores']}")
print(f"  untouched fields still None : num_turns={rec2['num_turns']}")

print("\n=== helper equivalence: nbf_stats_from vs the original inline formula ===")


def original(nbf_scores, filtered):
    if nbf_scores:
        return {"n": len(nbf_scores), "filter_count": filtered,
                "min": round(min(nbf_scores), 4), "max": round(max(nbf_scores), 4),
                "mean": round(sum(nbf_scores) / len(nbf_scores), 4)}
    return {"n": 0, "filter_count": 0, "min": None, "max": None, "mean": None}


cases = [[], [0.1], [1.0, 2.0, 3.0], [-0.5, 0.0, 0.5, 0.25],
         [round(float(i) / 7, 4) for i in range(50)]]
ok = True
for cs in cases:
    for f in (0, 1, 7):
        a = P14.nbf_stats_from(cs, f)
        b = original(cs, f)
        if a != b:
            ok = False
            print(f"  MISMATCH for {len(cs)} scores / filtered={f}: {a} != {b}")
print(f"  identical on {len(cases) * 3} input combinations: {ok}")

print("\n=== JSON round-trip of a canonical record ===")
blob = json.dumps(rec, ensure_ascii=False)
back = json.loads(blob)
print(f"  round-trips with identical keys: {set(back) == set(canon)}")
print(f"  nulls survive serialization    : {back['num_turns'] is None}")
