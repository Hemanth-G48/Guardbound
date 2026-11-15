"""Part 5 - evaluator/attacker/target call frequency from existing Phase 12/13/14 results.

Reads only. No GPU, no model loading.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound")
FILES = [
    ROOT / "results/phase12/smoke_A_nbf_off.jsonl",
    ROOT / "results/phase13/nbf_off.jsonl",
    ROOT / "results/phase13/nbf_on.jsonl",
    ROOT / "results/phase14/batch00_pre14_1.jsonl",
    ROOT / "results/phase14/phase14_1_test_results.jsonl",
]


def get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


print("=" * 90)
print("PART 5 - CALL FREQUENCY EXTRACTION")
print("=" * 90)

rows = []
for f in FILES:
    if not f.exists():
        print(f"  MISSING: {f}")
        continue
    n = 0
    with open(f, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            calls = r.get("llm_calls") or {}
            rows.append({
                "file": f.name,
                "run_id": r.get("run_id"),
                "attack": r.get("attack"),
                "condition": r.get("condition"),
                "success": r.get("success"),
                "failure_class": r.get("failure_class"),
                "num_turns": r.get("num_turns"),
                "runtime": r.get("runtime") or r.get("runtime_seconds"),
                "attacker_calls": get(calls, "attacker"),
                "target_calls": get(calls, "target"),
                "evaluator_calls": get(calls, "evaluator"),
                "ev_purpose": r.get("evaluator_purpose_calls") or {},
                "at_purpose": r.get("attacker_purpose_calls") or {},
                "tg_purpose": r.get("target_purpose_calls") or {},
            })
    print(f"  {f.name:<45} {n:>4} records")

print(f"\nTotal records parsed: {len(rows)}")

# ---------------------------------------------------------------- per attack
print("\n" + "=" * 90)
print("PER-ATTACK AGGREGATE (completed runs only, num_turns present)")
print("=" * 90)
agg = defaultdict(list)
for r in rows:
    if r["num_turns"] is None:
        continue
    agg[r["attack"]].append(r)

hdr = f"{'attack':<22}{'runs':>5}{'atk':>8}{'tgt':>8}{'eval':>8}{'turns':>7}{'runtime_s':>11}{'ev/run':>8}"
print(hdr)
print("-" * len(hdr))
for atk, rs in sorted(agg.items(), key=lambda kv: str(kv[0])):
    n = len(rs)

    def mean(key):
        vals = [r[key] for r in rs if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"{str(atk):<22}{n:>5}{mean('attacker_calls'):>8.1f}"
          f"{mean('target_calls'):>8.1f}{mean('evaluator_calls'):>8.1f}"
          f"{mean('num_turns'):>7.1f}{mean('runtime'):>11.1f}"
          f"{mean('evaluator_calls'):>8.1f}")

# ------------------------------------------------- evaluator purpose breakdown
print("\n" + "=" * 90)
print("EVALUATOR CALL BREAKDOWN BY PURPOSE (mean count per run)")
print("=" * 90)
purposes = sorted({p for r in rows for p in (r["ev_purpose"] or {})})
if not purposes:
    print("  NO evaluator_purpose_calls present in any record.")
    print("  (records predate the purpose-counter instrumentation, or the runs errored early)")
for atk in sorted({r["attack"] for r in rows}, key=str):
    rs = [r for r in rows if r["attack"] == atk and r["num_turns"] is not None]
    if not rs:
        continue
    print(f"\n  {atk}  (n={len(rs)})")
    for p in purposes:
        vals = [(r["ev_purpose"] or {}).get(p, 0) for r in rs]
        if any(vals):
            print(f"    {p:<40} mean={sum(vals)/len(vals):>6.2f}  max={max(vals):>4}")

print("\n" + "=" * 90)
print("ATTACKER CALL BREAKDOWN BY PURPOSE (mean count per run)")
print("=" * 90)
apurp = sorted({p for r in rows for p in (r["at_purpose"] or {})})
for atk in sorted({r["attack"] for r in rows}, key=str):
    rs = [r for r in rows if r["attack"] == atk and r["num_turns"] is not None]
    if not rs:
        continue
    print(f"\n  {atk}  (n={len(rs)})")
    for p in apurp:
        vals = [(r["at_purpose"] or {}).get(p, 0) for r in rs]
        if any(vals):
            print(f"    {p:<40} mean={sum(vals)/len(vals):>6.2f}  max={max(vals):>4}")

# raw per-record dump for the largest sample
print("\n" + "=" * 90)
print("RAW PER-RECORD (all)")
print("=" * 90)
print(f"{'file':<28}{'run_id':<32}{'turns':>6}{'atk':>6}{'tgt':>6}{'eval':>6}{'rt_s':>9}")
for r in sorted(rows, key=lambda r: (str(r["file"]), str(r["run_id"]))):
    print(f"{str(r['file'])[:27]:<28}{str(r['run_id'])[:31]:<32}"
          f"{str(r['num_turns']):>6}{str(r['attacker_calls']):>6}"
          f"{str(r['target_calls']):>6}{str(r['evaluator_calls']):>6}"
          f"{(f'{r[chr(114)+chr(117)+chr(110)+chr(116)+chr(105)+chr(109)+chr(101)]:.1f}' if r['runtime'] else 'None'):>9}")
