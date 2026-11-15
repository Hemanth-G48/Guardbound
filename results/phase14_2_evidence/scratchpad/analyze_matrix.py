import json
from collections import Counter
from pathlib import Path

p = Path("results/phase14/batch00.jsonl")
recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print("records:", len(recs))
print()
print("%-24s %-8s %-6s %-22s %-8s %-8s %s" % (
    "run_id", "attack", "cond", "failure_class", "turns", "rt_s", "residency"))
for r in recs:
    res = r.get("residency") or {}
    snap = res.get("snapshot") or {}
    print("%-24s %-8s %-6s %-22s %-8s %-8s loads=%s evicts=%s resident=%s" % (
        r.get("run_id"), str(r.get("attack"))[:8], r.get("condition"),
        str(r.get("failure_class")), r.get("num_turns"),
        round(r.get("runtime") or 0, 1),
        res.get("loads"), res.get("evictions"),
        snap.get("resident_model_ids")))

print()
print("failure_class counts:", dict(Counter(r.get("failure_class") for r in recs)))
print("error_type counts   :", dict(Counter(r.get("error_type") for r in recs)))
print()
print("=== peaks ===")
for r in recs:
    res = r.get("residency") or {}
    s = r.get("system") or {}
    print("%-24s peak_vram=%s residency_peak=%s free=%s reserved=%s temp=%s util=%s" % (
        r.get("run_id"), r.get("peak_vram_gb"), res.get("peak_vram_gb"),
        s.get("gpu_free_gb"), s.get("gpu_reserved_gb"), s.get("gpu_temp_c"),
        s.get("gpu_util_pct")))

print()
print("=== rotations for the longest run ===")
worst = max(recs, key=lambda r: (r.get("residency") or {}).get("loads", 0))
print(worst.get("run_id"), "loads=", (worst.get("residency") or {}).get("loads"))
for rot in (worst.get("residency") or {}).get("rotations") or []:
    print("   ", rot)
