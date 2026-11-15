import json
from pathlib import Path

rows = [json.loads(l) for l in Path("results/phase14/batch00.jsonl")
        .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=== VRAM composition of the last 6 records ===")
print("%-24s %-9s %-9s %-9s %-9s %s" % (
    "run_id", "sys_free", "alloc", "reserved", "cached_free", "residency snapshot"))
for r in rows[-6:]:
    s = r.get("system") or {}
    snap = (r.get("residency") or {}).get("snapshot") or {}
    print("%-24s %-9s %-9s %-9s %-9s %s" % (
        r["run_id"], s.get("gpu_free_gb"), s.get("gpu_allocated_gb"),
        s.get("gpu_reserved_gb"), snap.get("cached_free_gb"),
        snap.get("resident_model_ids")))

print("\n=== the 5 records with the least free VRAM ===")
worst = sorted(rows, key=lambda r: (r.get("system") or {}).get("gpu_free_gb") or 99)[:5]
for r in worst:
    s = r.get("system") or {}
    snap = (r.get("residency") or {}).get("snapshot") or {}
    print(f"  {r['run_id']:<24} free={s.get('gpu_free_gb')} alloc={s.get('gpu_allocated_gb')} "
          f"reserved={s.get('gpu_reserved_gb')} cached_free={snap.get('cached_free_gb')} "
          f"resident={snap.get('resident_model_ids')} peak={r.get('peak_vram_gb')}")

print("\n=== free + cached_free (effective availability) over the run ===")
eff = []
for r in rows:
    s = r.get("system") or {}
    snap = (r.get("residency") or {}).get("snapshot") or {}
    f, c = s.get("gpu_free_gb"), snap.get("cached_free_gb")
    if f is not None and c is not None:
        eff.append((r["run_id"], f, c, round(f + c, 3)))
print(f"  effective free: min={min(e[3] for e in eff)}  max={max(e[3] for e in eff)}")
print(f"  raw free      : min={min(e[1] for e in eff)}")
print("  records where RAW free < 6 GB but EFFECTIVE free >= 6 GB:")
for rid, f, c, e in eff:
    if f < 6.0 <= e:
        print(f"    {rid:<24} raw={f} cached={c} effective={e}")

print("\n=== per-run overhead: loads per run ===")
lc = [(r["run_id"], (r.get("residency") or {}).get("loads") or 0) for r in rows]
print(f"  loads/run: min={min(l for _, l in lc)} max={max(l for _, l in lc)} "
      f"mean={sum(l for _, l in lc)/len(lc):.1f}")
print(f"  total loads={sum(l for _, l in lc)}")
