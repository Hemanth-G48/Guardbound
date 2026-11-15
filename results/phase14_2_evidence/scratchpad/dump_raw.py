import json
from pathlib import Path

r = json.loads(Path("results/phase14/batch00.jsonl").read_text(encoding="utf-8").splitlines()[0])
print("FULL error_message:")
print(r.get("error_message"))
print()
print("=== attacker_raw_outputs ===")
for item in (r.get("attacker_raw_outputs") or []):
    print(item)
print()
print("=== attacker_purpose_calls / parse_failures ===")
print(r.get("attacker_purpose_calls"))
print(r.get("attacker_parse_failures"))
print()
print("=== VRAM accounting ===")
print("peak_vram_gb (max_memory_allocated):", r.get("peak_vram_gb"))
print("system:", r.get("system"))
res = r.get("residency") or {}
print("snapshot:", res.get("snapshot"))
