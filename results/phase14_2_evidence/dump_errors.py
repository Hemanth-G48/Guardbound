import json
from pathlib import Path

for line in Path("results/phase14/batch00.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    print("=" * 88)
    print(r.get("run_id"), "|", r.get("failure_class"), "|", r.get("error_type"))
    print("  error_message:", r.get("error_message"))
    res = r.get("residency") or {}
    print("  loads=%s evicts=%s refusals=%s load_s=%s" % (
        res.get("loads"), res.get("evictions"), res.get("load_refusals"),
        res.get("load_seconds")))
    for rot in res.get("rotations") or []:
        print("    rot:", rot)
    print("  snapshot:", res.get("snapshot"))
    print("  system:", r.get("system"))
