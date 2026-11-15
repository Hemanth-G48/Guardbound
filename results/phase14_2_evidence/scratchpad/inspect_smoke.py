import json
from pathlib import Path

p = Path("results/phase14/batch00.jsonl")
for line in p.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    print("run_id        :", r.get("run_id"))
    print("attack        :", r.get("attack"), "| condition:", r.get("condition"))
    print("success       :", r.get("success"))
    print("failure_class :", r.get("failure_class"))
    print("error_type    :", r.get("error_type"))
    print("error_message :", str(r.get("error_message"))[:600])
    print("num_turns     :", r.get("num_turns"), "| runtime:", r.get("runtime"))
    print("nbf_scores    :", len(r.get("nbf_scores") or []), "| filtered:", r.get("filtered_queries"))
    print("llm_calls     :", r.get("llm_calls"))
    print("peak_vram_gb  :", r.get("peak_vram_gb"))
    print("attacker_model:", r.get("attacker_model"))
    print("target_model  :", r.get("target_model"))
    print("evaluator_model:", r.get("evaluator_model"))
    res = r.get("residency") or {}
    print("residency     : snapshot=%s" % (res.get("snapshot"),))
    print("  loads=%s evictions=%s load_s=%s evict_s=%s refusals=%s" % (
        res.get("loads"), res.get("evictions"), res.get("load_seconds"),
        res.get("eviction_seconds"), res.get("load_refusals")))
    for rot in res.get("rotations") or []:
        print("    rot:", rot)
    print("system        :", r.get("system"))
    print("num_turns/turns:", [(t.get("query") or "")[:60] for t in (r.get("turns") or [])])
