import json
from pathlib import Path

for p in ["results/phase13/nbf_off.jsonl", "results/phase13/nbf_on.jsonl"]:
    f = Path(p)
    print("===", p, "(exists=%s)" % f.exists())
    if not f.exists():
        continue
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        print("  attack=%-16s goal=%-5s succ=%-6s turns=%-4s rt=%-8s nbf=%-3s filt=%-3s calls=%s" % (
            r.get("attack"),
            r.get("goal_idx", r.get("goal_id", r.get("goal"))),
            r.get("success"),
            r.get("num_turns"),
            round(r.get("runtime", 0) or 0, 1),
            len(r.get("nbf_scores") or []),
            r.get("filtered_queries"),
            r.get("llm_calls"),
        ))
