import json
from pathlib import Path

d = json.loads(Path("results/phase14_6_rubric_semantic_sanity.json").read_text(encoding="utf-8"))
for r in d["records"]:
    print(f"--- {r['probe']} / {r['case_id']}  (kind={r['kind']})")
    print(f"    score_returned : {r['score_returned']}")
    print(f"    raw_type       : {r['raw_type']}")
    print(f"    raw            : {r['raw']}")
    print(f"    latency        : {r['elapsed_s']}s   prompt_chars={r.get('prompt_chars')} "
          f"messages={r.get('messages')}")
    print(f"    flags          : {[k for k, v in r['failure_checks'].items() if v] or 'none'}")
    print(f"    call_error     : {r['call_error']}")
    if r["probe"] == "probe2":
        print(f"    source         : {r.get('source')}  run_id={r['case_id']}")
        print(f"    goal           : {r['goal'][:110]}")
        print(f"    response[:150] : {r['response'][:150]!r}")
    print()
print("=== call-shape uniformity ===")
print("  mode_seen_by_recorder:", {r["case_id"]: r["mode_seen_by_recorder"] for r in d["records"]})
print("  temperature_seen     :", {r["temperature_seen"] for r in d["records"]})
print("  json_format_seen     :", {r["json_format_seen"] for r in d["records"]})
print("  prompt_chars range   :", min(r["prompt_chars"] for r in d["records"]),
      "-", max(r["prompt_chars"] for r in d["records"]))
print()
lat = sorted(r["elapsed_s"] for r in d["records"])
print("latency: n=%d mean=%.2fs median=%.2fs max=%.2fs" % (
    len(lat), sum(lat) / len(lat), lat[len(lat) // 2], max(lat)))
