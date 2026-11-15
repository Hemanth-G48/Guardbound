import json
from collections import Counter, defaultdict
from pathlib import Path

A = [json.loads(l) for l in Path("results/phase14/phase14_3_attacker_qualification.jsonl")
     .read_text(encoding="utf-8").splitlines() if l.strip()]
E = [json.loads(l) for l in Path("results/phase14/phase14_3_evaluator_qualification.jsonl")
     .read_text(encoding="utf-8").splitlines() if l.strip()]

print("=" * 92)
print("ATTACKER — split by stage (the 0.5 'usable_output_rate' explained)")
print("=" * 92)
for stage in ("round1", "round2"):
    s = [r for r in A if r["stage"] == stage]
    print(f"  {stage}: n={len(s)} valid_json={sum(r['valid_json'] for r in s)} "
          f"accepted={sum(r['parser_accepted'] for r in s)} "
          f"q_usable={sum(r['generated_question_usable'] for r in s)} "
          f"summary_usable={sum(r['last_response_summary_usable'] for r in s)}")
print("  NOTE: round1 has no prior response, so an empty lastResponseSummary is")
print("        correct there and is NOT required by the round1 contract.")

print()
print("  usable_output_rate over round2 only (both fields required):")
r2 = [r for r in A if r["stage"] == "round2"]
print("   ", round(sum(1 for r in r2 if r["generated_question_usable"]
                       and r["last_response_summary_usable"]) / len(r2), 4))
print("  question length stats (round2):",
      min(r["question_chars"] for r in r2), "-",
      max(r["question_chars"] for r in r2))
print("  summary  length stats (round2):",
      min(r["summary_chars"] for r in r2), "-",
      max(r["summary_chars"] for r in r2))

print()
print("=" * 92)
print("EVALUATOR — failures by contract and case kind")
print("=" * 92)
fails = [r for r in E if not r["parser_accepted"]]
print("total failures:", len(fails))
print("by contract  :", dict(Counter(r["contract"].split(".")[-1] for r in fails)))
print("by case      :", dict(Counter(r["case_id"] for r in fails)))
print("by raw_type  :", dict(Counter(r["raw_type"] for r in fails)))
print("parser crashes:", dict(Counter(r["contract"].split(".")[-1]
                                      for r in E if r.get("parser_crash"))))

print()
print("acceptance by case kind (refusal-family contracts only):")
ref = [r for r in E if "check_refusal" in r["contract"] or "disclaimer" in r["contract"]]
for kind in ("target_response", "refusal_style"):
    k = [r for r in ref if r.get("case_kind") == kind]
    acc = sum(1 for r in k if r["parser_accepted"])
    print(f"  {kind:<16} {acc}/{len(k)} = {acc/len(k):.3f}")

print()
print("acceptance by case kind (rubric contracts only):")
rub = [r for r in E if "rubric" in r["contract"]]
for kind in ("target_response", "refusal_style"):
    k = [r for r in rub if r.get("case_kind") == kind]
    acc = sum(1 for r in k if r["parser_accepted"])
    print(f"  {kind:<16} {acc}/{len(k)} = {acc/len(k):.3f}")

print()
print("=== sample invalid refusal outputs (truncated) ===")
shown = 0
for r in fails:
    if "check_refusal" in r["contract"] and shown < 3:
        raw = (r["raw"] or "")
        print(f"\n  contract={r['contract'].split('.')[-1]} case={r['case_id']} "
              f"len={len(raw)}")
        print("   ", repr(raw[:340]))
        shown += 1

print()
print("=== the 5 parser crashes (check_disclaimer on a str) ===")
for r in E:
    if r.get("parser_crash"):
        print(f"  {r['case_id']:<9} raw_type={r['raw_type']:<5} "
              f"valid_json={r['valid_json']}  {r['parser_crash'][:90]}")

print()
print("=" * 92)
print("EVALUATOR — is the failure mode the same as the attacker disqualification?")
print("=" * 92)
print("invalid-JSON rate on refusal-family :",
      f"{1 - (len(ref) - sum(1 for r in ref if r['valid_json']))/len(ref):.3f}")
print("invalid-JSON rate on rubric-family  :",
      f"{1 - (len(rub) - sum(1 for r in rub if r['valid_json']))/len(rub):.3f}")

print()
print("=== rubric score distribution (sanity: did the model really score?) ===")
scores = Counter()
for r in rub:
    obj = None
    try:
        obj = json.loads(r["raw"]) if r["raw_type"] == "str" else r["raw"]
        if isinstance(obj, dict):
            obj = json.loads(obj) if isinstance(obj, str) else obj
    except Exception:
        pass
print("  parser_return values:", dict(Counter(r.get("parser_return") for r in rub)))
