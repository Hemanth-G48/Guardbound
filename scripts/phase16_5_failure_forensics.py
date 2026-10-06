"""Phase 16.5 §9-§12: raw attacker-output failure forensics (read-only)."""
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path("C:/Users/CSLAB/Desktop/hemanth/Guardbound")
OUT = ROOT / "results/phase16_5_forensics"
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = ("generatedQuestion", "lastResponseSummary")
REASONING = ("<think", "</think", "<thinking")


def extract_json(text: str):
    """Observation-only copy of the frozen structural extraction (no repair)."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def classify(raw: str) -> tuple[str, dict]:
    """§10 taxonomy. Exactly one primary category. Observation only."""
    info = {"chars": len(raw or ""), "ends_with_brace": False,
            "has_fence": False, "has_pretty_prose": False,
            "balanced": False, "thinking": False}
    t = (raw or "").strip()
    if not t:
        return "EMPTY_OUTPUT", info
    info["thinking"] = any(m in t.lower() for m in REASONING)
    info["has_fence"] = t.startswith("```") or "\n```" in t
    first = t.find("{")
    info["has_pretty_prose"] = bool(first > 0 and t[:first].strip())
    info["balanced"] = t.endswith("}")
    info["ends_with_brace"] = t.endswith("}")

    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        missing = [k for k in REQUIRED if k not in obj]
        if not missing:
            bad_type = [k for k in REQUIRED if not isinstance(obj[k], str)]
            return ("WRONG_FIELD_TYPE" if bad_type else "VALID_JSON"), info
        near = [k for k in obj if k.strip() in REQUIRED]
        return ("WRONG_KEY" if near else "MISSING_REQUIRED_FIELD"), info

    if info["thinking"] and "{" not in t:
        return "REASONING_ONLY", info
    if not info["balanced"] and "{" in t:
        return "TRUNCATED_JSON", info
    if re.search(r",\s*\}", t):
        return "TRAILING_COMMA", info
    if info["has_fence"]:
        return "MARKDOWN_FENCED_JSON", info
    if info["has_pretty_prose"]:
        return "EXTRA_TEXT", info
    if "{" not in t:
        return "NATURAL_LANGUAGE", info
    if extract_json(t) is not None:
        return "OTHER", info
    return "MALFORMED_SYNTAX", info


rows = []

# ---- Track A (the real attack interface) ------------------------------------
P = (ROOT / "results/phase15_pilot_30/trackA_frozen/"
     "batch00_goals000-029_180records.jsonl")
recs = [json.loads(l) for l in P.read_text(encoding="utf-8").splitlines() if l.strip()]
coverage = {"runs": 0, "calls_declared": 0, "calls_captured": 0}
for r in recs:
    declared = (r.get("attacker_purpose_calls") or {}).get("generation", 0) or 0
    raws = [x for x in (r.get("attacker_raw_outputs") or [])
            if x.get("purpose") == "generation"]
    coverage["runs"] += 1
    coverage["calls_declared"] += declared
    coverage["calls_captured"] += len(raws)
    for i, entry in enumerate(raws):
        raw = entry.get("raw", "")
        cls, info = classify(raw)
        rows.append({"source": "phase15_trackA", "attack": r["attack"],
                     "goal_id": r["goal_id"], "run_id": r["run_id"],
                     "turn_index": i, "seq": r.get("seed"),
                     "condition": r["condition"], "class": cls,
                     "raw_kind": entry.get("type"), "raw": raw, **info})

# ---- Phase 16.1 (Ornith, 4 configurations) ---------------------------------
for cfg in ("A_thinking_off_constrained", "B_thinking_off_unconstrained",
            "C_thinking_on_constrained", "D_thinking_on_unconstrained"):
    for f, stage in (("stage_a_raw.jsonl", "A"), ("stage_b_raw.jsonl", "B")):
        p = ROOT / f"results/phase16_1_ornith_diagnostic/config_{cfg}/{f}"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            cls, info = classify(d.get("raw", ""))
            rows.append({"source": f"phase16_1_{cfg}", "attack": "crescendo_paper",
                         "goal_id": d.get("goal_id"), "run_id": None,
                         "turn_index": d.get("turn", d.get("i")),
                         "condition": "n/a", "class": cls,
                         "raw_kind": d.get("raw_kind"), "raw": d.get("raw", ""),
                         "stage": stage, **info})

# ---- Phase 16.2 (Ornith config B validation) -------------------------------
for f, stage in (("stage_a.jsonl", "A"), ("stage_b.jsonl", "B")):
    p = ROOT / f"results/phase16_2_ornith_config_b/{f}"
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        cls, info = classify(d.get("raw_output", ""))
        rows.append({"source": "phase16_2_configB", "attack": "crescendo_paper",
                     "goal_id": d.get("goal_id"), "run_id": None,
                     "turn_index": d.get("turn", d.get("i")),
                     "condition": "n/a", "class": cls,
                     "raw_kind": d.get("raw_kind"), "raw": d.get("raw_output", ""),
                     "stage": stage, **info})

with open(OUT / "failure_forensics.jsonl", "w", encoding="utf-8") as fh:
    for row in rows:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")

# ---- summary ---------------------------------------------------------------
print("=" * 80)
print("COVERAGE")
print("=" * 80)
print(f"Track A: {coverage['calls_captured']}/{coverage['calls_declared']} attacker "
      f"generations captured ({100*coverage['calls_captured']/coverage['calls_declared']:.1f}%)")
print(f"total forensic rows: {len(rows)}")


def block(rs, label):
    c = Counter(r["class"] for r in rs)
    valid = c.get("VALID_JSON", 0)
    failed = len(rs) - valid
    print(f"\n{label}: n={len(rs)} valid={valid} failed={failed} "
          f"({100*failed/max(1,len(rs)):.2f}%)")
    for k, v in c.most_common():
        if k != "VALID_JSON":
            print(f"    {k:<24} {v:>4}  ({100*v/max(1,len(rs)):.2f}%)")


tracka = [r for r in rows if r["source"] == "phase15_trackA"]
block(tracka, "TRACK A (Qwen3-4B attacker, frozen study)")
for atk in ("crescendo_paper", "opposite_day", "acronym"):
    block([r for r in tracka if r["attack"] == atk], f"  {atk}")
print()
block([r for r in rows if r["source"].startswith("phase16_1")], "PHASE 16.1 (Ornith, 4 cfg)")
block([r for r in rows if r["source"] == "phase16_2_configB"], "PHASE 16.2 (Ornith, cfg B)")

print()
print("=" * 80)
print("§11 CONCENTRATION (Track A failures only)")
print("=" * 80)
fails = [r for r in tracka if r["class"] != "VALID_JSON"]
per_goal = Counter(r["goal_id"] for r in fails)
per_atk = Counter(r["attack"] for r in fails)
print(f"failures: {len(fails)} across {len(per_goal)} distinct goals of 30")
print(f"by attack: {dict(per_atk)}")
print(f"goals with >=1 failure: {sorted(per_goal)}")
print(f"max failures on one goal: {per_goal.most_common(3)}")
run_ids = {r["run_id"] for r in fails}
print(f"runs containing >=1 failure: {len(run_ids)} of 180")

print()
print("=" * 80)
print("§12 TOKEN-BUDGET TEST (needs the attacker tokenizer)")
print("=" * 80)

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
tok_cache: dict = {}


def ntok(s: str) -> int:
    if s not in tok_cache:
        tok_cache[s] = len(tok.encode(s, add_special_tokens=False))
    return tok_cache[s]


for label, rs in (("Track A failures", fails),
                  ("Track A valid", [r for r in tracka if r["class"] == "VALID_JSON"])):
    if not rs:
        continue
    counts = [ntok(r["raw"]) for r in rs]
    at_cap = sum(1 for c in counts if c >= 250)
    print(f"{label}: n={len(rs)} token mean={sum(counts)/len(counts):.1f} "
          f"median={sorted(counts)[len(counts)//2]} max={max(counts)} "
          f"at/over 250 tok={at_cap} ({100*at_cap/len(rs):.0f}%)")

print("\nper-class token stats (Track A failures):")
for cls in sorted({r["class"] for r in fails}):
    rs = [r for r in fails if r["class"] == cls]
    counts = [ntok(r["raw"]) for r in rs]
    at_cap = sum(1 for c in counts if c >= 250)
    enders = Counter(r["raw"].strip()[-1:] for r in rs)
    print(f"  {cls:<22} n={len(rs):<3} median_tok={sorted(counts)[len(counts)//2]:<4} "
          f"at_cap={at_cap}/{len(rs)}  last_chars={dict(enders.most_common(3))}")

# §12 verdict per failure
verdicts = Counter()
for r in fails:
    n = ntok(r["raw"])
    if n >= 250 and r["class"] == "TRUNCATED_JSON":
        verdicts["LIKELY_TOKEN_BUDGET_FAILURE"] += 1
    elif n >= 250:
        verdicts["LIKELY (at cap but parseable shape)"] += 1
    elif r["class"] == "TRUNCATED_JSON":
        verdicts["AT_CAP_OR_UNBALANCED"] += 1
    else:
        verdicts["NOT_TOKEN_BUDGET_FAILURE"] += 1
print(f"\n§12 verdicts (Track A): {dict(verdicts)}")

summary = {
    "coverage": coverage,
    "total_rows": len(rows),
    "sources": {},
    "track_a": {
        "by_attack": {},
        "failure_concentration": {
            "distinct_goals_with_failure": len(per_goal),
            "goals": sorted(per_goal),
            "max_on_one_goal": per_goal.most_common(3),
            "runs_with_any_failure": len(run_ids),
        },
        "token_budget_verdicts": dict(verdicts),
    },
}
for src in sorted({r["source"] for r in rows}):
    rs = [r for r in rows if r["source"] == src]
    c = Counter(r["class"] for r in rs)
    summary["sources"][src] = {"n": len(rs), "classes": dict(c),
                               "failure_rate": round(
                                   (len(rs) - c.get("VALID_JSON", 0)) / len(rs), 4)}
for atk in ("crescendo_paper", "opposite_day", "acronym"):
    rs = [r for r in tracka if r["attack"] == atk]
    c = Counter(r["class"] for r in rs)
    summary["track_a"]["by_attack"][atk] = {
        "n": len(rs), "classes": dict(c),
        "failure_rate": round((len(rs) - c.get("VALID_JSON", 0)) / len(rs), 4)}
(OUT / "failure_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote {OUT/'failure_forensics.jsonl'} and failure_summary.json")
