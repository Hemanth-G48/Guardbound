"""Phase 17 Stage 5 — attacker reliability investigation: analysis.

Reads, read-only:
  * the frozen Stage 4.8 run records (371 real attacker calls)
  * the Stage 5 controlled experiments under `raw/`
  * the bounded real-trajectory replay, if present

Writes the Stage 5 artifact set under
`results/phase17_pilot/analysis/stage5_attacker_reliability/`.

Nothing here modifies the pilot data or the production configuration; every failed
generation stays a failed generation, and no raw value is replaced by a summary.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "results" / "phase17_pilot"
STAGE5 = PILOT / "analysis" / "stage5_attacker_reliability"
RAW = STAGE5 / "raw"
FIG = STAGE5 / "figures"
PILOT_RUNS = PILOT / "runs" / "raw_results.jsonl"
REPLAY_RUNS = STAGE5 / "replay" / "runs" / "raw_results.jsonl"

VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")
ATTACKS = ("crescendo_paper", "opposite_day", "acronym")
LABEL = {"crescendo_paper": "Crescendo", "opposite_day": "OppositeDay", "acronym": "Acronym"}
TESTS: list[dict] = []


# --------------------------------------------------------------------------- #
# stats                                                                        #
# --------------------------------------------------------------------------- #
def wilson(s: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = s / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return [round((centre - margin) / denom, 4), round((centre + margin) / denom, 4)]


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return math.exp(_log_comb(r1, x) + _log_comb(n - r1, c1 - x) - _log_comb(n, c1))

    obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return round(min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-12)), 6)


def test(name: str, a: int, b: int, c: int, d: int, note: str = "") -> dict:
    p = fisher(a, b, c, d)
    TESTS.append({"question": name, "p_value": p})
    pa, pb = a / (a + b) if a + b else None, c / (c + d) if c + d else None
    return {"question": name, "table": [[a, b], [c, d]], "p_value": p,
            "rate_row1": round(pa, 4) if pa is not None else None,
            "rate_row2": round(pb, 4) if pb is not None else None,
            "risk_difference": (round(pa - pb, 4) if pa is not None and pb is not None else None),
            "note": note}


def rate(calls: list[dict], key: str = "json_valid") -> dict:
    n = len(calls)
    s = sum(1 for c in calls if c.get(key))
    return {"numerator": s, "denominator": n, "rate": round(s / n, 4) if n else None,
            "ci95_wilson": wilson(s, n)}


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(values), "mean": round(statistics.mean(values), 1),
            "median": round(statistics.median(values), 1),
            "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
            "min": round(min(values), 1), "max": round(max(values), 1)}


# --------------------------------------------------------------------------- #
# §3 instrumentation of the real pilot calls                                   #
# --------------------------------------------------------------------------- #
def instrument_real_calls() -> list[dict]:
    runs = [json.loads(line) for line in PILOT_RUNS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows: list[dict] = []
    for run in runs:
        cumulative_attacker = 0
        previous_validity = None
        previous_length = None
        for index, call in enumerate(run["attacker_calls"]):
            stats = call
            raw = call.get("raw_output")
            cumulative_attacker += (call.get("generated_tokens") or 0)
            rows.append({
                "run_id": run["run_id"], "attack": run["attack"],
                "goal_id": run["goal_id"], "seed": run["seed"],
                "call_index": index + 1,
                # not recoverable from the frozen records: the pilot stored attacker and
                # target calls in separate lists without a turn mapping
                "turn_index": None,
                "input_message_count": None,
                "input_token_count": call.get("prompt_tokens"),
                "input_token_count_estimate": (round(call["prompt_chars"] / 4, 1)
                                               if call.get("prompt_chars") else None),
                "previous_attacker_validity": previous_validity,
                "previous_attacker_output_length": previous_length,
                "previous_target_output_length": None,
                "cumulative_attacker_tokens": cumulative_attacker,
                "cumulative_target_tokens": None,
                "run_total_target_tokens": sum(
                    c.get("generated_tokens") or 0 for c in run["target_calls"]),
                "cumulative_context_tokens": call.get("prompt_tokens"),
                "attacker_output_tokens": call.get("generated_tokens"),
                "attacker_output_characters": len(raw) if isinstance(raw, str) else None,
                "json_valid": call["category"] in VALID,
                "semantic_valid": (call["category"] in VALID
                                   and call.get("empty_query") is not True),
                "failure_type": call.get("failure_type"),
                "category": call.get("category"),
                "generated_question_present": call.get("has_generatedQuestion"),
                "generated_question_length": (len(call["generated_question"])
                                              if isinstance(call.get("generated_question"), str)
                                              else None),
                "temperature": call.get("temperature"),
                "top_p": run["top_p"],
                "prompt_hash": call.get("prompt_hash") or run.get("prompt_hash"),
                "model_revision": run["attacker_revision"],
                "raw_output": raw,
                "source": "stage4.8_pilot",
            })
            previous_validity = call["category"] in VALID and call.get("empty_query") is not True
            previous_length = len(raw) if isinstance(raw, str) else None
    return rows


def controlled_rows() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    if not RAW.is_dir():
        return out
    for path in sorted(RAW.glob("*.jsonl")):
        out[path.stem] = [json.loads(line) for line
                          in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


# --------------------------------------------------------------------------- #
# §11 critical table                                                           #
# --------------------------------------------------------------------------- #
def call_index_table(rows: list[dict]) -> dict:
    table: dict[str, dict] = {}
    buckets = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 99)]
    for lo, hi in buckets:
        key = f"{lo}+" if hi == 99 else str(lo)
        subset = [c for c in rows if lo <= c["call_index"] <= hi]
        if not subset:
            continue
        contexts = [c["input_token_count"] for c in subset
                    if c.get("input_token_count") is not None]
        table[key] = {
            "calls": len(subset),
            "json_valid": sum(1 for c in subset if c["json_valid"]),
            "validity_rate": round(sum(1 for c in subset if c["json_valid"]) / len(subset), 4),
            "ci95_wilson": wilson(sum(1 for c in subset if c["json_valid"]), len(subset)),
            "mean_context_tokens": round(statistics.mean(contexts), 1) if contexts else None,
            "prose_failures": sum(1 for c in subset
                                  if c["failure_type"] == "prose_output"),
            "empty_question_failures": sum(1 for c in subset
                                           if c["failure_type"] == "empty_generatedQuestion"),
        }
    return table


def context_bucket_table(rows: list[dict], buckets: list[tuple[int, int]]) -> dict:
    table: dict[str, dict] = {}
    for lo, hi in buckets:
        subset = [c for c in rows if c.get("input_token_count") is not None
                  and lo <= c["input_token_count"] < hi]
        if not subset:
            continue
        table[f"{lo}-{hi}"] = {
            "label": f"{lo}-{hi} tokens",
            "calls": len(subset),
            "json_valid": sum(1 for c in subset if c["json_valid"]),
            "validity_rate": round(sum(1 for c in subset if c["json_valid"]) / len(subset), 4),
            "ci95_wilson": wilson(sum(1 for c in subset if c["json_valid"]), len(subset)),
            "prose_failures": sum(1 for c in subset if c["failure_type"] == "prose_output"),
            "prose_rate": round(sum(1 for c in subset
                                    if c["failure_type"] == "prose_output") / len(subset), 4),
            "mean_output_tokens": round(statistics.mean(
                [c["attacker_output_tokens"] for c in subset
                 if c.get("attacker_output_tokens") is not None]), 1)
            if any(c.get("attacker_output_tokens") is not None for c in subset) else None,
        }
    return table


# --------------------------------------------------------------------------- #
# §9 generation-level forensics                                                #
# --------------------------------------------------------------------------- #
REFUSAL_PREFIXES = ("i can't", "i cannot", "i can not", "i'm sorry", "i am sorry",
                    "i won't", "i will not", "sorry", "i'm not able", "i am not able",
                    "i must decline", "i'd rather not")


def generation_shape(text) -> str:
    """Classify WHERE a failed generation went wrong, from its own text only.

    Documentary rules, applied to the raw sequence:
      empty           -> no output at all
      reasoning_text  -> explicit reasoning markers before any object
      json_like_prefix-> the response opens an object and then stops being JSON
                         (a truncation if it never closes, otherwise malformed)
      refusal         -> opens with an unambiguous refusal/disclaimer phrase
      immediate_prose -> natural-language continuation with no object at all
    """
    if text is None:
        return "no_output"
    stripped = text.strip()
    if not stripped:
        return "empty"
    lowered = stripped.lower()
    if "<think" in lowered:
        return "reasoning_text"
    if stripped.startswith("{"):
        closes = stripped.count("{") <= stripped.count("}")
        return "json_like_prefix_closed" if closes else "json_like_prefix_unterminated"
    if any(lowered.startswith(p) for p in REFUSAL_PREFIXES):
        return "refusal_or_disclaimer"
    return "immediate_prose"


def forensics(rows: list[dict]) -> dict:
    failures = [c for c in rows if not c["json_valid"] or c["failure_type"]]
    shapes = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for call in failures:
        shape = generation_shape(call.get("raw_output"))
        shapes[shape] += 1
        if len(examples[shape]) < 3:
            head = (call.get("raw_output") or "")[:120]
            examples[shape].append({"run_id": call.get("run_id", call.get("condition")),
                                    "call_index": call["call_index"],
                                    "failure_type": call["failure_type"],
                                    "raw_head": head})
    return {
        "phase": "17", "stage": "5", "deliverable": "failure_taxonomy.json",
        "total_failed_calls": len(failures),
        "failure_type_counts": dict(Counter(c["failure_type"] for c in failures)),
        "generation_shape_counts": dict(shapes),
        "generation_shape_by_failure_type": {
            ft: dict(Counter(generation_shape(c.get("raw_output"))
                             for c in failures if c["failure_type"] == ft))
            for ft in sorted({c["failure_type"] for c in failures})},
        "examples": {k: v for k, v in examples.items()},
        "rules": {
            "reasoning_text": "explicit reasoning markers present",
            "json_like_prefix_unterminated": "object opened, never closed (truncated)",
            "json_like_prefix_closed": "object opened and closed but not parseable",
            "refusal_or_disclaimer": "opens with an unambiguous refusal phrase",
            "immediate_prose": "natural-language continuation with no object at all",
            "empty": "no characters produced",
        },
        "note": "shape is read from the raw text only; nothing is repaired or converted, "
                "and the stored failure_type remains the authoritative classification",
    }


def transition_analysis(rows: list[dict]) -> dict:
    """Transitions between consecutive calls inside a run.

    Structural caveat: in the frozen pipeline an invalid attacker call ends the run, so
    'invalid -> valid' and 'invalid -> invalid' are unobservable by construction. What can
    be measured is the valid->valid rate and the valid->invalid event, plus the association
    between the *previous* call's properties and the next call's validity.
    """
    by_run: dict[str, list[dict]] = defaultdict(list)
    for call in rows:
        if call.get("source") == "stage4.8_pilot":
            by_run[call["run_id"]].append(call)
    transitions = Counter()
    prev_valid_next_valid = 0
    prev_valid_next_invalid = 0
    prev_len_short_next_valid = prev_len_short_next_invalid = 0
    prev_len_long_next_valid = prev_len_long_next_invalid = 0
    lengths = [c["previous_attacker_output_length"] for c in rows
               if c.get("previous_attacker_output_length")]
    median_len = statistics.median(lengths) if lengths else None
    for run_id, calls in by_run.items():
        calls.sort(key=lambda c: c["call_index"])
        for prev, nxt in zip(calls, calls[1:]):
            if prev["json_valid"]:
                if nxt["json_valid"]:
                    transitions["valid->valid"] += 1
                    prev_valid_next_valid += 1
                else:
                    transitions["valid->invalid"] += 1
                    prev_valid_next_invalid += 1
                prev_len = prev.get("attacker_output_characters")
                if prev_len is not None and median_len:
                    if prev_len <= median_len:
                        if nxt["json_valid"]:
                            prev_len_short_next_valid += 1
                        else:
                            prev_len_short_next_invalid += 1
                    else:
                        if nxt["json_valid"]:
                            prev_len_long_next_valid += 1
                        else:
                            prev_len_long_next_invalid += 1
            else:
                transitions["invalid->* (unobservable: the run ended)"] += 0
    result = {
        "phase": "17", "stage": "5", "deliverable": "transition_analysis.json",
        "source": "frozen Stage 4.8 records (371 calls, 90 runs)",
        "observed_transitions": dict(transitions),
        "structural_note": "an invalid attacker call terminates its run in the frozen "
                           "pipeline, so invalid->valid and invalid->invalid transitions "
                           "cannot occur and are not reported as if they had been observed",
        "valid_to_valid_rate": (round(prev_valid_next_valid
                                      / (prev_valid_next_valid + prev_valid_next_invalid), 4)
                                if (prev_valid_next_valid + prev_valid_next_invalid) else None),
        "previous_output_length_median_chars": median_len,
        "previous_output_length_vs_next_validity": {
            "previous_short_valid_rate": (round(prev_len_short_next_valid
                                                / (prev_len_short_next_valid
                                                   + prev_len_short_next_invalid), 4)
                                          if (prev_len_short_next_valid
                                              + prev_len_short_next_invalid) else None),
            "previous_long_valid_rate": (round(prev_len_long_next_valid
                                               / (prev_len_long_next_valid
                                                  + prev_len_long_next_invalid), 4)
                                         if (prev_len_long_next_valid
                                             + prev_len_long_next_invalid) else None),
            "n_short_pairs": prev_len_short_next_valid + prev_len_short_next_invalid,
            "n_long_pairs": prev_len_long_next_valid + prev_len_long_next_invalid,
            "test": test("previous attacker output length (median split) vs next-call validity",
                         prev_len_short_next_valid, prev_len_short_next_invalid,
                         prev_len_long_next_valid, prev_len_long_next_invalid),
        },
    }
    return result


# --------------------------------------------------------------------------- #
# controlled-experiment summaries                                              #
# --------------------------------------------------------------------------- #
def controlled_summary(rows: list[dict], key_field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for call in rows:
        groups[str(call.get(key_field))].append(call)
    out = {}
    for condition, subset in sorted(groups.items()):
        contexts = [c["input_token_count"] for c in subset
                    if c.get("input_token_count") is not None]
        out[condition] = {
            "calls": len(subset),
            "json_valid": sum(1 for c in subset if c["json_valid"]),
            "validity_rate": round(sum(1 for c in subset if c["json_valid"]) / len(subset), 4),
            "ci95_wilson": wilson(sum(1 for c in subset if c["json_valid"]), len(subset)),
            "semantic_valid": sum(1 for c in subset if c["semantic_valid"]),
            "prose_failures": sum(1 for c in subset
                                  if c["failure_type"] == "prose_output"),
            "empty_question_failures": sum(1 for c in subset
                                           if c["failure_type"] == "empty_generatedQuestion"),
            "mean_context_tokens": round(statistics.mean(contexts), 1) if contexts else None,
            "mean_output_tokens": round(statistics.mean(
                [c["attacker_output_tokens"] for c in subset
                 if c.get("attacker_output_tokens") is not None]), 1)
            if any(c.get("attacker_output_tokens") is not None for c in subset) else None,
            "by_attack": {LABEL[a]: rate([c for c in subset if c["attack"] == a])
                          for a in ATTACKS if any(c["attack"] == a for c in subset)},
        }
    return out


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #
def figures(real: list[dict], index_table: dict, ctx_table: dict,
            controlled: dict[str, list[dict]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def save(fig, name, title):
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=130)
        plt.close(fig)
        made.append(name)

    # Plot 1: JSON validity vs call index (real pilot)
    keys = [k for k in ("1", "2", "3", "4", "5+") if k in index_table]
    vals = [100 * index_table[k]["validity_rate"] for k in keys]
    ns = [index_table[k]["calls"] for k in keys]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(keys, vals, marker="o", color="#2e6f95")
    ax.set_ylim(0, 100)
    ax.set_xlabel("attacker call index within the run")
    ax.set_ylabel("JSON validity (%)")
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.annotate(f"{v:.0f}%\nn={n}", (i, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    save(fig, "plot1_validity_vs_call_index.png",
         "Plot 1 — JSON validity vs call index (frozen Stage 4.8 data, 371 calls)")

    # Plot 2: validity vs context length
    labels = list(ctx_table)
    vals = [100 * ctx_table[k]["validity_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(labels, vals, color="#4c9f70")
    ax.set_ylim(0, 100)
    ax.set_xlabel("context length (prompt tokens)")
    ax.set_ylabel("JSON validity (%)")
    for i, k in enumerate(labels):
        ax.text(i, vals[i], f"{vals[i]:.0f}%\nn={ctx_table[k]['calls']}", ha="center",
                va="bottom", fontsize=8)
    save(fig, "plot2_validity_vs_context_length.png",
         "Plot 2 — JSON validity vs context length (frozen Stage 4.8 data)")

    # Plot 3: prose failure probability vs context length
    vals = [100 * ctx_table[k]["prose_rate"] for k in labels]
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(labels, vals, color="#b5651d")
    ax.set_xlabel("context length (prompt tokens)")
    ax.set_ylabel("prose failure rate (%)")
    for i, k in enumerate(labels):
        ax.text(i, vals[i], f"{vals[i]:.0f}%", ha="center", va="bottom", fontsize=8)
    save(fig, "plot3_prose_rate_vs_context_length.png",
         "Plot 3 — prose failure probability vs context length (frozen Stage 4.8 data)")

    # Plot 4: previous output length -> next validity (real)
    lengths = [c["previous_attacker_output_length"] for c in real
               if c.get("previous_attacker_output_length")]
    if lengths:
        median = statistics.median(lengths)
        short = [c for c in real if c.get("previous_attacker_output_length")
                 and c["previous_attacker_output_length"] <= median]
        long_ = [c for c in real if c.get("previous_attacker_output_length")
                 and c["previous_attacker_output_length"] > median]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        vals = [100 * rate(short)["rate"], 100 * rate(long_)["rate"]]
        ax.bar([f"previous output\n<= {median:.0f} chars",
                f"previous output\n> {median:.0f} chars"], vals, color="#8d5a97")
        ax.set_ylim(0, 100)
        ax.set_ylabel("next-call JSON validity (%)")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom")
        save(fig, "plot4_previous_output_vs_next_validity.png",
             "Plot 4 — previous attacker output length vs next-call validity (real data)")

    # Plot 5: attacker output length vs context length (real)
    points = [(c["input_token_count"], c["attacker_output_tokens"])
              for c in real if c.get("input_token_count") and c.get("attacker_output_tokens")]
    if points:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        valid = [p for p, c in zip(points, real)
                 if c.get("input_token_count") and c.get("attacker_output_tokens")
                 and c["json_valid"]]
        invalid = [p for p, c in zip(points, real)
                   if c.get("input_token_count") and c.get("attacker_output_tokens")
                   and not c["json_valid"]]
        ax.scatter([p[0] for p in valid], [p[1] for p in valid], s=14, label="valid",
                   color="#2e6f95", alpha=0.7)
        ax.scatter([p[0] for p in invalid], [p[1] for p in invalid], s=14, label="invalid",
                   color="#c25b56", alpha=0.8)
        ax.set_xlabel("context length (prompt tokens)")
        ax.set_ylabel("attacker output tokens")
        ax.legend()
        save(fig, "plot5_output_length_vs_context.png",
             "Plot 5 — attacker output length vs context length (real data)")

    # controlled: context-length manipulation
    if "context_length" in controlled:
        summary = controlled_summary(controlled["context_length"], "condition")
        keys = sorted(summary, key=lambda k: summary[k]["mean_context_tokens"] or 0)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        xs = [summary[k]["mean_context_tokens"] for k in keys]
        ys = [100 * summary[k]["validity_rate"] for k in keys]
        ax.plot(xs, ys, marker="o", color="#3f7d8c")
        for x, y, k in zip(xs, ys, keys):
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_xlabel("mean context length (prompt tokens)")
        ax.set_ylabel("JSON validity (%)")
        save(fig, "plot6_controlled_depth_validity.png",
             "Controlled context-length experiment: validity vs context (depth C0-C5)")

    # controlled: instruction position, ablation, temperature
    for name, key, fname, title in (
            ("instruction_position", "condition", "plot7_instruction_position.png",
             "Instruction-position diagnostic (I0-I4)"),
            ("prev_output", "condition", "plot8_previous_output_ablation.png",
             "Previous-output ablation (P0-P4)"),
            ("temperature", "condition", "plot9_temperature.png",
             "Temperature diagnostic at fixed 4-turn context")):
        if name not in controlled:
            continue
        summary = controlled_summary(controlled[name], key)
        keys = sorted(summary)
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        vals = [100 * summary[k]["validity_rate"] for k in keys]
        ax.bar(keys, vals, color="#5b7fa6")
        ax.set_ylim(0, 100)
        ax.set_ylabel("JSON validity (%)")
        for i, k in enumerate(keys):
            ax.text(i, vals[i], f"{vals[i]:.0f}%\nn={summary[k]['calls']}", ha="center",
                    va="bottom", fontsize=8)
        save(fig, fname, title)

    return made


# --------------------------------------------------------------------------- #
def main() -> int:
    STAGE5.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    real = instrument_real_calls()
    with (STAGE5 / "call_level_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in real:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    controlled = controlled_rows()
    index_table = call_index_table(real)
    ctx_table = context_bucket_table(real, [(0, 1500), (1500, 2000), (2000, 2500),
                                            (2500, 3000), (3000, 100000)])

    # §11 critical table + §4-style bucket analysis of the real data
    context_payload = {
        "phase": "17", "stage": "5", "deliverable": "context_length_analysis.json",
        "source": "frozen Stage 4.8 records (371 attacker calls from 90 runs)",
        "call_index_table": index_table,
        "context_bucket_table": ctx_table,
        "first_vs_later": {
            "call_1": rate([c for c in real if c["call_index"] == 1]),
            "calls_2_plus": rate([c for c in real if c["call_index"] >= 2]),
            "test": test("call 1 vs later calls",
                         sum(1 for c in real if c["call_index"] == 1 and c["json_valid"]),
                         sum(1 for c in real if c["call_index"] == 1 and not c["json_valid"]),
                         sum(1 for c in real if c["call_index"] >= 2 and c["json_valid"]),
                         sum(1 for c in real if c["call_index"] >= 2 and not c["json_valid"])),
            "mean_context_tokens_call_1": describe(
                [c["input_token_count"] for c in real
                 if c["call_index"] == 1 and c.get("input_token_count")]),
            "mean_context_tokens_later": describe(
                [c["input_token_count"] for c in real
                 if c["call_index"] >= 2 and c.get("input_token_count")]),
        },
        "within_later_calls_context_split": None,
        "controlled_context_length_experiment": (
            controlled_summary(controlled["context_length"], "condition")
            if "context_length" in controlled else None),
    }
    later = [c for c in real if c["call_index"] >= 2 and c.get("input_token_count")]
    if later:
        median_ctx = statistics.median(c["input_token_count"] for c in later)
        low = [c for c in later if c["input_token_count"] <= median_ctx]
        high = [c for c in later if c["input_token_count"] > median_ctx]
        context_payload["within_later_calls_context_split"] = {
            "median_context_tokens_of_later_calls": median_ctx,
            "lower_half": rate(low), "upper_half": rate(high),
            "test": test("context length (median split, later calls only) vs validity",
                         sum(1 for c in low if c["json_valid"]),
                         sum(1 for c in low if not c["json_valid"]),
                         sum(1 for c in high if c["json_valid"]),
                         sum(1 for c in high if not c["json_valid"]),
                         note="controls for call index by restricting to calls >= 2"),
        }
    (STAGE5 / "context_length_analysis.json").write_text(
        json.dumps(context_payload, indent=2), encoding="utf-8")

    (STAGE5 / "transition_analysis.json").write_text(
        json.dumps(transition_analysis(real), indent=2), encoding="utf-8")

    (STAGE5 / "failure_taxonomy.json").write_text(
        json.dumps(forensics(real), indent=2), encoding="utf-8")

    # attack comparison, with context controlled
    attack_payload = {
        "phase": "17", "stage": "5", "deliverable": "attack_comparison.json",
        "framing": "observed measurements; no attack is ranked, and the per-attack "
                   "context profiles are the explanatory variable of interest",
        "attacks": {},
    }
    for attack in ATTACKS:
        subset = [c for c in real if c["attack"] == attack]
        later_subset = [c for c in subset if c["call_index"] >= 2]
        attack_payload["attacks"][LABEL[attack]] = {
            "calls": len(subset),
            "validity": rate(subset),
            "later_call_validity": rate(later_subset),
            "context_tokens": describe([c["input_token_count"] for c in subset
                                        if c.get("input_token_count")]),
            "output_tokens": describe([c["attacker_output_tokens"] for c in subset
                                       if c.get("attacker_output_tokens") is not None]),
            "prose_failures": sum(1 for c in subset
                                  if c["failure_type"] == "prose_output"),
            "empty_question_failures": sum(1 for c in subset
                                           if c["failure_type"] == "empty_generatedQuestion"),
            "mean_call_depth_reached": None,
        }
    # context-matched comparison: restrict to the overlapping context band
    band_low, band_high = 1500, 2200
    banded = {LABEL[a]: rate([c for c in real if c["attack"] == a
                              and c.get("input_token_count")
                              and band_low <= c["input_token_count"] < band_high])
              for a in ATTACKS}
    attack_payload["context_matched_band"] = {
        "band_tokens": [band_low, band_high],
        "per_attack_validity": banded,
        "note": "if attack differences persist inside a matched context band, they are not "
                "explained by context length alone; with these sample sizes this is "
                "descriptive",
    }
    (STAGE5 / "attack_comparison.json").write_text(json.dumps(attack_payload, indent=2),
                                                   encoding="utf-8")

    (STAGE5 / "temperature_analysis.json").write_text(json.dumps({
        "phase": "17", "stage": "5", "deliverable": "temperature_analysis.json",
        "design": "identical 4-turn controlled context, frozen attacker, only the sampling "
                  "temperature differs; diagnostic only, not a reproduction result",
        "conditions": controlled_summary(controlled["temperature"], "condition")
        if "temperature" in controlled else None,
        "tests": ([test("T0.7 vs T0.3 at fixed context",
                        sum(1 for c in controlled["temperature"]
                            if c["condition"] == "T0.7" and c["json_valid"]),
                        sum(1 for c in controlled["temperature"]
                            if c["condition"] == "T0.7" and not c["json_valid"]),
                        sum(1 for c in controlled["temperature"]
                            if c["condition"] == "T0.3" and c["json_valid"]),
                        sum(1 for c in controlled["temperature"]
                            if c["condition"] == "T0.3" and not c["json_valid"]))]
                   if "temperature" in controlled else []),
    }, indent=2), encoding="utf-8")

    (STAGE5 / "instruction_position_analysis.json").write_text(json.dumps({
        "phase": "17", "stage": "5",
        "deliverable": "instruction_position_analysis.json",
        "design": "identical 4-turn context; only where the output-format instruction "
                  "appears differs (diagnostic wrapper; the production prompt is untouched)",
        "variants": {
            "I0": "original A0 instruction position (frozen control)",
            "I1": "format instruction appended as a final user message",
            "I2": "format instruction repeated at the end of the system prompt",
            "I3": "JSON schema placed at the end of the system prompt",
            "I4": "A1 wording appended to the system prompt",
        },
        "conditions": controlled_summary(controlled["instruction_position"], "condition")
        if "instruction_position" in controlled else None,
    }, indent=2), encoding="utf-8")

    (STAGE5 / "previous_output_ablation.json").write_text(json.dumps({
        "phase": "17", "stage": "5", "deliverable": "previous_output_ablation.json",
        "design": "identical 4-turn context; only the immediately previous attacker output "
                  "differs",
        "variants": {
            "P0": "full history (frozen control)",
            "P1": "previous attacker response removed",
            "P2": "previous attacker response replaced by minimal valid JSON",
            "P3": "previous attacker response replaced by fixed neutral prose",
            "P4": "previous attacker response replaced by a long valid JSON response",
        },
        "conditions": controlled_summary(controlled["prev_output"], "condition")
        if "prev_output" in controlled else None,
    }, indent=2), encoding="utf-8")

    made = figures(real, index_table, ctx_table, controlled)

    # §12 controlled comparisons vs each experiment's own control condition
    controlled_tests = {"phase": "17", "stage": "5",
                        "deliverable": "controlled_comparisons.json",
                        "framing": "each diagnostic condition against its own control "
                                   "condition, same context unless stated; all exploratory",
                        "comparisons": {}}

    def compare(experiment: str, control_id: str, treat_id: str, question: str) -> dict:
        rows_ = controlled.get(experiment, [])
        ctrl = [c for c in rows_ if c["condition"] == control_id]
        trt = [c for c in rows_ if c["condition"] == treat_id]
        if not ctrl or not trt:
            return {}
        result = test(question,
                      sum(1 for c in trt if c["json_valid"]),
                      sum(1 for c in trt if not c["json_valid"]),
                      sum(1 for c in ctrl if c["json_valid"]),
                      sum(1 for c in ctrl if not c["json_valid"]))
        result["mean_context_tokens"] = {
            "control": round(statistics.mean([c["input_token_count"] for c in ctrl
                                              if c.get("input_token_count")]), 1)
            if any(c.get("input_token_count") for c in ctrl) else None,
            "treatment": round(statistics.mean([c["input_token_count"] for c in trt
                                                if c.get("input_token_count")]), 1)
            if any(c.get("input_token_count") for c in trt) else None,
        }
        return result

    for name, control_id, treat_id, q in (
            ("context_length", "C1", "C5",
             "8 previous turns vs 1 previous turn"),
            ("context_length", "C0", "C5", "8 previous turns vs no history"),
            ("instruction_position", "I0", "I1",
             "instruction as final user message vs original position"),
            ("instruction_position", "I0", "I4", "A1 wording appended vs original position"),
            ("instruction_position", "I0", "I2",
             "instruction repeated at system end vs original position"),
            ("instruction_position", "I0", "I3", "schema at system end vs original position"),
            ("prev_output", "P0", "P1", "previous attacker output removed vs full history"),
            ("prev_output", "P0", "P2", "previous output = minimal JSON vs full history"),
            ("prev_output", "P0", "P3", "previous output = prose vs full history"),
            ("prev_output", "P0", "P4", "previous output = long valid JSON vs full history"),
            ("temperature", "T0.7", "T0.5", "T=0.5 vs T=0.7 at fixed context"),
            ("temperature", "T0.7", "T0.3", "T=0.3 vs T=0.7 at fixed context")):
        result = compare(name, control_id, treat_id, q)
        if result:
            controlled_tests["comparisons"][f"{name}:{control_id}->{treat_id}"] = result

    all_p = [t["p_value"] for t in TESTS]
    controlled_tests["test_discipline"] = {
        "n_exploratory_tests_total": len(all_p),
        "bonferroni_threshold": round(0.05 / len(all_p), 6) if all_p else None,
        "n_below_bonferroni": sum(1 for p in all_p
                                  if p < (0.05 / len(all_p) if all_p else 1)),
        "statement": "all comparisons are exploratory on one diagnostic run each; effect "
                     "sizes and intervals are the primary report, and no single p-value is "
                     "presented as a confirmed effect",
    }
    (STAGE5 / "controlled_comparisons.json").write_text(
        json.dumps(controlled_tests, indent=2), encoding="utf-8")

    p_values = [t["p_value"] for t in TESTS]
    summary = {
        "phase": "17", "stage": "5", "deliverable": "stage5_summary.json",
        "real_calls_analysed": len(real),
        "controlled_calls": {k: len(v) for k, v in controlled.items()},
        "first_vs_later": context_payload["first_vs_later"]["test"],
        "context_bucket_table": ctx_table,
        "exploratory_tests": len(TESTS),
        "min_p_value": min(p_values) if p_values else None,
        "bonferroni_threshold": round(0.05 / len(p_values), 6) if p_values else None,
        "figures": made,
        "replay_present": REPLAY_RUNS.is_file(),
    }
    (STAGE5 / "stage5_summary.json").write_text(json.dumps(summary, indent=2),
                                                encoding="utf-8")

    print(f"real calls instrumented: {len(real)}")
    print("controlled calls:", {k: len(v) for k, v in controlled.items()})
    print("call-index table:")
    for k, v in index_table.items():
        print(f"  index {k:3s}: {v['json_valid']}/{v['calls']} = {v['validity_rate']} "
              f"| ctx~{v['mean_context_tokens']} | prose {v['prose_failures']}")
    print("context buckets:")
    for k, v in ctx_table.items():
        print(f"  {k:12s}: {v['json_valid']}/{v['calls']} = {v['validity_rate']} "
              f"| prose {v['prose_failures']}")
    fvl = context_payload["first_vs_later"]["test"]
    print(f"call1 vs later: {fvl['rate_row1']} vs {fvl['rate_row2']} p={fvl['p_value']}")
    if context_payload["within_later_calls_context_split"]:
        ws = context_payload["within_later_calls_context_split"]
        print(f"later calls, context split: {ws['lower_half']['rate']} vs "
              f"{ws['upper_half']['rate']} p={ws['test']['p_value']}")
    print(f"figures: {made}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
