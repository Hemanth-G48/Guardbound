"""Phase 17 Stage 4 — isolated replication on REAL attack history.

Why this exists: the first isolated sample (150 calls) built its history from
fixed synthetic strings, and 6 of its 8 failures turned out to be the model
continuing that synthetic text rather than answering — a property of the probe,
not necessarily of the production pipeline. The production pipeline feeds back
the attacker's own previous questions and its own summaries of the target's
replies.

This script replays attacker calls with **history reconstructed from the
completed multi-turn runs**: for call k of a run, the history is the
`generatedQuestion` / `lastResponseSummary` pairs of calls 0..k-1, which are
exactly the strings the production attack appends to its history. Prompt,
call signature, sampling and classification are unchanged.

No retries, no repair, no prompt edits.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage4_qwen38_reliability"
SURVIVAL_RAW = OUT_DIR / "multiturn_survival_raw.jsonl"
RAW_PATH = OUT_DIR / "attacker_reliability_realhistory_raw.jsonl"
SUMMARY_PATH = OUT_DIR / "attacker_reliability_realhistory_summary.json"

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
VALID = ("VALID_DIRECT_JSON", "VALID_FROZEN_EXTRACTION")


def load_stage4_isolated():
    spec = importlib.util.spec_from_file_location(
        "phase17_stage4_isolated_reliability",
        REPO_ROOT / "scripts" / "phase17_stage4_isolated_reliability.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase17_stage4_isolated_reliability"] = module
    spec.loader.exec_module(module)
    return module


def reconstruct_history(calls: list[dict]) -> list[tuple[str, str]]:
    """(question, summary) pairs from a run's recorded attacker outputs."""
    pairs: list[tuple[str, str]] = []
    for call in calls:
        raw = call.get("raw_output")
        if not isinstance(raw, str):
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            from guardbound.llm.local_client import _extract_json_block

            parsed = _extract_json_block(raw)
        if isinstance(parsed, dict):
            pairs.append((str(parsed.get("generatedQuestion", "")),
                          str(parsed.get("lastResponseSummary", ""))))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-calls", type=int, default=200)
    args = parser.parse_args()

    if not SURVIVAL_RAW.is_file():
        print("no survival runs recorded yet")
        return 2
    runs = [json.loads(line) for line in SURVIVAL_RAW.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    stage4 = load_stage4_isolated()
    if RAW_PATH.exists():
        RAW_PATH.unlink()

    from guardbound.llm.qwen38_native_client import Qwen38NativeChatLLM

    client = Qwen38NativeChatLLM(
        model_id=MODEL_ID, revision=REVISION, device_map={"": 0},
        max_new_tokens=None, top_p=1.0, quantization="nf4",
        chat_template_kwargs={"enable_thinking": False},
    )
    print(f"loading native attacker for the real-history replication", flush=True)
    client._get_pipeline()

    step_functions = {attack: stage4.load_step(attack) for attack in
                      ("crescendo_paper", "opposite_day", "acronym")}
    rows: list[dict] = []
    for run in runs:
        attack = run["attack"]
        step = step_functions.get(attack)
        if step is None:
            continue
        history_pairs = reconstruct_history(run["attacker_calls"])
        # Replay every call after the first: the first has no history, which the
        # synthetic sample already covers.
        for call_index, call in enumerate(run["attacker_calls"][1:], start=1):
            if len(rows) >= args.max_calls:
                break
            round_num = call_index + 1
            history_attacker = [q for q, _ in history_pairs[:call_index]]
            history_target = [f"(Summary) {s}" for _, s in history_pairs[:call_index]]
            scores = [3] * len(history_attacker)
            last_response = history_pairs[call_index - 1][1] if call_index else ""
            proxy = stage4.RecordingProxy(client)
            error = None
            try:
                step(round_num=round_num, goal=run["goal"],
                     history_attacker=history_attacker, history_target=history_target,
                     scores=scores, last_response=last_response,
                     attacker_llm=proxy, max_rounds=8)
                outcome = "OK"
            except Exception as exc:  # noqa: BLE001
                outcome = type(exc).__name__
                error = str(exc)[:400]
            recorded = proxy.calls[-1] if proxy.calls else {}
            raw = recorded.get("pre_parse_reply")
            verdict = stage4.classify_reply(raw if raw is not None else "")
            stats = recorded.get("generation_stats") or {}
            row = {
                "sample": "real_history",
                "source_run": run["run_index"],
                "attack": attack,
                "goal": run["goal"],
                "round": round_num,
                "history_depth": len(history_attacker),
                "step_outcome": outcome,
                "step_error": error,
                "category": verdict["category"],
                "failure_shape": verdict["failure_shape"],
                "cause": ("CAUSE UNKNOWN" if verdict["category"] not in VALID else None),
                "termination": stats.get("termination"),
                "generated_tokens": stats.get("generated_tokens"),
                "prompt_tokens": stats.get("prompt_tokens"),
                "latency_s": recorded.get("latency_s"),
                "raw_output": raw,
            }
            rows.append(row)
            with RAW_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{len(rows)}/{args.max_calls}] run {run['run_index']} {attack} "
                  f"round={round_num} {row['category']:<26} outcome={outcome:<22} "
                  f"gen={row['generated_tokens']}", flush=True)
        if len(rows) >= args.max_calls:
            break

    n = len(rows)
    valid = sum(1 for r in rows if r["category"] in VALID)
    per_attack = {}
    for attack in ("crescendo_paper", "opposite_day", "acronym"):
        subset = [r for r in rows if r["attack"] == attack]
        if not subset:
            continue
        ok = sum(1 for r in subset if r["category"] in VALID)
        per_attack[attack] = {
            "n": len(subset), "valid": ok, "valid_rate": round(ok / len(subset), 4),
            "failures": len(subset) - ok,
            "categories": dict(Counter(r["category"] for r in subset)),
            "failure_shapes": dict(Counter(r["failure_shape"] for r in subset if r["failure_shape"])),
        }
    latencies = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    summary = {
        "phase": "17", "stage": "4",
        "deliverable": "attacker_reliability_realhistory_summary.json",
        "sample": "real_history",
        "why": ("history reconstructed from the completed multi-turn runs (the attacker's own "
                "questions and summaries), so the probe does not inject synthetic text the "
                "model can continue"),
        "overall": {
            "n": n, "valid": valid, "valid_rate": round(valid / n, 4) if n else None,
            "failures": n - valid,
            "categories": dict(Counter(r["category"] for r in rows)),
            "failure_shapes": dict(Counter(r["failure_shape"] for r in rows if r["failure_shape"])),
            "latency_s": {"mean": round(statistics.mean(latencies), 3),
                          "max": max(latencies)} if latencies else None,
        },
        "per_attack": per_attack,
        "by_round": {
            str(round_num): {
                "n": len(subset),
                "valid": sum(1 for r in subset if r["category"] in VALID),
                "valid_rate": round(sum(1 for r in subset if r["category"] in VALID) / len(subset), 4),
            }
            for round_num, subset in sorted(
                ((r["round"], [x for x in rows if x["round"] == r["round"]]) for r in rows),
                key=lambda kv: kv[0],
            )
        } if rows else {},
        "no_retry_policy": "no retry, no repair, no prompt change",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nreal-history sample: {valid}/{n} valid "
          f"({round(valid / n, 4) if n else 'n/a'})")
    for attack, block in per_attack.items():
        print(f"  {attack:16s} {block['valid']}/{block['n']} = {block['valid_rate']}")
    print("failure shapes:", dict(Counter(r["failure_shape"] for r in rows if r["failure_shape"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
