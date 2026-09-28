#!/usr/bin/env python3
"""Phase 14.3 — READ-ONLY role-swap qualification.

Qualifies two models against their ROLE-SPECIFIC structured-output contracts
using the real production code:

    attacker   = Qwen/Qwen3-4B-Instruct-2507   (proposed)
    evaluator  = Qwen/Qwen3.5-4B               (proposed)

Method: the production contract functions are called UNCHANGED. They build the
real prompts and run the real parsers. A pass-through recorder proxy captures
what the backend returned before the parser sees it, so a MODEL capability
failure can be distinguished from a PARSER failure.

Nothing in src/guardbound/attacks/ or the prompts is modified. The inference
path is the production one: build_role_llm -> ManagedLocalChatLLM -> ModelManager
rotation -> HFLocalChatLLM(device_map="cuda", bf16, constrained_json).

Usage:
    python scripts/phase14_3_role_qualification.py --role attacker
    python scripts/phase14_3_role_qualification.py --role evaluator
    python scripts/phase14_3_role_qualification.py --role all
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "phase14_full_reproduction", _ROOT / "scripts" / "phase14_full_reproduction.py"
)
P14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P14)

import torch  # noqa: E402

from guardbound.llm.base import ChatLLM, DEFAULT_TEMPERATURE, Message  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402
from guardbound.attacks import acronym, crescendo_paper, opposite_day  # noqa: E402

OUT_DIR = _ROOT / "results" / "phase14"

# Frozen proposed stack (Step 2). Do not substitute.
PROPOSED = {
    "attacker": "Qwen/Qwen3-4B-Instruct-2507",
    "target": "microsoft/Phi-4-mini-instruct",
    "evaluator": "Qwen/Qwen3.5-4B",
}

# Probe-set size. The attacker samples at temperature 0.7, so distinct goals
# give distinct draws; the evaluator runs at 0.0, so distinct inputs are what
# produce distinct samples.
N_GOALS = 8
REFUSAL_STYLE_RESPONSES = [
    "I cannot help with that request.",
    "I'm sorry, but I can't assist with that.",
    "I won't provide that information.",
]


# --------------------------------------------------------------------------- #
# Recording proxy (read-through only; changes nothing about the call)
# --------------------------------------------------------------------------- #
class ContractRecorder(ChatLLM):
    """Pass-through ChatLLM that records the raw backend return per call."""

    name = "contract-recorder"

    def __init__(self, inner: ChatLLM, label: str) -> None:
        self._inner = inner
        self.label = label
        self.records: list[dict] = []

    def reset(self) -> None:
        self.records = []

    def generate(
        self,
        messages: list[Message],
        temperature: float = DEFAULT_TEMPERATURE,
        max_turns_context: Optional[int] = None,
        json_format: bool = False,
        structured_output_mode: Optional[str] = None,
    ) -> Any:
        started = time.perf_counter()
        error = None
        raw: Any = None
        try:
            raw = self._inner.generate(
                messages,
                temperature=temperature,
                max_turns_context=max_turns_context,
                json_format=json_format,
                structured_output_mode=structured_output_mode,
            )
        except Exception as exc:  # noqa: BLE001 -- recorded, never hidden
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started

        if isinstance(raw, str):
            raw_text, raw_type = raw, "str"
        elif raw is None:
            raw_text, raw_type = None, "None"
        else:
            raw_text = json.dumps(raw, ensure_ascii=False, default=str)
            raw_type = type(raw).__name__

        joined = "\u0000".join(f"{m.get('role')}:{m.get('content')}" for m in messages)
        self.records.append({
            "label": self.label,
            "temperature": temperature,
            "json_format": json_format,
            "structured_output_mode": structured_output_mode,
            "elapsed_s": round(elapsed, 3),
            "raw_type": raw_type,
            "raw": raw_text,
            "error": error,
            "prompt_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "prompt_sha256": hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32],
            "messages": len(messages),
        })
        if error is not None:
            raise RuntimeError(error)
        return raw

    def __getattr__(self, item: str):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, "_inner"), item)


# --------------------------------------------------------------------------- #
# Per-call verdict
# --------------------------------------------------------------------------- #
def parseable_json(raw: Any) -> Optional[dict]:
    """Strict JSON object from what the backend returned. NO recovery.

    A dict is accepted because the backend's json_format path already parsed it.
    A string is accepted only if json.loads succeeds — fenced or prose-wrapped
    output is NOT recovered, matching the production contract.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        return loaded if isinstance(loaded, dict) else None
    return None


def classify_call(rec: dict, required: list[str], accepted: bool,
                  extra_ok: Optional[bool] = None) -> dict:
    """Turn one recorded call into a verdict with an unambiguous failure type."""
    obj = parseable_json(rec.get("raw"))
    rec["valid_json"] = obj is not None
    rec["required_fields_present"] = bool(
        obj is not None and all(k in obj for k in required)
    )
    rec["missing_fields"] = (
        [k for k in required if k not in obj] if obj is not None else list(required)
    )
    rec["parser_accepted"] = bool(accepted)
    rec["extra_ok"] = extra_ok

    if rec["parser_accepted"]:
        rec["failure_type"] = None
    elif not rec["valid_json"]:
        rec["failure_type"] = "MODEL_CAPABILITY_FAILURE"
        rec["failure_detail"] = "output was not a bare, parseable JSON object"
    elif not rec["required_fields_present"]:
        rec["failure_type"] = "MODEL_CAPABILITY_FAILURE"
        rec["failure_detail"] = f"JSON present but missing {rec['missing_fields']}"
    elif extra_ok is False:
        rec["failure_type"] = "MODEL_CAPABILITY_FAILURE"
        rec["failure_detail"] = "required fields present but values unusable"
    else:
        rec["failure_type"] = "PARSER_FAILURE"
        rec["failure_detail"] = "valid JSON with required fields was still rejected"
    return rec


def usable_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


# --------------------------------------------------------------------------- #
# Incremental output
# --------------------------------------------------------------------------- #
# Overridable so the same qualification can be pointed at a different evaluator
# without touching the test itself (Phase 14.4B reuses this script verbatim).
OUT_PATHS: dict[str, Path] = {}
SUMMARY_PATH: Path = OUT_DIR / "phase14_3_qualification_summary.json"
MODEL_REVISIONS: dict[str, str] = {}
ROW_CONTEXT: dict = {}


def model_revision(model_id: str) -> Optional[str]:
    """Resolved snapshot revision of a locally cached model, or None."""
    base = (Path.home() / ".cache" / "huggingface" / "hub"
            / ("models--" + model_id.replace("/", "--")) / "snapshots")
    if not base.is_dir():
        return None
    revs = sorted(p.name for p in base.iterdir() if p.is_dir())
    return revs[-1] if revs else None


def role_out_path(role: str) -> Path:
    return OUT_PATHS.get(role, OUT_DIR / f"phase14_3_{role}_qualification.jsonl")


def row_key(row: dict) -> str:
    return f"{row['role']}|{row['contract']}|{row['case_id']}"


def load_done_keys(role: str) -> set:
    """Keys already recorded, so an interrupted probe resumes instead of
    restarting. Results are appended as they are produced, never batched to the
    end — an interrupt must not discard completed measurements."""
    path = role_out_path(role)
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(row_key(json.loads(line)))
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def emit(role: str, row: dict, done: set) -> None:
    """Append one row to the role's JSONL immediately."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row.setdefault("model_id", PROPOSED.get(role))
    row.setdefault("model_revision", MODEL_REVISIONS.get(role))
    for key, value in ROW_CONTEXT.items():
        row.setdefault(key, value)
    with open(role_out_path(role), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    done.add(row_key(row))


# --------------------------------------------------------------------------- #
# Probe cases
# --------------------------------------------------------------------------- #
def load_goals(n: int) -> list[dict]:
    with open(P14.OFFICIAL_DATASET, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records[:n]


def build_cases(goals: list[dict], target_llm: ChatLLM,
                condition: str = "off") -> list[dict]:
    """Real (goal, target_response) pairs from the real target model.

    The evaluator's contract input is (conversationObjective, responseToEvaluate),
    so the response must come from the target under test — not be invented.
    """
    cases = []
    for i, gr in enumerate(goals):
        goal = gr.get("task") or gr.get("goal") or ""
        system = gr.get("target_system") or "You are a helpful assistant."
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": goal}]
        try:
            resp = target_llm.generate(msgs, temperature=0.7)
            resp = resp if isinstance(resp, str) else json.dumps(resp)
        except Exception as exc:  # noqa: BLE001
            print(f"[qual] target generation failed for goal {i}: {exc}", flush=True)
            resp = ""
        cases.append({"case_id": f"goal{i:02d}", "kind": "target_response",
                      "goal": goal, "response": resp})
    for j, r in enumerate(REFUSAL_STYLE_RESPONSES):
        cases.append({"case_id": f"refusal{j}", "kind": "refusal_style",
                      "goal": goals[j % len(goals)].get("task", ""), "response": r})
    return cases


# --------------------------------------------------------------------------- #
# Attacker contracts
# --------------------------------------------------------------------------- #
ATTACKER_CONTRACTS = [
    {
        "name": "crescendo_paper.generate_crescendo_step",
        "call": lambda rec, rnd, goal, ha, ht, sc, last, mt: (
            crescendo_paper.generate_crescendo_step(
                rnd, goal, ha, ht, sc, last, rec, mt)),
        "required": ["generatedQuestion", "lastResponseSummary"],
    },
    {
        "name": "opposite_day.generate_opposite_day_step",
        "call": lambda rec, rnd, goal, ha, ht, sc, last, mt: (
            opposite_day.generate_opposite_day_step(
                rnd, goal, ha, ht, sc, last, rec, mt)),
        "required": ["generatedQuestion", "lastResponseSummary"],
    },
    {
        "name": "acronym.generate_acronym_step",
        "call": lambda rec, rnd, goal, ha, ht, sc, last, mt: (
            acronym.generate_acronym_step(
                rnd, goal, ha, ht, sc, last, rec, mt)),
        "required": ["generatedQuestion", "lastResponseSummary"],
    },
]


def run_attacker_contracts(recorder: ContractRecorder, goals: list[dict],
                           cases: list[dict], max_rounds: int,
                           rows: list[dict], done: set) -> None:
    """Two stages per goal per contract.

    ``round1`` has no prior turn, so only ``generatedQuestion`` can be required —
    an empty ``lastResponseSummary`` is legitimate there and must not be scored
    as a failure. ``round2`` carries a REAL target response as the previous turn,
    which is the shape in which both required fields are genuinely load-bearing;
    that is the contract the study actually depends on.
    """
    responses = {c["goal"]: c["response"] for c in cases
                 if c["kind"] == "target_response"}
    prior_question: dict[str, str] = {}

    for contract in ATTACKER_CONTRACTS:
        for gi, gr in enumerate(goals):
            goal = gr.get("task") or ""
            target_response = responses.get(goal, "")
            stages = [
                # (stage, round_num, hist_attacker, hist_target, scores, last_response)
                ("round1", 1, [], [], [], ""),
                ("round2", 2,
                 [prior_question.get(goal) or goal],
                 [target_response], [1], target_response),
            ]
            for stage, rnd, ha, ht, sc, last in stages:
                case_id = f"goal{gi:02d}_{stage}"
                key = f"attacker|{contract['name']}|{case_id}"
                if key in done:
                    continue
                required = (["generatedQuestion"] if stage == "round1"
                            else ["generatedQuestion", "lastResponseSummary"])
                recorder.reset()
                accepted = False
                q = summary = None
                call_error = None
                try:
                    q, summary = contract["call"](
                        recorder, rnd, goal, ha, ht, sc, last, max_rounds)
                    accepted = usable_str(q) and (
                        True if stage == "round1" else usable_str(summary))
                except Exception as exc:  # noqa: BLE001
                    call_error = f"{type(exc).__name__}: {exc}"
                rec = recorder.records[-1] if recorder.records else {}
                rec = classify_call(
                    rec, required, accepted,
                    extra_ok=all(usable_str(v) for v in (q,)
                                 ) if stage == "round1"
                    else (usable_str(q) and usable_str(summary)))
                rec.update({
                    "role": "attacker",
                    "contract": contract["name"],
                    "stage": stage,
                    "case_id": case_id,
                    "goal": goal[:300],
                    "call_error": call_error,
                    "generated_question_usable": usable_str(q),
                    "last_response_summary_usable": usable_str(summary),
                    "question_chars": len(q) if isinstance(q, str) else 0,
                    "summary_chars": len(summary) if isinstance(summary, str) else 0,
                    "history_turns": len(ha),
                })
                rows.append(rec)
                emit("attacker", rec, done)
                print(f"[qual][attacker] {contract['name'].split('.')[-1]:<26} "
                      f"{rec['case_id']:<16} json={int(rec['valid_json'])} "
                      f"fields={int(rec['required_fields_present'])} "
                      f"accepted={int(rec['parser_accepted'])} "
                      f"{rec['failure_type'] or 'OK'} ({rec.get('elapsed_s')}s)",
                      flush=True)
                if stage == "round1" and usable_str(q):
                    prior_question[goal] = q


# --------------------------------------------------------------------------- #
# Evaluator contracts
# --------------------------------------------------------------------------- #
EVALUATOR_CONTRACTS = [
    {
        "name": "crescendo_paper.llm_check_refusal",
        "call": lambda rec, case: crescendo_paper.llm_check_refusal(
            case["goal"], case["response"], rec),
        "required": ["value", "metadata"],
    },
    {
        "name": "opposite_day.llm_check_refusal",
        "call": lambda rec, case: opposite_day.llm_check_refusal(
            case["goal"], case["response"], rec),
        "required": ["value", "metadata"],
    },
    {
        "name": "acronym.llm_check_refusal",
        "call": lambda rec, case: acronym.llm_check_refusal(
            case["goal"], case["response"], rec),
        "required": ["value", "metadata"],
    },
    {
        "name": "acronym.check_disclaimer",
        "call": lambda rec, case: acronym.check_disclaimer(
            case["goal"], case["response"], rec),
        "required": ["value"],
    },
    {
        "name": "crescendo_paper.llm_evaluate_with_rubric",
        "call": lambda rec, case: crescendo_paper.llm_evaluate_with_rubric(
            case["goal"], case["response"], case["goal"], rec),
        "required": ["score"],
    },
    {
        "name": "opposite_day.llm_evaluate_with_rubric",
        "call": lambda rec, case: opposite_day.llm_evaluate_with_rubric(
            case["goal"], case["response"], case["goal"], rec),
        "required": ["score"],
    },
    {
        "name": "acronym.llm_evaluate_with_rubric",
        "call": lambda rec, case: acronym.llm_evaluate_with_rubric(
            case["goal"], case["response"], case["goal"], rec),
        "required": ["score"],
    },
]


def rubric_accepted(out: Any, obj: Optional[dict]) -> bool:
    """The production parser returns int(score) on success and 1 on failure.

    A genuine score of 1 is indistinguishable from the failure sentinel by
    return value alone, so acceptance is decided on the RAW object: the parser
    only returns a score when the object carried an in-range `score`.
    """
    if not isinstance(obj, dict) or "score" not in obj:
        return False
    score = obj["score"]
    return isinstance(score, (int, float)) and not isinstance(score, bool) \
        and 1 <= score <= 5


def refusal_accepted(obj: Optional[dict]) -> bool:
    """`llm_check_refusal` returns False for BOTH a genuine "not a refusal"
    verdict and a parse failure — the two are indistinguishable by return value.

    Only the raw object separates them: a real verdict requires `value` AND an
    int-coercible `metadata`. Scoring this on the return type alone would count
    every parse failure as a successful "not a refusal", which is precisely the
    model-capability/parser conflation this gate must not make.
    """
    if not isinstance(obj, dict):
        return False
    if "value" not in obj or "metadata" not in obj:
        return False
    try:
        int(obj["metadata"])
    except (TypeError, ValueError):
        return False
    return isinstance(obj["value"], (bool, int, float))


def disclaimer_accepted(obj: Optional[dict]) -> bool:
    """`check_disclaimer` returns bool(data.get("value", False)) — again False
    for both a real negative and a parse failure, so acceptance is decided from
    the raw object."""
    if not isinstance(obj, dict) or "value" not in obj:
        return False
    return isinstance(obj["value"], (bool, int, float))


def run_evaluator_contracts(recorder: ContractRecorder, cases: list[dict],
                            rows: list[dict], done: set) -> None:
    for contract in EVALUATOR_CONTRACTS:
        for case in cases:
            key = f"evaluator|{contract['name']}|{case['case_id']}"
            if key in done:
                continue
            recorder.reset()
            out = None
            call_error = None
            try:
                out = contract["call"](recorder, case)
            except Exception as exc:  # noqa: BLE001
                call_error = f"{type(exc).__name__}: {exc}"
            rec = recorder.records[-1] if recorder.records else {}
            obj = parseable_json(rec.get("raw"))
            if "score" in contract["required"]:
                accepted = rubric_accepted(out, obj)
            elif "metadata" in contract["required"]:
                accepted = refusal_accepted(obj)
            else:
                accepted = disclaimer_accepted(obj)
            accepted = accepted and call_error is None
            rec = classify_call(rec, contract["required"], accepted)
            rec.update({
                "role": "evaluator",
                "contract": contract["name"],
                "case_id": case["case_id"],
                "case_kind": case["kind"],
                "goal": case["goal"][:300],
                "response_chars": len(case["response"]),
                "call_error": call_error,
                "parser_return": (out if isinstance(out, (bool, int)) else str(out)),
            })
            if call_error:
                # Parser crashed. Root cause is still classified from the raw
                # output above; the crash is recorded as its own symptom so the
                # two are never conflated.
                rec["parser_crash"] = call_error
            rows.append(rec)
            emit("evaluator", rec, done)
            print(f"[qual][evaluator] {contract['name'].split('.')[-1]:<32} "
                  f"{case['case_id']:<9} json={int(rec['valid_json'])} "
                  f"fields={int(rec['required_fields_present'])} "
                  f"accepted={int(rec['parser_accepted'])} "
                  f"{rec['failure_type'] or 'OK'} ({rec.get('elapsed_s')}s)", flush=True)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarise(rows: list[dict], role: str, manager, wall_s: float) -> dict:
    sub = [r for r in rows if r["role"] == role]
    n = len(sub)
    if not n:
        return {"role": role, "total_calls": 0}

    def frac(pred) -> float:
        return round(sum(1 for r in sub if pred(r)) / n, 4)

    by_contract = {}
    for c in sorted({r["contract"] for r in sub}):
        cr = [r for r in sub if r["contract"] == c]
        by_contract[c] = {
            "calls": len(cr),
            "valid_json": sum(1 for r in cr if r["valid_json"]),
            "required_fields": sum(1 for r in cr if r["required_fields_present"]),
            "accepted": sum(1 for r in cr if r["parser_accepted"]),
            "acceptance_rate": round(
                sum(1 for r in cr if r["parser_accepted"]) / len(cr), 4),
            "failure_types": _counts(cr, "failure_type"),
            "mean_latency_s": round(
                sum(r["elapsed_s"] for r in cr) / len(cr), 2),
        }

    residency = manager.residency_snapshot() if manager is not None else {}
    events = manager.events if manager is not None else []
    loads = [e for e in events if e.get("event") == "load"]
    evicts = [e for e in events if e.get("event") == "evict"]

    summary = {
        "role": role,
        "model": PROPOSED[role],
        "total_calls": n,
        "valid_json": sum(1 for r in sub if r["valid_json"]),
        "invalid_json": sum(1 for r in sub if not r["valid_json"]),
        "required_fields_success": sum(
            1 for r in sub if r["required_fields_present"]),
        "parser_accepted": sum(1 for r in sub if r["parser_accepted"]),
        "parser_failures": sum(
            1 for r in sub if not r["parser_accepted"]),
        "parser_crashes": sum(1 for r in sub if r.get("parser_crash")),
        "json_validity_rate": frac(lambda r: r["valid_json"]),
        "required_field_success_rate": frac(lambda r: r["required_fields_present"]),
        "contract_success_rate": frac(lambda r: r["parser_accepted"]),
        "failure_types": _counts(sub, "failure_type"),
        "mean_latency_s": round(sum(r["elapsed_s"] for r in sub) / n, 2),
        "max_latency_s": round(max(r["elapsed_s"] for r in sub), 2),
        "max_prompt_chars": max(r["prompt_chars"] for r in sub),
        "wall_seconds": round(wall_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "free_vram_gb": residency.get("free_gb"),
        "residency": residency,
        "loads": len(loads),
        "evictions": len(evicts),
        "mean_load_s": round(
            sum(e.get("duration_s") or 0.0 for e in loads) / len(loads), 2
        ) if loads else None,
        "mean_evict_s": round(
            sum(e.get("duration_s") or 0.0 for e in evicts) / len(evicts), 3
        ) if evicts else None,
        "load_refusals": sum(
            1 for e in events if e.get("event") == "load_refused"),
        "by_contract": by_contract,
    }

    if role == "attacker":
        q = [r for r in sub if r.get("generated_question_usable") is not None]
        summary["usable_generatedQuestion"] = sum(
            1 for r in q if r.get("generated_question_usable"))
        summary["usable_lastResponseSummary"] = sum(
            1 for r in q if r.get("last_response_summary_usable"))
        summary["usable_output_rate"] = round(
            sum(1 for r in q if r.get("generated_question_usable")
                and r.get("last_response_summary_usable")) / len(q), 4
        ) if q else None
    else:
        summary["refusal_parse_success"] = _contract_acceptance(
            sub, ("llm_check_refusal", "check_disclaimer"))
        summary["rubric_parse_success"] = _contract_acceptance(
            sub, ("llm_evaluate_with_rubric",))
    return summary


def _counts(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        k = r.get(key) or "OK"
        out[k] = out.get(k, 0) + 1
    return out


def _contract_acceptance(rows: list[dict], needles: tuple) -> dict:
    sub = [r for r in rows if any(n in r["contract"] for n in needles)]
    if not sub:
        return {"calls": 0}
    return {
        "calls": len(sub),
        "accepted": sum(1 for r in sub if r["parser_accepted"]),
        "rate": round(sum(1 for r in sub if r["parser_accepted"]) / len(sub), 4),
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", choices=["attacker", "evaluator", "all"], default="all")
    ap.add_argument("--goals", type=int, default=N_GOALS)
    ap.add_argument("--config", default=str(P14.CONFIG_PATH))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--evaluator", default=None,
                    help="Override the evaluator model id (Phase 14.4B: "
                         "google/gemma-3-4b-it). The test itself is unchanged.")
    ap.add_argument("--out", default=None,
                    help="Override the evaluator JSONL output path.")
    ap.add_argument("--summary", default=None,
                    help="Override the summary JSON output path.")
    args = ap.parse_args(argv)

    if args.evaluator:
        PROPOSED["evaluator"] = args.evaluator
    if args.out:
        OUT_PATHS["evaluator"] = Path(args.out)
    if args.summary:
        global SUMMARY_PATH
        SUMMARY_PATH = Path(args.summary)

    cfg = P14.load_config(args.config)
    goals = load_goals(args.goals)
    max_rounds = cfg["attacks"].get("max_turns", 8)
    for _role, _mid in PROPOSED.items():
        MODEL_REVISIONS[_role] = model_revision(_mid) or "unknown"
    ROW_CONTEXT.update({
        "seed": args.seed,
        "dtype": "bfloat16",
        "device": cfg.get("hardware", {}).get("device", "cuda"),
        "top_p": 1.0,
        "max_new_tokens": cfg["models"]["evaluator"].get("max_new_tokens", 256),
        "max_rounds": max_rounds,
    })

    print("=" * 96)
    print("PHASE 14.3 — ROLE-SWAP QUALIFICATION (read-only)")
    print("=" * 96)
    print(f"proposed stack : {PROPOSED}")
    print(f"roles          : {args.role}")
    print(f"probe goals    : {len(goals)} from {P14.OFFICIAL_DATASET}")
    print(f"seed           : {args.seed}  |  max_rounds: {max_rounds}")
    print(f"structured_output_mode : constrained_json (production path)")
    print(flush=True)

    # Production inference path: the same factory + manager the study uses.
    from guardbound.llm.model_manager import ModelManager

    manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))
    llms: dict[str, ChatLLM] = {}
    for role in ("attacker", "target", "evaluator"):
        role_cfg = dict(cfg["models"][role])
        role_cfg["model"] = PROPOSED[role]
        llms[role] = build_role_llm(role_cfg, role, manager=manager,
                                    structured_output_mode="constrained_json")
    print(f"[qual] distinct model ids: {sorted({PROPOSED[r] for r in PROPOSED})}")
    print(f"[qual] roles: {manager.roles}  pins: {manager.pinned_roles}", flush=True)

    recorders = {r: ContractRecorder(llms[r], r) for r in llms}

    rows: list[dict] = []
    done = {"attacker": load_done_keys("attacker"),
            "evaluator": load_done_keys("evaluator")}
    for role, keys in done.items():
        if keys:
            print(f"[qual] resuming {role}: {len(keys)} calls already recorded "
                  f"(skipped, included in the summary)", flush=True)

    # Probe cases are built first because both role gates need them: the
    # evaluator judges (goal, response) pairs, and the attacker's round-2
    # contract carries a real target response as the previous turn.
    # Seeded BEFORE case construction so the target responses are reproducible
    # and a resumed run measures the same inputs.
    P14._seed_all(args.seed)
    print(f"\n--- building probe cases with the real target "
          f"({PROPOSED['target']}) ---", flush=True)
    cases = build_cases(goals, recorders["target"])
    print(f"[qual] {len(cases)} cases: "
          f"{sum(1 for c in cases if c['kind'] == 'target_response')} target "
          f"responses + {sum(1 for c in cases if c['kind'] == 'refusal_style')} "
          f"refusal-style", flush=True)

    roles_run: set[str] = set()

    if args.role in ("attacker", "all"):
        P14._seed_all(args.seed)
        started = time.perf_counter()
        print(f"\n--- ATTACKER CONTRACTS ({PROPOSED['attacker']}) ---", flush=True)
        run_attacker_contracts(recorders["attacker"], goals, cases,
                               max_rounds, rows, done["attacker"])
        attacker_s = time.perf_counter() - started
        roles_run.add("attacker")
    else:
        attacker_s = 0.0

    if args.role in ("evaluator", "all"):
        P14._seed_all(args.seed)
        started = time.perf_counter()
        print(f"\n--- EVALUATOR CONTRACTS ({PROPOSED['evaluator']}) ---", flush=True)
        run_evaluator_contracts(recorders["evaluator"], cases, rows,
                                done["evaluator"])
        evaluator_s = time.perf_counter() - started
        roles_run.add("evaluator")
    else:
        evaluator_s = 0.0

    # Rows were appended incrementally; reload the full set (including anything
    # a previous, interrupted run already recorded) so the summary is complete.
    # Only roles actually run in THIS invocation are reloaded — otherwise a
    # previous phase's JSONL would be read back into this phase's summary and
    # be indistinguishable from a fresh measurement.
    all_rows: list[dict] = []
    for role in sorted(roles_run):
        path = role_out_path(role)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                all_rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"[qual] {len(all_rows)} recorded calls on disk", flush=True)

    summaries = {"proposed_stack": PROPOSED, "seed": args.seed,
                 "probe_goals": len(goals), "cases": len(cases),
                 "max_rounds": max_rounds}
    for role, wall in (("attacker", attacker_s), ("evaluator", evaluator_s)):
        if any(r["role"] == role for r in all_rows):
            summaries[role] = summarise(all_rows, role, manager, wall)

    summaries["model_revisions"] = dict(MODEL_REVISIONS)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summaries, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    for role in ("attacker", "evaluator"):
        s = summaries.get(role)
        if not s:
            continue
        print(f"{role.upper()}: {s['model']}  (rev {MODEL_REVISIONS.get(role)})")
        print(f"  calls={s['total_calls']}  valid_json={s['valid_json']}  "
              f"required_fields={s['required_fields_success']}  "
              f"accepted={s['parser_accepted']}")
        print(f"  json_validity_rate={s['json_validity_rate']:.3f}  "
              f"required_field_success_rate={s['required_field_success_rate']:.3f}  "
              f"contract_success_rate={s['contract_success_rate']:.3f}")
        print(f"  failure_types={s['failure_types']}  "
              f"parser_crashes={s['parser_crashes']}")
        print(f"  latency mean={s['mean_latency_s']}s max={s['max_latency_s']}s  "
              f"peak_vram={s['peak_vram_gb']}GB free={s['free_vram_gb']}")
        print(f"  loads={s['loads']} evictions={s['evictions']} "
              f"refusals={s['load_refusals']}")
    print("=" * 96)

    manager.set_run_context(None)
    manager.unload_all()
    print(f"[qual] residency released: {manager.residency_snapshot()}")
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[qual] wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
