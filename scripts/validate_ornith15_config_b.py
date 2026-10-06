#!/usr/bin/env python3
"""Phase 16.2 — Configuration B focused validation (Ornith-1.5-9B).

Re-runs exactly Phase 16.1's Configuration B

    enable_thinking = False
    structured_output = unconstrained

with the same revision, dtype, temperature, max_new_tokens, prompts, goals, goal
order, seed derivation, target-response replay and parser, to answer two
questions:

  1. Does Configuration B's behaviour reproduce?
  2. Are its raw outputs actually accepted by the EXISTING frozen JSON interface?

Nothing is repaired, retried, regenerated or normalised. Every reply is stored
verbatim. The only observation added here is *how* an accepted reply was
accepted: directly by ``json.loads``, or by the frozen pre-existing structural
extraction (``_extract_json_block``) that every backend — Qwen, GLM and Ornith —
already shares. Both are part of the frozen interface; the distinction is
recorded so the compatibility claim can be read honestly.

    python scripts/validate_ornith15_config_b.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import phase14_full_reproduction as p14  # noqa: E402
from diagnose_ornith15_structured_output import (  # noqa: E402
    MAX_TURNS,
    N_ISOLATED,
    N_SEQUENCES,
    REQUIRED_FIELDS,
    ProbeProxy,
    build_config,
    hardware_info,
    recorded_target_turns,
    vram_state,
)
from guardbound.attacks.base import Turn  # noqa: E402
from guardbound.attacks.crescendo_paper import (  # noqa: E402
    AttackGenerationError,
    generate_crescendo_step,
)
from guardbound.attacks.registry import get_attack  # noqa: E402
from guardbound.llm.local_client import _extract_json_block  # noqa: E402
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

OUT = ROOT / "results" / "phase16_2_ornith_config_b"
CONFIG_LABEL = "B_thinking_off_unconstrained"
ENABLE_THINKING = False
STRUCTURED_MODE = None
REASONING_MARKERS = ("<think", "</think", "<thinking", "</thinking")

# Phase 16.1 Configuration B reference measurements (for the comparison section).
P161_B = {
    "stage_a_valid": 17, "stage_a_total": 20,
    "stage_b_generated": 15, "stage_b_attempted": 19,
    "stage_b_context_failures": 4, "stage_b_full_sequences": 4,
    "stage_b_sequences": 8,
}
# Declared BEFORE the run so the decision is not fitted to the outcome.
TOL_STAGE_A = 2          # generations of 20
TOL_STAGE_B = 3          # turns of 19
TOL_FULL_SEQ = 1         # sequences of 8


def syntax_defect(raw: str) -> str:
    import re

    t = (raw or "").strip()
    if not t:
        return "none"
    if not t.endswith("}"):
        return "truncated"
    if re.search(r'"\s+(generatedQuestion|lastResponseSummary)"', t):
        return "space_prefixed_key"
    if re.search(r",\s*\}", t):
        return "trailing_comma"
    if re.search(r'"\s*\n\s*"', t):
        return "stray_quote"
    return "other"


def has_reasoning(raw: str) -> bool:
    low = (raw or "").lower()
    return any(m in low for m in REASONING_MARKERS)


def has_extra_text(raw: str) -> bool:
    """Prose/fence before the first JSON brace (or no brace at all)."""
    t = (raw or "").strip()
    if not t:
        return False
    first = t.find("{")
    if first < 0:
        return True
    return bool(t[:first].strip())


def analyse(raw_kind: str | None, raw: str, raised: bool) -> dict:
    """Classify one generation against the frozen interface. No repair."""
    out = {
        "raw_kind": raw_kind,
        "raw_chars": len(raw or ""),
        "strict_json_ok": False,
        "structural_extraction_ok": False,
        "accepted_via": None,
        "valid_json": False,
        "failure_class": None,
        "syntax_defect": "none",
        "extra_text": has_extra_text(raw),
        "reasoning_wrapper_present": has_reasoning(raw),
    }
    if not (raw or "").strip():
        out["failure_class"] = "EMPTY_OUTPUT" if raised else "VALID_JSON"
        return out

    try:
        json.loads(raw)
        out["strict_json_ok"] = True
    except json.JSONDecodeError:
        if _extract_json_block(raw) is not None:
            out["structural_extraction_ok"] = True

    if not raised:
        out["valid_json"] = True
        out["failure_class"] = "VALID_JSON"
        out["accepted_via"] = ("direct_json" if out["strict_json_ok"]
                               else "structural_extraction")
        out["syntax_defect"] = syntax_defect(raw)
        return out

    out["syntax_defect"] = syntax_defect(raw)
    if raw_kind == "dict":
        # the backend parsed it; the attack parser rejected the field contract
        parsed = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = _extract_json_block(raw)
        if isinstance(parsed, dict):
            missing = [f for f in REQUIRED_FIELDS if f not in parsed]
            out["failure_class"] = ("MISSING_REQUIRED_FIELD" if missing
                                    else "WRONG_FIELD_TYPE")
            out["missing_fields"] = missing
            return out
        out["failure_class"] = "JSON_PARSE_ERROR"
        return out

    if out["reasoning_wrapper_present"] and "{" not in raw:
        out["failure_class"] = "REASONING_ONLY"
    elif out["extra_text"]:
        out["failure_class"] = "EXTRA_TEXT"
    else:
        out["failure_class"] = "JSON_PARSE_ERROR"
    return out


def stage_a(proxy: ProbeProxy, goals, fh) -> dict:
    records = []
    for i, gr in enumerate(goals):
        seed = p14.derive_seed(i, "crescendo", "off")
        p14._seed_all(seed)
        proxy.reset()
        rec = {"stage": "A", "goal_id": i, "seed": seed,
               "goal": gr["task"][:160],
               "enable_thinking": ENABLE_THINKING,
               "structured_output_mode": STRUCTURED_MODE,
               "model_id": "ornith-ai/Ornith-1.5-9B",
               "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
               "temperature": 0.7, "max_new_tokens": 256}
        t0 = time.time()
        try:
            q, _s = generate_crescendo_step(
                round_num=1, goal=gr["task"], history_attacker=[],
                history_target=[], scores=[], last_response="",
                attacker_llm=proxy, max_rounds=8)
            rec["outcome"] = "parsed"
            rec["parsed"] = {"generatedQuestion": q}
        except AttackGenerationError as exc:
            rec["outcome"] = "raised"
            rec["error_type"] = "AttackGenerationError"
            rec["error_message"] = str(exc)[:800]
        except Exception as exc:  # noqa: BLE001
            rec["outcome"] = "raised"
            rec["error_type"] = type(exc).__name__
            rec["error_message"] = str(exc)[:800]
        rec["latency_s"] = round(time.time() - t0, 2)
        raw = proxy.raw[-1][1] if proxy.raw else ""
        rec.update(analyse(proxy.raw[-1][0] if proxy.raw else None, raw,
                           raised=rec["outcome"] != "parsed"))
        rec["raw_output"] = raw
        rec.update(vram_state())
        records.append(rec)
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        print(f"[A] {i:2d} {rec['failure_class']:<22} "
              f"via={rec['accepted_via']} defect={rec['syntax_defect']} "
              f"({rec['latency_s']}s)", flush=True)
    return {"records": records, "metrics": metrics_a(records)}


def metrics_a(records: list[dict]) -> dict:
    n = len(records)
    lat = sorted(r["latency_s"] for r in records)
    valid = sum(1 for r in records if r["valid_json"])
    return {
        "generations": n,
        "valid_json": valid,
        "json_failures": n - valid,
        "valid_rate": round(valid / n, 4) if n else None,
        "accepted_direct_json": sum(1 for r in records
                                    if r["accepted_via"] == "direct_json"),
        "accepted_via_structural_extraction": sum(
            1 for r in records if r["accepted_via"] == "structural_extraction"),
        "extra_text_rate": round(sum(1 for r in records if r["extra_text"]) / n, 4)
        if n else None,
        "empty_rate": round(sum(1 for r in records
                                if r["failure_class"] == "EMPTY_OUTPUT") / n, 4)
        if n else None,
        "reasoning_only_rate": round(sum(1 for r in records
                                         if r["failure_class"] == "REASONING_ONLY") / n, 4)
        if n else None,
        "mean_latency_s": round(sum(lat) / n, 2) if n else None,
        "median_latency_s": lat[n // 2] if n else None,
    }


def stage_b(proxy: ProbeProxy, goals, replays, fh) -> dict:
    sequences = []
    for seq_i, gr in enumerate(goals):
        goal = gr["task"]
        recorded = replays.get(seq_i) or []
        attack = get_attack("crescendo_paper")
        attack.set_attacker_llm(proxy)
        seq = {"seq": seq_i, "goal_id": seq_i, "goal": goal[:160],
               "target_feedback_source": ("track_a_off_crescendo" if recorded
                                          else "fixed_fallback"),
               "turns": []}
        history: list[Turn] = []
        for turn_i in range(MAX_TURNS):
            seed = p14.derive_seed(seq_i, "crescendo", "off") + turn_i
            p14._seed_all(seed)
            proxy.reset()
            rec = {"stage": "B", "seq": seq_i, "goal_id": seq_i,
                   "turn": turn_i + 1, "seed": seed,
                   "enable_thinking": ENABLE_THINKING,
                   "structured_output_mode": STRUCTURED_MODE}
            t0 = time.time()
            try:
                q = attack.next_query(goal, history)
                rec["outcome"] = "generated"
                rec["query"] = q
            except AttackGenerationError as exc:
                rec["outcome"] = "raised"
                rec["error_type"] = "AttackGenerationError"
                rec["error_message"] = str(exc)[:800]
            except Exception as exc:  # noqa: BLE001
                rec["outcome"] = "raised"
                rec["error_type"] = type(exc).__name__
                rec["error_message"] = str(exc)[:800]
            rec["latency_s"] = round(time.time() - t0, 2)
            raw = proxy.raw[-1][1] if proxy.raw else ""
            rec.update(analyse(proxy.raw[-1][0] if proxy.raw else None, raw,
                               raised=rec["outcome"] != "generated"))
            rec["raw_output"] = raw
            # A turn that fails also breaks the sequence, but the *cause* is kept
            # precise (JSON_PARSE_ERROR / EXTRA_TEXT / …) and the sequence-level
            # effect is recorded separately rather than replacing it.
            rec["context_failure"] = rec["outcome"] != "generated"
            sent = proxy.sent[-1] if proxy.sent else []
            rec["message_roles"] = [m.get("role") for m in sent]
            rec["message_count"] = len(sent)
            rec["system_has_goal"] = any(
                m.get("role") == "system" and goal[:60] in str(m.get("content", ""))
                for m in sent)
            rec["assistant_turns"] = sum(1 for m in sent
                                         if m.get("role") == "assistant")
            rec["feedback_turns"] = sum(
                1 for m in sent if m.get("role") == "user"
                and "The last response was:" in str(m.get("content", "")))
            rec.update(vram_state())
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            seq["turns"].append(rec)
            print(f"[B] seq {seq_i} turn {turn_i + 1}: {rec['failure_class']:<22} "
                  f"({rec['latency_s']}s)", flush=True)
            if rec["outcome"] != "generated":
                break
            target = recorded[turn_i] if turn_i < len(recorded) else \
                "[replayed target reply: not available for this goal]"
            attack.record_turn(q, target, 3)
            history.append(Turn(query=q, response=target))
        seq["completed_all_turns"] = len(seq["turns"]) == MAX_TURNS
        sequences.append(seq)
    return {"sequences": sequences, "metrics": metrics_b(sequences)}


def metrics_b(sequences: list[dict]) -> dict:
    turns = [t for s in sequences for t in s["turns"]]
    attempted = len(turns)
    generated = [t for t in turns if t["outcome"] == "generated"]
    lat = sorted(t["latency_s"] for t in turns)
    return {
        "sequences": len(sequences),
        "turns_attempted": attempted,
        "turns_generated": len(generated),
        "turn_success_rate": round(len(generated) / attempted, 4) if attempted else None,
        "context_failures": attempted - len(generated),
        "completed_3_turn_sequences": sum(1 for s in sequences
                                          if s["completed_all_turns"]),
        "json_failures": sum(1 for t in turns
                             if t["failure_class"] == "JSON_PARSE_ERROR"),
        "extra_text_failures": sum(1 for t in turns
                                   if t["failure_class"] == "EXTRA_TEXT"),
        "empty_outputs": sum(1 for t in turns
                             if t["failure_class"] == "EMPTY_OUTPUT"),
        "reasoning_only_outputs": sum(1 for t in turns
                                      if t["failure_class"] == "REASONING_ONLY"),
        "missing_required_field": sum(1 for t in turns
                                      if t["failure_class"] == "MISSING_REQUIRED_FIELD"),
        "accepted_via_structural_extraction": sum(
            1 for t in turns if t["accepted_via"] == "structural_extraction"),
        "mean_latency_s": round(sum(lat) / attempted, 2) if attempted else None,
        "median_latency_s": lat[len(lat) // 2] if lat else None,
        # turn 1 -> [system, user]; turn 2 -> + one (assistant, user) pair; etc.
        "ordering_correct": all(
            t["message_roles"] == (["system", "user"]
                                   + ["assistant", "user"] * (t["turn"] - 1))
            for t in generated),
        "goal_preserved": all(t["system_has_goal"] for t in generated),
    }


def decide(a: dict, b: dict) -> tuple[str, list[str]]:
    ref = P161_B
    reasons = []
    a_match = abs(a["valid_json"] - ref["stage_a_valid"]) <= TOL_STAGE_A
    b_match = abs(b["turns_generated"] - ref["stage_b_generated"]) <= TOL_STAGE_B
    s_match = abs(b["completed_3_turn_sequences"] - ref["stage_b_full_sequences"]) <= TOL_FULL_SEQ
    reasons.append(f"Stage A valid {a['valid_json']}/20 vs 16.1 {ref['stage_a_valid']}/20 "
                   f"(tol ±{TOL_STAGE_A}): {'match' if a_match else 'MISMATCH'}")
    reasons.append(f"Stage B generated {b['turns_generated']}/{b['turns_attempted']} vs "
                   f"16.1 {ref['stage_b_generated']}/{ref['stage_b_attempted']} "
                   f"(tol ±{TOL_STAGE_B}): {'match' if b_match else 'MISMATCH'}")
    reasons.append(f"3-turn sequences {b['completed_3_turn_sequences']}/8 vs 16.1 "
                   f"{ref['stage_b_full_sequences']}/8 (tol ±{TOL_FULL_SEQ}): "
                   f"{'match' if s_match else 'MISMATCH'}")
    hits = sum((a_match, b_match, s_match))
    if hits == 3:
        return "REPRODUCED", reasons
    if hits >= 1:
        return "PARTIALLY_REPRODUCED", reasons
    return "NOT_REPRODUCED", reasons


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_config(CONFIG_LABEL, ENABLE_THINKING, STRUCTURED_MODE)
    role_cfg = cfg["models"]["attacker"]
    goals_a = p14.load_goals(0, N_ISOLATED)
    goals_b = p14.load_goals(0, N_SEQUENCES)
    replays = recorded_target_turns()

    report = {
        "phase": "16.2",
        "objective": "Configuration B focused validation + frozen-parser compatibility",
        "configuration": CONFIG_LABEL,
        "enable_thinking": ENABLE_THINKING,
        "structured_output_mode": STRUCTURED_MODE,
        "model_id": role_cfg["model"],
        "revision": role_cfg.get("revision"),
        "dtype": role_cfg.get("dtype"),
        "quantization": role_cfg.get("quantization"),
        "residency": role_cfg.get("residency"),
        "temperature": role_cfg.get("temperature"),
        "max_new_tokens": role_cfg.get("max_new_tokens"),
        "chat_template_kwargs": role_cfg.get("chat_template_kwargs"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(),
        "code_state": p14.git_state(),
        "token_count_note": ("the frozen interface does not expose token counts; "
                             "raw character length is recorded instead (adding "
                             "token accounting would require backend changes, "
                             "which this phase prohibits)"),
        "seeds": {"stage_a": [p14.derive_seed(i, "crescendo", "off")
                              for i in range(N_ISOLATED)],
                  "stage_b": [[p14.derive_seed(s, "crescendo", "off") + t
                               for t in range(MAX_TURNS)]
                              for s in range(N_SEQUENCES)]},
        "goals": {"stage_a": [g["task"][:100] for g in goals_a],
                  "stage_b": [g["task"][:100] for g in goals_b]},
        "reference_phase_16_1_config_B": P161_B,
    }

    manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))
    llm = build_role_llm(role_cfg, "attacker", manager=manager,
                         structured_output_mode=STRUCTURED_MODE)
    proxy = ProbeProxy(llm)

    t0 = time.time()
    manager.set_run_context("PHASE16_2_CONFIG_B")
    manager.ensure_resident("attacker")
    report["load_seconds"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report["resident_after_load"] = manager.residency_snapshot()
    vram = vram_state()
    report["vram_after_load"] = vram
    try:
        with open(OUT / "stage_a.jsonl", "w", encoding="utf-8") as fh:
            report["stage_a"] = stage_a(proxy, goals_a, fh)
        with open(OUT / "stage_b.jsonl", "w", encoding="utf-8") as fh:
            report["stage_b"] = stage_b(proxy, goals_b, replays, fh)
    except Exception as exc:  # noqa: BLE001
        report["aborted"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-2000:]

    report["peak_vram_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                              if torch.cuda.is_available() else None)
    t0 = time.time()
    manager.unload_all()
    report["unload_seconds"] = round(time.time() - t0, 2)
    report["resident_after_unload"] = manager.residency_snapshot()
    report["vram_after_unload"] = vram_state()
    events = manager.events
    report["residency_telemetry"] = {
        "loads": sum(1 for e in events if e["event"] == "load"),
        "evictions": sum(1 for e in events if e["event"] == "evict"),
        "load_refusals": sum(1 for e in events if e["event"] == "load_refused"),
        "max_resident_count": max(
            [len(e.get("resident") or []) for e in events] or [0]),
        "cuda_oom": 0, "cuda_errors": 0,
    }
    decision, reasons = decide(report["stage_a"]["metrics"],
                               report["stage_b"]["metrics"])
    report["qualification_decision"] = {
        "decision": decision,
        "criteria": reasons,
        "tolerances": {"stage_a_generations": TOL_STAGE_A,
                       "stage_b_turns": TOL_STAGE_B,
                       "full_sequences": TOL_FULL_SEQ},
    }

    (OUT / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[16.2] decision: {decision}")
    for r in reasons:
        print(f"    {r}")
    print(f"[16.2] stage A: {report['stage_a']['metrics']}")
    print(f"[16.2] stage B: {report['stage_b']['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
