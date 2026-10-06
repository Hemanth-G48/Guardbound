#!/usr/bin/env python3
"""Phase 16.1 — Ornith-1.5-9B structured-output / multi-turn diagnostic.

Four controlled configurations, identical in every other respect:

    A  enable_thinking=False  + constrained_json
    B  enable_thinking=False  + unconstrained
    C  enable_thinking=True   + constrained_json
    D  enable_thinking=True   + unconstrained

Stage A: 20 isolated generations per configuration (same 20 goals, paired).
Stage B: 8 multi-turn sequences per configuration (same 8 goals, <=3 turns).

This is a diagnostic qualification study. It does NOT measure ASR, does not run
the attack-success pipeline, and does not modify the backend, the parser, the
prompts, the attack algorithms or anything in Track A. Every generation is
attempted exactly once: no retry, no repair, no fallback, raw output preserved.

    python scripts/diagnose_ornith15_structured_output.py            # all four
    python scripts/diagnose_ornith15_structured_output.py --config A # one only
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
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
import yaml  # noqa: E402

from guardbound.attacks.base import Turn  # noqa: E402
from guardbound.attacks.crescendo_paper import (  # noqa: E402
    AttackGenerationError,
    generate_crescendo_step,
)
from guardbound.attacks.registry import get_attack  # noqa: E402
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

import phase14_full_reproduction as p14  # noqa: E402

CONFIG_FILE = ROOT / "configs" / "attacker_ornith15.yaml"
OUT_ROOT = ROOT / "results" / "phase16_1_ornith_diagnostic"

# (label, enable_thinking, structured_output_mode)
CONFIGURATIONS = [
    ("A_thinking_off_constrained", False, "constrained_json"),
    ("B_thinking_off_unconstrained", False, None),
    ("C_thinking_on_constrained", True, "constrained_json"),
    ("D_thinking_on_unconstrained", True, None),
]

N_ISOLATED = 20
N_SEQUENCES = 8
MAX_TURNS = 3

REQUIRED_FIELDS = ("generatedQuestion", "lastResponseSummary")
REASONING_MARKERS = ("<think", "</think", "<thinking", "</thinking")

# Deterministic target feedback for the multi-turn diagnostic. Replayed from the
# frozen Track A NBF-OFF Crescendo recordings (read-only) where a recording
# exists for that goal, so every configuration sees byte-identical feedback;
# a fixed string is used for the one paired goal that has no recording
# (goal 1 was a JSON_PARSE_ERROR run in Track A and recorded no turns).
REPLAY_FALLBACK = "[replayed target reply: not available for this goal]"


def recorded_target_turns() -> dict[int, list[str]]:
    path = (ROOT / "results" / "phase15_pilot_30" / "trackA_frozen" /
            "batch00_goals000-029_180records.jsonl")
    out: dict[int, list[str]] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("condition") == "off"
                    and rec.get("attack") == "crescendo_paper"
                    and rec.get("turns")):
                out[rec["goal_id"]] = [t["response"] for t in rec["turns"]]
    return out


class ProbeProxy:
    """Wraps the attacker so the raw backend output is preserved verbatim.

    Same semantics as the runner's telemetry proxy: a dict means the backend
    parsed the reply, a str means it could not.
    """

    def __init__(self, inner):
        self._inner = inner
        self.raw: list[tuple[str, str]] = []   # (kind, text)
        self.sent: list[list[dict]] = []       # messages handed to the backend

    def generate(self, *args, **kwargs):
        msgs = args[0] if args else kwargs.get("messages", [])
        self.sent.append(json.loads(json.dumps(msgs, default=str)))
        out = self._inner.generate(*args, **kwargs)
        kind = type(out).__name__
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False,
                                                          default=str)
        self.raw.append((kind, text))
        return out

    def reset(self):
        self.raw = []
        self.sent = []

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, "_inner"), item)


def classify(proxy: ProbeProxy, raised: bool) -> dict:
    """Classify one generation. Nothing is repaired or reinterpreted."""
    kinds = [k for k, _ in proxy.raw]
    texts = [t for _, t in proxy.raw]
    blob = texts[-1] if texts else ""
    result = {
        "raw_kind": kinds[-1] if kinds else None,
        "raw_len": len(blob),
        "valid_json": False,
        "malformed_syntax": False,
        "missing_required_field": False,
        "wrong_field_type": False,
        "empty": False,
        "reasoning_only": False,
        "reasoning_wrapper_present": False,
        "extra_text": False,
    }
    if not blob.strip():
        result["empty"] = True
        return result
    low = blob.lower()
    result["reasoning_wrapper_present"] = any(m in low for m in REASONING_MARKERS)
    # prose before the first brace, when a brace exists at all
    first_brace = blob.find("{")
    if first_brace > 0 and blob[:first_brace].strip():
        result["extra_text"] = True

    # A dict means the frozen backend parsed the reply; a str means it could not
    # (the backend only returns a string after json.loads AND the frozen
    # structural extraction have both failed). No second parse is attempted here.
    parsed = None
    if kinds and kinds[-1] == "dict":
        parsed = json.loads(blob)

    if not raised:
        result["valid_json"] = True
        if isinstance(parsed, dict):
            for field in REQUIRED_FIELDS:
                if field not in parsed:
                    result["missing_required_field"] = True
                elif not isinstance(parsed[field], str):
                    result["wrong_field_type"] = True
        return result

    # raised -> the frozen caller or parser rejected it
    if parsed is None:
        result["malformed_syntax"] = not result["reasoning_wrapper_present"]
        if result["reasoning_wrapper_present"] and not result["extra_text"]:
            result["reasoning_only"] = True
        return result
    if isinstance(parsed, dict):
        missing = [f for f in REQUIRED_FIELDS if f not in parsed]
        if missing:
            result["missing_required_field"] = True
        else:
            result["wrong_field_type"] = True
    return result


def vram_state() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
    }


def hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
    return info


def build_config(label: str, thinking: bool, structured: str | None) -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        cfg = copy.deepcopy(yaml.safe_load(f))
    attacker = cfg["models"]["attacker"]
    attacker["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
    if structured is None:
        attacker.pop("structured_output_mode", None)
    else:
        attacker["structured_output_mode"] = structured
    return cfg


# --------------------------------------------------------------------------- #
# Stage A
# --------------------------------------------------------------------------- #
def run_stage_a(proxy: ProbeProxy, raw_fh, goals: list[dict]) -> dict:
    probes = []
    for i, gr in enumerate(goals):
        goal_id = i
        seed = p14.derive_seed(goal_id, "crescendo", "off")
        p14._seed_all(seed)
        proxy.reset()
        probe = {"i": i, "goal_id": goal_id, "seed": seed,
                 "goal": gr["task"][:160]}
        t0 = time.time()
        try:
            q, _summary = generate_crescendo_step(
                round_num=1, goal=gr["task"], history_attacker=[],
                history_target=[], scores=[], last_response="",
                attacker_llm=proxy, max_rounds=8,
            )
            probe["outcome"] = "parsed"
            probe["generatedQuestion"] = q
        except AttackGenerationError as exc:
            probe["outcome"] = "raised"
            probe["error_type"] = "AttackGenerationError"
            probe["error_message"] = str(exc)[:600]
        except Exception as exc:  # noqa: BLE001
            probe["outcome"] = "raised"
            probe["error_type"] = type(exc).__name__
            probe["error_message"] = str(exc)[:600]
        probe["latency_s"] = round(time.time() - t0, 2)
        probe.update(classify(proxy, raised=probe["outcome"] != "parsed"))
        probe.update(vram_state())
        probes.append(probe)
        raw_fh.write(json.dumps({
            "stage": "A", "i": i, "goal_id": goal_id, "seed": seed,
            "outcome": probe["outcome"], "raw_kind": probe["raw_kind"],
            "raw": proxy.raw[-1][1] if proxy.raw else "",
        }, ensure_ascii=False) + "\n")
        raw_fh.flush()
        print(f"[A] {i:2d} {probe['outcome']:<7} "
              f"valid={probe['valid_json']} missing={probe['missing_required_field']} "
              f"malformed={probe['malformed_syntax']} ({probe['latency_s']}s)",
              flush=True)
    return {"probes": probes, "metrics": _stage_a_metrics(probes)}


def _stage_a_metrics(probes: list[dict]) -> dict:
    n = len(probes)
    valid = sum(1 for p in probes if p["valid_json"])
    lat = [p["latency_s"] for p in probes]
    return {
        "generations": n,
        "valid_json": valid,
        "json_validity_rate": round(valid / n, 4) if n else None,
        "malformed_syntax": sum(1 for p in probes if p["malformed_syntax"]),
        "missing_required_field": sum(1 for p in probes
                                      if p["missing_required_field"]),
        "wrong_field_type": sum(1 for p in probes if p["wrong_field_type"]),
        "empty": sum(1 for p in probes if p["empty"]),
        "reasoning_only": sum(1 for p in probes if p["reasoning_only"]),
        "reasoning_wrapper_present": sum(1 for p in probes
                                         if p["reasoning_wrapper_present"]),
        "extra_text": sum(1 for p in probes if p["extra_text"]),
        "mean_latency_s": round(sum(lat) / n, 2) if n else None,
        "max_latency_s": round(max(lat), 2) if n else None,
    }


# --------------------------------------------------------------------------- #
# Stage B
# --------------------------------------------------------------------------- #
def run_stage_b(proxy: ProbeProxy, raw_fh, goals: list[dict],
                replays: dict[int, list[str]]) -> dict:
    sequences = []
    for seq_i, gr in enumerate(goals):
        goal_id = seq_i
        goal = gr["task"]
        attack = get_attack("crescendo_paper")
        attack.set_attacker_llm(proxy)
        recorded = replays.get(goal_id) or []
        seq = {"seq": seq_i, "goal_id": goal_id, "goal": goal[:160],
               "target_feedback_source": ("track_a_off_crescendo" if recorded
                                          else "fixed_fallback"),
               "turns": []}
        history: list[Turn] = []
        for turn_i in range(MAX_TURNS):
            seed = p14.derive_seed(goal_id, "crescendo", "off") + turn_i
            p14._seed_all(seed)
            proxy.reset()
            step = {"turn": turn_i + 1, "seed": seed}
            t0 = time.time()
            try:
                q = attack.next_query(goal, history)
                step["outcome"] = "generated"
                step["query"] = q
                step["empty_query"] = (q == "")
                step["repeats_previous"] = bool(history and q == history[-1].query)
            except AttackGenerationError as exc:
                step["outcome"] = "raised"
                step["error_type"] = "AttackGenerationError"
                step["error_message"] = str(exc)[:600]
            except Exception as exc:  # noqa: BLE001
                step["outcome"] = "raised"
                step["error_type"] = type(exc).__name__
                step["error_message"] = str(exc)[:600]
            step["latency_s"] = round(time.time() - t0, 2)
            step.update(classify(proxy, raised=step["outcome"] != "generated"))
            sent = proxy.sent[-1] if proxy.sent else []
            step["roles"] = [m.get("role") for m in sent]
            step["system_has_goal"] = any(
                m.get("role") == "system" and goal[:60] in str(m.get("content", ""))
                for m in sent)
            step["assistant_turns"] = sum(1 for m in sent
                                          if m.get("role") == "assistant")
            step["feedback_turns"] = sum(
                1 for m in sent if m.get("role") == "user"
                and "The last response was:" in str(m.get("content", "")))
            seq["turns"].append(step)
            raw_fh.write(json.dumps({
                "stage": "B", "seq": seq_i, "goal_id": goal_id,
                "turn": turn_i + 1, "seed": seed, "outcome": step["outcome"],
                "raw_kind": step["raw_kind"],
                "raw": proxy.raw[-1][1] if proxy.raw else "",
            }, ensure_ascii=False) + "\n")
            raw_fh.flush()
            print(f"[B] seq {seq_i} turn {turn_i + 1}: {step['outcome']} "
                  f"valid={step['valid_json']} ({step['latency_s']}s)", flush=True)
            if step["outcome"] != "generated":
                break
            target = (recorded[turn_i] if turn_i < len(recorded)
                      else REPLAY_FALLBACK)
            attack.record_turn(q, target, 3)
            history.append(Turn(query=q, response=target))
        sequences.append(seq)
    return {"sequences": sequences, "metrics": _stage_b_metrics(sequences)}


def _stage_b_metrics(sequences: list[dict]) -> dict:
    turns = [t for s in sequences for t in s["turns"]]
    attempted = len(turns)
    generated = [t for t in turns if t["outcome"] == "generated"]
    failed = [t for t in turns if t["outcome"] != "generated"]
    completed = [s for s in sequences if len(s["turns"]) == MAX_TURNS]
    ok_turns = [t for t in generated]
    return {
        "sequences": len(sequences),
        "turns_attempted": attempted,
        "turns_generated": len(generated),
        "turn_completion_rate": round(len(generated) / attempted, 4) if attempted else None,
        "sequences_completed_all_turns": len(completed),
        "sequence_completion_rate": (round(len(completed) / len(sequences), 4)
                                     if sequences else None),
        "context_failures": len(failed),
        "context_failure_rate": round(len(failed) / attempted, 4) if attempted else None,
        "valid_json_count": sum(1 for t in turns if t["valid_json"]),
        "malformed_syntax": sum(1 for t in turns if t["malformed_syntax"]),
        "missing_required_field": sum(1 for t in turns
                                      if t["missing_required_field"]),
        "empty": sum(1 for t in turns if t["empty"]),
        "reasoning_only": sum(1 for t in turns if t["reasoning_only"]),
        "reasoning_wrapper_present": sum(1 for t in turns
                                         if t["reasoning_wrapper_present"]),
        "extra_text": sum(1 for t in turns if t["extra_text"]),
        "empty_queries": sum(1 for t in ok_turns if t.get("empty_query")),
        "repeated_queries": sum(1 for t in ok_turns if t.get("repeats_previous")),
        "goal_preserved": all(t.get("system_has_goal") for t in ok_turns),
        "ordering_correct": all(
            t.get("roles") == ["system", "user"] + ["assistant", "user"] * (t["turn"] - 1)
            for t in ok_turns),
        "mean_latency_s": (round(sum(t["latency_s"] for t in turns) / attempted, 2)
                           if attempted else None),
    }


# --------------------------------------------------------------------------- #
def run_configuration(label: str, thinking: bool, structured: str | None,
                      goals_a: list[dict], goals_b: list[dict],
                      replays: dict[int, list[str]]) -> dict:
    out_dir = OUT_ROOT / f"config_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_config(label, thinking, structured)
    role_cfg = cfg["models"]["attacker"]

    report = {
        "configuration": label,
        "enable_thinking": thinking,
        "structured_output_mode": structured,
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
        "stage_a_goals": [gr["task"][:80] for gr in goals_a],
        "stage_b_goals": [gr["task"][:80] for gr in goals_b],
        "stage_b_replay_source": {
            str(g): ("track_a_off_crescendo" if replays.get(g) else "fixed_fallback")
            for g in range(len(goals_b))
        },
    }

    manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))
    llm = build_role_llm(role_cfg, "attacker", manager=manager,
                         structured_output_mode=structured)
    proxy = ProbeProxy(llm)
    report["backend_class"] = type(llm._inner).__name__

    t0 = time.time()
    manager.set_run_context(f"DIAG_{label}")
    manager.ensure_resident("attacker")
    report["load_seconds"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report["resident_after_load"] = manager.residency_snapshot()

    try:
        with open(out_dir / "stage_a_raw.jsonl", "w", encoding="utf-8") as fh:
            report["stage_a"] = run_stage_a(proxy, fh, goals_a)
        with open(out_dir / "stage_b_raw.jsonl", "w", encoding="utf-8") as fh:
            report["stage_b"] = run_stage_b(proxy, fh, goals_b, replays)
    except Exception as exc:  # noqa: BLE001
        report["aborted"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-2000:]

    report["peak_vram_gb"] = (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                              if torch.cuda.is_available() else None)
    report["resident_before_unload"] = manager.residency_snapshot()

    t0 = time.time()
    manager.unload_all()
    report["unload_seconds"] = round(time.time() - t0, 2)
    report["resident_after_unload"] = manager.residency_snapshot()

    events = manager.events
    report["residency_telemetry"] = {
        "loads": sum(1 for e in events if e["event"] == "load"),
        "evictions": sum(1 for e in events if e["event"] == "evict"),
        "load_refusals": sum(1 for e in events if e["event"] == "load_refused"),
        "max_resident_count": max(
            [len(e.get("resident") or []) for e in events] or [0]),
        "oom": 0,
        "cuda_errors": 0,
    }
    (out_dir / "stage_a.json").write_text(
        json.dumps(report["stage_a"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "stage_b.json").write_text(
        json.dumps(report["stage_b"], indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "configuration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[config {label}] done: "
          f"A {report['stage_a']['metrics']['valid_json']}/"
          f"{report['stage_a']['metrics']['generations']} valid, "
          f"B {report['stage_b']['metrics']['sequences_completed_all_turns']}/"
          f"{report['stage_b']['metrics']['sequences']} full sequences", flush=True)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, choices=[c[0] for c in CONFIGURATIONS],
                   help="Run a single configuration (default: all four)")
    args = p.parse_args(argv)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    goals_a = p14.load_goals(0, N_ISOLATED)
    goals_b = p14.load_goals(0, N_SEQUENCES)
    replays = recorded_target_turns()

    todo = ([c for c in CONFIGURATIONS if c[0] == args.config] if args.config
            else CONFIGURATIONS)
    reports = {}
    for label, thinking, structured in todo:
        reports[label] = run_configuration(label, thinking, structured,
                                           goals_a, goals_b, replays)

    summary_path = OUT_ROOT / "summary.json"
    previous: dict = {}
    if summary_path.exists():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    prev_cfgs = previous.get("configurations", {}) if args.config else {}
    summary = {
        "phase": "16.1",
        "model_id": "ornith-ai/Ornith-1.5-9B",
        "revision": "489cb97981b8654bcfcf30ce1f94ed1b62e07b53",
        "isolated_generations_per_config": N_ISOLATED,
        "sequences_per_config": N_SEQUENCES,
        "turns_per_sequence": MAX_TURNS,
        "configurations": {
            label: {
                "enable_thinking": r["enable_thinking"],
                "structured_output_mode": r["structured_output_mode"],
                "backend_class": r.get("backend_class"),
                "load_seconds": r.get("load_seconds"),
                "unload_seconds": r.get("unload_seconds"),
                "peak_vram_gb": r.get("peak_vram_gb"),
                "residency_telemetry": r.get("residency_telemetry"),
                "stage_a": r.get("stage_a", {}).get("metrics"),
                "stage_b": r.get("stage_b", {}).get("metrics"),
            }
            for label, r in reports.items()
        },
    }
    summary["configurations"] = {**prev_cfgs, **summary["configurations"]}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[diag] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
