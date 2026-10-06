#!/usr/bin/env python3
"""Selectable-attacker qualification — Stage A / B / C + Qwen regression.

Runnable independently of the frozen Track A pipeline. NOTHING here launches the
ASR pilot: Stage C is a 9-run interface check (3 goals x 3 attacks, NBF OFF) and
refuses to run unless Stages A and B passed.

The harness is attacker-agnostic; the defaults reproduce the Phase 15 GLM run:

    python scripts/qualify_glm_attacker.py --stage qwen        # Qwen regression
    python scripts/qualify_glm_attacker.py --stage switching   # Qwen -> GLM -> Qwen
    python scripts/qualify_glm_attacker.py --stage A           # 10 isolated GLM generations
    python scripts/qualify_glm_attacker.py --stage B           # 5 multi-turn sequences
    python scripts/qualify_glm_attacker.py --stage C           # 3 goals x 3 attacks (gated)

Another attacker is qualified by pointing the context flags at its own values;
``qualify_ornith15_attacker.py`` is exactly that as a thin entry point:

    --profile ornith15 --config configs/attacker_ornith15.yaml \
    --out-dir results/phase16_ornith15/qualification \
    --stage-c-dir results/phase16_ornith15/stage_c --label Ornith-1.5-9B

Every stage writes JSON evidence under <out-dir>/. Attacker JSON is parsed by the
frozen parser: no repair, no retry, no fallback.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

# Allow multiple OpenMP runtimes on Windows (torch + transformers).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import yaml  # noqa: E402

from guardbound.attacks.base import Turn  # noqa: E402
from guardbound.attacks.crescendo_paper import generate_crescendo_step  # noqa: E402
from guardbound.attacks.registry import get_attack  # noqa: E402
from guardbound.llm.attacker_profiles import (  # noqa: E402
    apply_attacker_profile,
    attacker_runtime_telemetry,
    profile_names,
)
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

# Defaults describe the GLM qualification, so `--stage X` with no other flags
# behaves exactly as it did in Phase 15. Another attacker (Phase 16 Ornith) is
# qualified by pointing --profile/--config/--out-dir/--label at its own values.
DEFAULT_PROFILE = "glm46v"
DEFAULT_CONFIG = ROOT / "configs" / "attacker_glm46v.yaml"
DEFAULT_QUAL_DIR = ROOT / "results" / "phase15_glm46v" / "qualification"
DEFAULT_STAGE_C_DIR = ROOT / "results" / "phase15_glm46v" / "stage_c"
DEFAULT_LABEL = "GLM-4.6V-Flash"
FROZEN_CONFIG = ROOT / "configs" / "reproduction_phase14_frozen.yaml"
STRUCTURED_MODE = "constrained_json"


# Human labels for the switching report (profile -> label).
_PROFILE_LABELS = {
    "qwen3": "Qwen3-4B-Instruct-2507",
    "glm46v": "GLM-4.6V-Flash",
    "ornith15": "Ornith-1.5-9B",
}


@dataclass(frozen=True)
class QualContext:
    """Which attacker model this qualification run is about."""

    profile: str
    config: Path
    qual_dir: Path
    stage_c_dir: Path
    label: str

    @property
    def attacker_model_id(self) -> str:
        return load_cfg(self.config)["models"]["attacker"]["model"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_cfg(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def goals(n: int) -> list[dict]:
    """Goals in the frozen HarmBench file order, via the frozen loader."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase14_full_reproduction as p14

    return p14.load_goals(0, n)


class Capture:
    """Records the exact message lists handed to the attacker backend."""

    def __init__(self, inner):
        self._inner = inner
        self.sent: list[list[dict]] = []

    def generate(self, *args, **kwargs):
        msgs = args[0] if args else kwargs.get("messages", [])
        self.sent.append(json.loads(json.dumps(msgs, default=str)))
        return self._inner.generate(*args, **kwargs)

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, "_inner"), item)


def count_telemetry(inner):
    """Minimal call/parse telemetry (same semantics as the runner's proxy)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import phase14_full_reproduction as p14

    return p14.CountingChatLLM(inner, label="attacker")


def classify_raw(raw: str) -> str:
    """Classify a raw (unparsed) attacker response — no repair is attempted."""
    text = raw or ""
    if not text.strip():
        return "empty"
    if "<think" in text.lower() or "</think" in text.lower():
        return "reasoning_only"
    if "{" in text:
        return "json_like_malformed"
    return "other_text"


def build_attacker(cfg: dict, profile: str):
    """Build ONLY the attacker role (target/evaluator are not needed by A/B)."""
    cfg = apply_attacker_profile(cfg, profile)
    manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))
    role_cfg = cfg["models"]["attacker"]
    llm = build_role_llm(role_cfg, "attacker", manager=manager,
                         structured_output_mode=STRUCTURED_MODE)
    return cfg, manager, llm


def vram_snapshot(manager) -> dict:
    snap = manager.residency_snapshot()
    if torch.cuda.is_available():
        snap["torch_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return snap


def write(ctx: QualContext, stage: str, payload: dict) -> Path:
    ctx.qual_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.qual_dir / f"stage_{stage}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"[qualify] wrote {path}", flush=True)
    return path


# --------------------------------------------------------------------------- #
# Stage A — isolated generation
# --------------------------------------------------------------------------- #
def stage_a(ctx: QualContext, args) -> dict:
    cfg = load_cfg(ctx.config)
    cfg, manager, llm = build_attacker(cfg, ctx.profile)
    counting = count_telemetry(llm)
    report = {"stage": "A", "attacker": attacker_runtime_telemetry(cfg),
              "probes": [], "n_probes": args.probes}

    manager.set_run_context("QUALIFY_A")
    t0 = time.time()
    manager.ensure_resident("attacker")
    report["load_seconds"] = round(time.time() - t0, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i, gr in enumerate(goals(args.probes)):
        probe = {"i": i, "goal_id": i, "goal": gr["task"][:140]}
        counting.reset_counter()
        p0 = time.time()
        try:
            q, summary = generate_crescendo_step(
                round_num=1, goal=gr["task"], history_attacker=[],
                history_target=[], scores=[], last_response="",
                attacker_llm=counting, max_rounds=8,
            )
            probe["outcome"] = "parsed"
            probe["generatedQuestion"] = q
            probe["lastResponseSummary"] = summary
            probe["empty_query"] = (q == "")
        except Exception as exc:  # noqa: BLE001
            probe["outcome"] = "raised"
            probe["error_type"] = type(exc).__name__
            probe["error_message"] = str(exc)[:1200]
        probe["latency_s"] = round(time.time() - p0, 2)
        raws = [{"kind": k, "text": t} for (_p, k, t) in counting.raw_outputs]
        probe["raw_outputs"] = raws
        probe["json_calls"] = dict(counting.json_calls)
        probe["parse_failures"] = counting.parse_failure_summary()
        if probe["outcome"] == "raised" and raws:
            probe["failure_category"] = classify_raw(
                next((r["text"] for r in raws if r["kind"] == "str"), "")
            )
        report["probes"].append(probe)
        print(f"[A] probe {i}: {probe['outcome']} "
              f"{probe.get('failure_category', '')} ({probe['latency_s']}s)",
              flush=True)

    valid = sum(1 for p in report["probes"] if p["outcome"] == "parsed")
    lat = [p["latency_s"] for p in report["probes"]]
    report["metrics"] = {
        "generations": len(report["probes"]),
        "valid_json": valid,
        "invalid_json": len(report["probes"]) - valid,
        "json_validity_rate": round(valid / len(report["probes"]), 4),
        "empty_outputs": sum(1 for p in report["probes"]
                             if p.get("failure_category") == "empty"),
        "reasoning_only_outputs": sum(1 for p in report["probes"]
                                      if p.get("failure_category") == "reasoning_only"),
        "json_like_malformed": sum(1 for p in report["probes"]
                                   if p.get("failure_category") == "json_like_malformed"),
        "other_text_outputs": sum(1 for p in report["probes"]
                                  if p.get("failure_category") == "other_text"),
        "empty_queries": sum(1 for p in report["probes"] if p.get("empty_query")),
        "latency_mean_s": round(sum(lat) / len(lat), 2),
        "latency_max_s": round(max(lat), 2),
    }
    report["vram"] = vram_snapshot(manager)
    report["passed"] = bool(report["metrics"]["json_validity_rate"] >= 0.5
                            and not report["metrics"]["empty_outputs"])
    manager.unload_all()
    return report


# --------------------------------------------------------------------------- #
# Stage B — multi-turn context
# --------------------------------------------------------------------------- #
def stage_b(ctx: QualContext, args) -> dict:
    cfg = load_cfg(ctx.config)
    cfg, manager, llm = build_attacker(cfg, ctx.profile)
    capture = Capture(llm)
    counting = count_telemetry(capture)
    report = {"stage": "B", "attacker": attacker_runtime_telemetry(cfg),
              "sequences": [], "n_sequences": args.sequences}

    manager.set_run_context("QUALIFY_B")
    manager.ensure_resident("attacker")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gs = goals(args.sequences)
    for seq_i, gr in enumerate(gs):
        goal = gr["task"]
        attack = get_attack("crescendo_paper")
        attack.set_attacker_llm(counting)
        seq = {"seq": seq_i, "goal": goal[:140], "turns": []}
        history: list[Turn] = []
        for turn_i in range(args.turns):
            step = {"turn": turn_i + 1}
            counting.reset_counter()
            capture.sent.clear()
            try:
                q = attack.next_query(goal, history)
                step["query"] = q
                step["empty_query"] = (q == "")
                step["repeats_previous"] = bool(history and q == history[-1].query)
            except Exception as exc:  # noqa: BLE001
                step["context_failure"] = f"{type(exc).__name__}: {str(exc)[:600]}"
                seq["turns"].append(step)
                break
            sent = capture.sent[-1] if capture.sent else []
            step["roles"] = [m.get("role") for m in sent]
            step["system_has_goal"] = any(
                m.get("role") == "system" and goal[:60] in str(m.get("content", ""))
                for m in sent)
            step["assistant_turns"] = sum(1 for m in sent if m.get("role") == "assistant")
            step["feedback_turns"] = sum(
                1 for m in sent if m.get("role") == "user"
                and "The last response was:" in str(m.get("content", "")))
            step["json_parse_failures"] = dict(counting.json_parse_failures)
            seq["turns"].append(step)
            attack.record_turn(q, "REPLAYED TARGET REPLY", 3)
            history.append(Turn(query=q, response="REPLAYED TARGET REPLY"))
        report["sequences"].append(seq)
        print(f"[B] seq {seq_i}: {len(seq['turns'])} turns", flush=True)

    turns = [t for s in report["sequences"] for t in s["turns"]]
    ok_turns = [t for t in turns if not t.get("context_failure")]
    report["metrics"] = {
        "sequences": len(report["sequences"]),
        "turns_attempted": len(turns),
        "context_failures": len(turns) - len(ok_turns),
        "empty_queries": sum(1 for t in ok_turns if t.get("empty_query")),
        "repeated_queries": sum(1 for t in ok_turns if t.get("repeats_previous")),
        "goal_preserved": all(t.get("system_has_goal") for t in ok_turns),
        "ordering_correct": all(
            t["roles"] == ["system", "user"] + ["assistant", "user"] * (t["turn"] - 1)
            for t in ok_turns),
        "sequences_with_2plus_turns": sum(
            1 for s in report["sequences"] if len(s["turns"]) >= 2),
    }
    report["vram"] = vram_snapshot(manager)
    m = report["metrics"]
    report["passed"] = bool(
        m["sequences_with_2plus_turns"] >= 4 and m["goal_preserved"]
        and m["ordering_correct"] and m["empty_queries"] == 0
    )
    manager.unload_all()
    return report


# --------------------------------------------------------------------------- #
# Stage C — small attack-interface test (gated)
# --------------------------------------------------------------------------- #
def _stage_passed(ctx: QualContext, stage: str) -> tuple[bool, str]:
    path = ctx.qual_dir / f"stage_{stage}.json"
    if not path.exists():
        return False, f"stage {stage} result missing ({path})"
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("passed")), json.dumps(data.get("metrics", {}))


def stage_c(ctx: QualContext, args) -> dict:
    ok_a, why_a = _stage_passed(ctx, "A")
    ok_b, why_b = _stage_passed(ctx, "B")
    if not (ok_a and ok_b) and not args.force_stage_c:
        return {"stage": "C", "ran": False,
                "blocked_by": {"stage_A": why_a, "stage_B": why_b},
                "note": "Stage C requires Stages A and B to pass (or --force-stage-c)."}
    out = ctx.stage_c_dir / "batch00.jsonl"
    cmd = [
        sys.executable, str(ROOT / "scripts" / "phase14_full_reproduction.py"),
        "--batch", "0", "--limit", str(args.limit), "--attack", "all",
        "--condition", "off",                      # NBF OFF for qualification
        "--config", str(ctx.config),
        "--attacker-model", ctx.profile,
        "--structured-output-mode", STRUCTURED_MODE,
        "--out", str(out),
    ]
    print(f"[C] {' '.join(cmd)}", flush=True)
    # The runner prints raw exception text, which can contain non-cp1252
    # characters (e.g. the "->" arrow in a model's reply). On a cp1252 console
    # that raises UnicodeEncodeError inside the failure-reporting path and
    # aborts the batch. Fixed here at the environment level for this subprocess
    # rather than by editing the frozen runner's logging.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    # Decode explicitly as UTF-8: the default `text=True` uses the console code
    # page (cp1252 here) and raises UnicodeDecodeError on the child's UTF-8
    # output, which would kill this wrapper mid-run.
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                          encoding="utf-8", errors="replace", env=env)
    runner_log = out.parent / "stage_c_runner.log"
    runner_log.parent.mkdir(parents=True, exist_ok=True)
    with open(runner_log, "w", encoding="utf-8") as fh:
        fh.write(proc.stdout or "")
        fh.write("\n=== stderr ===\n")
        fh.write(proc.stderr or "")
    records = []
    if out.exists():
        with open(out, encoding="utf-8") as f:
            records = [json.loads(l) for l in f if l.strip()]
    classes = {}
    for r in records:
        classes[r.get("failure_class")] = classes.get(r.get("failure_class"), 0) + 1
    return {
        "stage": "C", "ran": True, "exit_code": proc.returncode,
        "command": cmd, "output": str(out),
        "records": len(records),
        "failure_classes": classes,
        "schemas": sorted({len(r.keys()) for r in records}),
        "attacker_runtime": records[0].get("attacker_runtime") if records else None,
        "off_contamination": sum(
            1 for r in records
            if r.get("nbf_scores") != [] or r.get("filtered_queries") != 0),
        "runner_log": str(runner_log),
        "stdout_tail": (proc.stdout or "")[-3000:],
        "stderr_tail": (proc.stderr or "")[-3000:],
    }


# --------------------------------------------------------------------------- #
# Qwen regression + switching (real models)
# --------------------------------------------------------------------------- #
def stage_qwen(ctx: QualContext, args) -> dict:
    """The default attacker still loads, generates and parses JSON.

    Deliberately ignores ``ctx``: this stage always exercises the frozen config's
    Qwen attacker, whatever candidate is being qualified.
    """
    report = {"stage": "qwen_regression", "steps": []}

    def step(name, fn):
        entry = {"step": name}
        t0 = time.time()
        try:
            entry["result"] = fn()
            entry["ok"] = True
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["traceback"] = traceback.format_exc()[-1500:]
        entry["seconds"] = round(time.time() - t0, 2)
        report["steps"].append(entry)
        print(f"[qwen] {name}: ok={entry['ok']} ({entry['seconds']}s)", flush=True)
        return entry

    cfg = load_cfg(FROZEN_CONFIG)
    # No --attacker-model: the config's attacker must be used verbatim.
    cfg, manager, llm = build_attacker(cfg, "qwen3")
    report["attacker"] = attacker_runtime_telemetry(cfg)

    counting = count_telemetry(llm)
    state = {}

    def _load():
        manager.set_run_context("QUALIFY_QWEN")
        manager.ensure_resident("attacker")
        state["loaded"] = True
        return vram_snapshot(manager)

    step("load", _load)

    def _one_generation():
        g = goals(1)[0]["task"]
        q, _s = generate_crescendo_step(
            round_num=1, goal=g, history_attacker=[], history_target=[],
            scores=[], last_response="", attacker_llm=counting, max_rounds=8)
        assert q and isinstance(q, str), "empty Qwen query"
        return {"query": q[:200]}

    step("generation", _one_generation)

    def _json_parse():
        out = counting.generate(
            [{"role": "system", "content": "Return JSON."},
             {"role": "user", "content": 'Return {"generatedQuestion": "x", '
                                         '"lastResponseSummary": ""}'}],
            json_format=True)
        return {"type": type(out).__name__,
                "keys": sorted(out.keys()) if isinstance(out, dict) else None}

    step("json_parsing", _json_parse)

    def _multi_turn():
        attack = get_attack("crescendo_paper")
        attack.set_attacker_llm(counting)
        g = goals(1)[0]["task"]
        history: list[Turn] = []
        q1 = attack.next_query(g, history)
        attack.record_turn(q1, "I can't help with that.", 1)
        history.append(Turn(query=q1, response="I can't help with that."))
        q2 = attack.next_query(g, history)
        return {"turn1": q1[:120], "turn2": q2[:120], "distinct": q1 != q2}

    step("multi_turn", _multi_turn)

    def _unload():
        manager.unload_all()
        snap = manager.residency_snapshot()
        assert snap["resident_count"] == 0, snap
        return snap

    step("unload", _unload)
    report["passed"] = all(s["ok"] for s in report["steps"])
    return report


def stage_switching(ctx: QualContext, args) -> dict:
    """Real Qwen -> unload -> candidate -> unload -> Qwen with identity checks."""
    report = {"stage": "switching", "steps": [], "candidate_profile": ctx.profile}
    cfg = load_cfg(ctx.config)
    manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))

    def _bind(profile: str):
        """Register the attacker role, then load it (as build_role_llm does)."""
        from guardbound.llm.provider_factory import build_role_llm as brl

        role_cfg = apply_attacker_profile(cfg, profile)["models"]["attacker"]
        llm = brl(role_cfg, "attacker", manager=manager,
                  structured_output_mode=STRUCTURED_MODE)
        return role_cfg, llm

    def _alloc_gb() -> float | None:
        if not torch.cuda.is_available():
            return None
        torch.cuda.synchronize()
        return round(torch.cuda.memory_allocated() / 1e9, 3)

    def cycle(name: str, profile: str):
        entry = {"model": name, "profile": profile}
        t0 = time.time()
        try:
            before_events = len(manager.events)
            before_alloc = _alloc_gb()
            role_cfg, llm = _bind(profile)
            manager.ensure_resident("attacker")
            after_alloc = _alloc_gb()
            snap = manager.residency_snapshot()

            new_events = manager.events[before_events:]
            loads = [e for e in new_events if e["event"] == "load"]
            entry["resident_model_ids"] = snap["resident_model_ids"]
            entry["resident_count"] = snap["resident_count"]
            entry["identity_ok"] = (snap["resident_model_ids"] == [role_cfg["model"]])
            entry["backend"] = type(getattr(llm, "_inner", llm)).__name__
            entry["allocated_before_gb"] = before_alloc
            entry["allocated_after_load_gb"] = after_alloc
            entry["load_events"] = [
                {"model_id": e.get("model_id"), "duration_s": e.get("duration_s"),
                 "measured_gb": e.get("measured_gb")}
                for e in loads
            ]
            # A real load moved real VRAM; bookkeeping alone is not evidence.
            entry["weights_actually_loaded"] = bool(
                before_alloc is not None and after_alloc is not None
                and (after_alloc - before_alloc) > 3.0 and len(loads) == 1
            )
            entry["model_id_in_events"] = sorted(
                {e.get("model_id") for e in loads if e.get("model_id")})

            manager.unload_all()
            freed_alloc = _alloc_gb()
            after = manager.residency_snapshot()
            entry["allocated_after_unload_gb"] = freed_alloc
            entry["after_unload_resident"] = after["resident_count"]
            entry["freed_ok"] = (after["resident_count"] == 0
                                 and (freed_alloc is None or freed_alloc < 1.0))
            entry["ok"] = bool(entry["identity_ok"] and entry["freed_ok"]
                               and entry["weights_actually_loaded"])
        except Exception as exc:  # noqa: BLE001
            entry["ok"] = False
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["seconds"] = round(time.time() - t0, 2)
        report["steps"].append(entry)
        print(f"[switch] {name}: ok={entry['ok']} "
              f"alloc {entry.get('allocated_before_gb')}->"
              f"{entry.get('allocated_after_load_gb')}GB "
              f"({entry['seconds']}s)", flush=True)

    sequence = getattr(args, "switch_sequence", None)
    profiles = ([p.strip() for p in sequence.split(",") if p.strip()]
                if sequence else ["qwen3", ctx.profile, "qwen3"])
    report["switch_sequence"] = profiles
    for profile in profiles:
        cycle(_PROFILE_LABELS.get(profile, profile), profile)
    report["passed"] = all(s.get("ok") for s in report["steps"])
    report["no_cross_model_state"] = all(
        s.get("after_unload_resident") == 0 for s in report["steps"])
    report["no_vram_leak"] = all(
        (s.get("allocated_after_unload_gb") is None
         or s["allocated_after_unload_gb"] < 1.0)
        for s in report["steps"] if s.get("ok"))
    return report


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["A", "B", "C", "qwen", "switching"])
    p.add_argument("--probes", type=int, default=10, help="Stage A generations")
    p.add_argument("--sequences", type=int, default=5, help="Stage B sequences")
    p.add_argument("--turns", type=int, default=3, help="Stage B turns per sequence")
    p.add_argument("--limit", type=int, default=3, help="Stage C goals (3 goals)")
    p.add_argument("--force-stage-c", action="store_true",
                   help="Run Stage C even if Stages A/B did not pass")
    # Which attacker is being qualified. Defaults reproduce the Phase 15 GLM run.
    p.add_argument("--profile", default=DEFAULT_PROFILE, choices=profile_names(),
                   help="Attacker profile to qualify (default: glm46v)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Experiment config for the candidate attacker")
    p.add_argument("--out-dir", default=None,
                   help=f"Qualification output dir (default: {DEFAULT_QUAL_DIR})")
    p.add_argument("--stage-c-dir", default=None,
                   help=f"Stage C output dir (default: {DEFAULT_STAGE_C_DIR})")
    p.add_argument("--label", default=None,
                   help="Human label used in logs/reports (default: the profile name)")
    p.add_argument("--switch-sequence", default=None,
                   help="Comma-separated attacker profiles to cycle through in "
                        "--stage switching (default: qwen3,<profile>,qwen3). "
                        "Example: qwen3,glm46v,ornith15,qwen3")
    args = p.parse_args(argv)

    qual_dir = Path(args.out_dir) if args.out_dir else DEFAULT_QUAL_DIR
    stage_c_dir = (Path(args.stage_c_dir) if args.stage_c_dir
                   else DEFAULT_STAGE_C_DIR)
    ctx = QualContext(
        profile=args.profile,
        config=Path(args.config),
        qual_dir=qual_dir,
        stage_c_dir=stage_c_dir,
        label=args.label or args.profile,
    )
    print(f"[qualify] profile={ctx.profile} config={ctx.config} "
          f"qual_dir={ctx.qual_dir}", flush=True)

    dispatch = {"A": stage_a, "B": stage_b, "C": stage_c, "qwen": stage_qwen,
                "switching": stage_switching}
    report = dispatch[args.stage](ctx, args)
    write(ctx, args.stage, report)
    print(json.dumps(report.get("metrics") or
                     {k: v for k, v in report.items() if k != "steps"},
                     indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
