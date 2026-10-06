#!/usr/bin/env python3
"""Phase 16.5 §16/§17 — structured-output diagnostic (unconstrained vs native constrained).

Paired, attack-step-only diagnostic with the Ornith attacker. No attack is run, no
ASR is computed. Both conditions use the SAME prompts, goals, seeds, temperature,
attacker token budget, revision and message construction; the only variable is the
decoding mode:

    A = current unconstrained generation     (structured_output_mode = None)
    B = native constrained JSON generation   (structured_output_mode = "constrained_json",
       i.e. the project's existing Phase 14.1 JSONGrammarLogitsProcessor + `}` stopping
       criterion — the backend's own mechanism, not an ad-hoc grammar engine)

Nothing is repaired, retried or reformatted; the frozen parser decides validity.

    python scripts/phase16_5_structured_output.py --per-attack 7
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402

import phase14_full_reproduction as p14  # noqa: E402
from guardbound.attacks.acronym import generate_acronym_step  # noqa: E402
from guardbound.attacks.crescendo_paper import (  # noqa: E402
    AttackGenerationError,
    generate_crescendo_step,
)
from guardbound.attacks.opposite_day import generate_opposite_day_step  # noqa: E402
from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.provider_factory import build_role_llm  # noqa: E402

OUT = ROOT / "results/phase16_5_forensics"
ORNITH_CFG = ROOT / "configs/attacker_ornith15.yaml"

STEP_FNS = {
    "crescendo_paper": lambda goal, llm: generate_crescendo_step(
        round_num=1, goal=goal, history_attacker=[], history_target=[],
        scores=[], last_response="", attacker_llm=llm, max_rounds=8),
    "opposite_day": lambda goal, llm: generate_opposite_day_step(
        round_num=1, goal=goal, history_attacker=[], history_target=[],
        scores=[], last_response="", attacker_llm=llm, max_rounds=8),
    "acronym": lambda goal, llm: generate_acronym_step(
        round_num=1, goal=goal, history_attacker=[], history_target=[],
        scores=[], last_response="", attacker_llm=llm, max_rounds=8),
}


class Proxy:
    """Records the raw backend return; same semantics as the runner's proxy."""

    def __init__(self, inner):
        self._inner = inner
        self.raw: list[tuple[str, str]] = []

    def generate(self, *a, **k):
        out = self._inner.generate(*a, **k)
        self.raw.append((type(out).__name__,
                         out if isinstance(out, str)
                         else json.dumps(out, ensure_ascii=False, default=str)))
        return out

    def reset(self):
        self.raw = []

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(object.__getattribute__(self, "_inner"), item)


def classify(raw: str) -> str:
    import re
    t = (raw or "").strip()
    if not t:
        return "EMPTY_OUTPUT"
    if any(m in t.lower() for m in ("<think", "</think")):
        return "REASONING_ONLY"
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        if "generatedQuestion" in obj and "lastResponseSummary" in obj:
            return "VALID_JSON"
        return "MISSING_REQUIRED_FIELD"
    if not t.endswith("}") and "{" in t:
        return "TRUNCATED_JSON"
    if re.search(r",\s*\}", t):
        return "TRAILING_COMMA"
    if t.startswith("```") or "\n```" in t:
        return "MARKDOWN_FENCED_JSON"
    first = t.find("{")
    if first > 0 and t[:first].strip():
        return "EXTRA_TEXT"
    if "{" not in t:
        return "NATURAL_LANGUAGE"
    return "MALFORMED_SYNTAX"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-attack", type=int, default=7)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(ORNITH_CFG.read_text(encoding="utf-8"))
    goals = p14.load_goals(0, 30)
    plan = []
    for atk in STEP_FNS:
        for i in range(args.per_attack):
            plan.append((atk, i, goals[i]["task"]))

    report = {
        "phase": "16.5", "kind": "structured_output_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attacker": cfg["models"]["attacker"]["model"],
        "revision": cfg["models"]["attacker"].get("revision"),
        "attacker_max_new_tokens": cfg["models"]["attacker"].get("max_new_tokens"),
        "temperature": cfg["models"]["attacker"].get("temperature"),
        "generations_per_condition": len(plan),
        "constrained_mechanism": (
            "existing Phase 14.1 JSONGrammarLogitsProcessor + `}` stopping criterion "
            "(json_constrained_decoder.build_json_logits_processor); no ad-hoc grammar "
            "engine, no repair, no retry"),
        "conditions": {},
    }

    for label, mode in (("A_unconstrained", None),
                        ("B_native_constrained", "constrained_json")):
        manager = ModelManager(device=cfg.get("hardware", {}).get("device", "cuda"))
        role_cfg = dict(cfg["models"]["attacker"])
        if mode is None:
            role_cfg.pop("structured_output_mode", None)
        else:
            role_cfg["structured_output_mode"] = mode
        llm = build_role_llm(role_cfg, "attacker", manager=manager,
                             structured_output_mode=mode)
        proxy = Proxy(llm)
        t0 = time.time()
        manager.set_run_context(f"P16_5_{label}")
        manager.ensure_resident("attacker")
        load_s = round(time.time() - t0, 2)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        rows = []
        for atk, goal_idx, goal in plan:
            p14._seed_all(p14.derive_seed(goal_idx, "crescendo", "off"))
            proxy.reset()
            t0 = time.time()
            try:
                q, s = STEP_FNS[atk](goal, proxy)
                outcome = "parsed"
                parsed = {"generatedQuestion": q, "lastResponseSummary": s}
            except AttackGenerationError as exc:
                outcome = "AttackGenerationError"
                parsed = None
                err = str(exc)[:400]
            except Exception as exc:  # noqa: BLE001
                outcome = type(exc).__name__
                parsed = None
                err = str(exc)[:400]
            raw = proxy.raw[-1][1] if proxy.raw else ""
            cls = classify(raw)
            # A parsed dict missing a required key is a field failure, not syntax
            if outcome == "parsed" and cls != "VALID_JSON":
                cls = cls if cls != "VALID_JSON" else "VALID_JSON"
            rows.append({"attack": atk, "goal_id": goal_idx, "outcome": outcome,
                         "class": cls, "raw_kind": proxy.raw[-1][0] if proxy.raw else None,
                         "raw": raw,
                         "error": (err if outcome != "parsed" else None),
                         "latency_s": round(time.time() - t0, 2),
                         "query_empty": bool(outcome == "parsed" and parsed
                                             and parsed["generatedQuestion"] == "")})
            print(f"[{label}] {atk} goal{goal_idx}: {outcome} class={cls} "
                  f"({rows[-1]['latency_s']}s)", flush=True)

        valid = sum(1 for r in rows if r["class"] == "VALID_JSON")
        by_attack = {}
        for atk in STEP_FNS:
            sub = [r for r in rows if r["attack"] == atk]
            by_attack[atk] = {
                "n": len(sub),
                "valid": sum(1 for r in sub if r["class"] == "VALID_JSON"),
                "classes": {c: sum(1 for r in sub if r["class"] == c)
                            for c in sorted({r["class"] for r in sub})}}
        report["conditions"][label] = {
            "structured_output_mode": mode,
            "load_seconds": load_s,
            "peak_vram_gb": (round(torch.cuda.max_memory_allocated() / 1e9, 3)
                             if torch.cuda.is_available() else None),
            "valid_json": valid,
            "n": len(rows),
            "valid_rate": round(valid / len(rows), 4),
            "classes": {c: sum(1 for r in rows if r["class"] == c)
                        for c in sorted({r["class"] for r in rows})},
            "by_attack": by_attack,
            "mean_latency_s": round(statistics.mean(r["latency_s"] for r in rows), 2),
            "residency": manager.residency_snapshot(),
            "rows": rows,
        }
        manager.unload_all()

    (OUT / "structured_output_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote", OUT / "structured_output_comparison.json")
    for label, d in report["conditions"].items():
        print(f"{label}: {d['valid_json']}/{d['n']} valid "
              f"({100*d['valid_rate']:.1f}%) classes={d['classes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
