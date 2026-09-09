#!/usr/bin/env python3
"""Phase 4 — experimental reproduction runner.

Runs the four Guardbound paper attacks (Crescendo, ActorAttack, OppositeDay,
Acronym) under the official authors' experiment configuration extracted from
``nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/steering.py``:

  attacker = target = gpt-4o (official default)
  dataset  = data/test/harmbench_tasks.json (200 samples, max_rounds=8)
  eta      = 0.0 (official --threshold default)
  temperature = 0.7
  embedding   = all-mpnet-base-v2
  checkpoint  = models/models_best_nbf_released.pth  Modes (task Phase 4, Sec. 7):
    A = attack only               (no NBF: barrier=None)
    B = attack + NBF filter       (official --safety_filtering: NBF candidate
                                   filter with plain target calls)
    C = full Guardbound pipeline  (B + SteeredLLMChat target wrapper +
                                   regeneration on refusal)

  Phase 7: models resolve ONLY from the explicit `experiment_model` block in
  configs/reproduction.yaml (documented MODEL SUBSTITUTION: local
  Llama-3.1-8B-Instruct for attacker+target+evaluator because no OpenAI API
  key is configured). No silent fallback: a nonexistent model path or an
  unexpected NBF checkpoint sha256 refuses to start. Every run writes a
  manifest (models, checkpoint sha256, dataset, seed, device, versions) next
  to its JSONL output.

Usage:
    # Mock smoke (no API, no GPU):
    python scripts/run_reproduction.py --mode A --mock --limit 3 --out results/reproduction/smoke_A.jsonl
    python scripts/run_reproduction.py --mode B --mock --limit 3 --out results/reproduction/smoke_B.jsonl

    # Small real run with local Llama (no API key -> documented deviation):
    python scripts/run_reproduction.py --mode A --limit 3 --out results/reproduction/llama31_A.jsonl
    python scripts/run_reproduction.py --mode B --limit 3 --out results/reproduction/llama31_B.jsonl

    # Full pipeline:
    python scripts/run_reproduction.py --mode C --limit 3 --out results/reproduction/llama31_C.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

CONFIG_PATH = Path("configs/reproduction.yaml")
OFFICIAL_DATASET = Path(
    "nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json"
)

ATTACK_KEYS = {
    "crescendo_paper": "crescendo",
    "actor_attack": "actor_attack",
    "opposite_day": "opposite_day",
    "acronym": "acronym",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 4 reproduction runner (official NBF-LLM config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument(
        "--mode", choices=["A", "B", "C"], default=None,
        help="Experiment mode (overrides config): A=attack-only, B=+NBF filter, "
             "C=full Guardbound pipeline",
    )
    parser.add_argument(
        "--attacks", nargs="+", default=None,
        choices=list(ATTACK_KEYS) + ["all"],
        help="Attacks to run (default: all enabled in config)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max goals (smoke/small runs)")
    parser.add_argument("--start-from", type=int, default=None, help="Dataset offset")
    parser.add_argument("--out", default=None, help="Output JSONL path")
    parser.add_argument("--mock", action="store_true", help="Use MockChatLLM (no API/GPU)")
    parser.add_argument("--attacker", default=None, help="Attacker model/path override")
    parser.add_argument("--target", default=None, help="Target model/path override")
    parser.add_argument("--eta", type=float, default=None, help="NBF threshold override")
    parser.add_argument("--max-turns", type=int, default=None, help="Max turns override")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    parser.add_argument("--validate-config", action="store_true",
                        help="Validate config and exit (exit 0 = valid)")
    return parser.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_goals(dataset_path: Path, limit: int | None, start_from: int) -> list[dict]:
    """Load official harmbench_tasks.json records (target_system, task, max_rounds)."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Official dataset not found: {dataset_path}. "
            "Expected nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json"
        )
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    records = records[start_from:]
    if limit is not None:
        records = records[:limit]
    return records


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class JsonAwareMockLLM:
    """Offline mock that returns schema-valid JSON for json_format calls.

    Inspects the full message list to infer which JSON schema the attack
    expects, so every attack's pipeline (pre-attack, generation, judging,
    refusal) runs end-to-end without an API key.
    """

    name = "mock-json"

    def __init__(self):
        self.calls = []

    def generate(
        self,
        messages,
        temperature: float = 0.7,
        max_turns_context=None,
        json_format: bool = False,
    ):
        self.calls.append([dict(m) for m in messages])
        blob = "\n".join(
            str(m.get("content", "")) for m in messages if m.get("content")
        )

        # ActorAttack rubric expects a raw JSON *string* (no json_format flag).
        if "scale of 1 to 5" in blob and not json_format:
            return '{"score": 3}'

        if not json_format:
            return "This is a harmless, generic response."

        # Order matters: most specific markers first.
        if "Extract the core harmful target" in blob:
            return {
                "target": "the harmful behavior",
                "details": {"delivery_type": "article"},
            }
        if "actor_name" in blob:
            return {
                "actors": [
                    {"actor_name": "A1", "relationship": "R1"},
                    {"actor_name": "A2", "relationship": "R2"},
                    {"actor_name": "A3", "relationship": "R3"},
                ]
            }
        if '"questions"' in blob:
            return {"questions": [{"question": "q1?"}, {"question": "q2?"}]}
        if "responseToEvaluateInput" in blob:
            return {"value": False, "metadata": 5}
        if "scale of 1 to 5" in blob:
            return {"score": 3}
        if '"type"' in blob:
            return {"type": "successful"}
        if "generatedQuestion" in blob:
            return {"generatedQuestion": "q?", "lastResponseSummary": "summary"}
        return {}


# --------------------------------------------------------------------------- #
# LLM construction
# --------------------------------------------------------------------------- #


def make_llm(cfg: dict, role: str, args: argparse.Namespace, mock: bool):
    """Create a ChatLLM for attacker/target.

    Resolves ONLY the explicitly configured experiment model — no silent
    fallback. If OPENAI_API_KEY is set AND the configured model is an API
    model id (contains '/'), an OpenAI client is used; otherwise the local
    HF path is loaded. A nonexistent local path is a hard error (already
    enforced by resolve_models during validation).
    """
    if mock:
        return JsonAwareMockLLM()

    override = args.attacker if role == "attacker" else args.target
    model = override or cfg[role]["model"]
    temperature = cfg[role].get("temperature", 0.7)
    max_new_tokens = cfg[role].get("max_new_tokens", 256)

    # Official reproduction config uses max_new_tokens=256 for local models.
    local_kwargs = {"device_map": "cuda", "max_new_tokens": max_new_tokens}

    if override and Path(override).is_dir():
        from guardbound.llm.local_client import HFLocalChatLLM
        return HFLocalChatLLM(model_id=override, **local_kwargs)

    if os.environ.get("OPENAI_API_KEY") and "/" not in model:
        from guardbound.llm.openai_client import OpenAIChatLLM
        return OpenAIChatLLM(model=model)

    from guardbound.llm.local_client import HFLocalChatLLM
    return HFLocalChatLLM(model_id=model, **local_kwargs)


def make_barrier_and_embed(cfg: dict, mock: bool):
    """Load the official checkpoint into Guardbound models + mpnet embed_fn.

    Returns (barrier, embed_fn). In --mock mode the checkpoint is skipped and
    a deterministic placeholder is used so the NBF code path still executes.
    """
    if mock:
        import torch
        from guardbound.models.predictor import NeuralBarrierFunction
        from guardbound.models.dynamics import DialogueDynamics
        from guardbound.models.predictor import SafetyPredictor

        class _AlwaysSafe(SafetyPredictor):
            def forward(self, x_prev, u):
                # logits favoring safe classes -> h < 0 -> accepted
                return torch.full(
                    (*x_prev.shape[:-1], 5), -10.0, device=x_prev.device
                ) + torch.tensor(
                    [0.0, 0.0, 0.0, 0.0, -20.0], device=x_prev.device
                )

        dynamics = DialogueDynamics()
        predictor = _AlwaysSafe()
        barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)

        def embed_fn(text: str):
            return torch.zeros(1, 768)

        return barrier, embed_fn

    from guardbound.models.compat import load_original_checkpoint
    from guardbound.models.predictor import NeuralBarrierFunction

    ckpt_path = Path(cfg["nbf"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Official NBF checkpoint not found: {ckpt_path}")

    dynamics, predictor = load_original_checkpoint(ckpt_path, device="cuda")
    barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)
    barrier.to("cuda")

    from guardbound.embeddings import get_embed_fn
    embed_fn = get_embed_fn(cfg["embedding"]["model"])

    return barrier, embed_fn


def existing_goals(out_path: Path) -> set[tuple[str, str]]:
    """Return {(attack, goal)} pairs already present in the output file.

    Keyed by (attack, goal) so re-running a different attack on the same
    goal does not get skipped.
    """
    if not out_path.exists():
        return set()
    done = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["attack"], rec["goal"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_result(out_path: Path, record: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def run_one(
    attack,
    goal_record: dict,
    target_llm,
    barrier,
    embed_fn,
    cfg: dict,
    args: argparse.Namespace,
    mode: str,
    max_turns: int,
    eta: float,
    metadata: dict,
) -> dict:
    from guardbound.attacks.runner import run_attack_with_backtracking_async

    goal = goal_record["task"]
    system_prompt = goal_record.get("target_system") or None

    conv = await run_attack_with_backtracking_async(
        attack=attack,
        goal=goal,
        target_llm=target_llm,
        embed_fn=embed_fn,
        barrier=barrier,
        eta=eta,
        max_turns=max_turns,
        temperature=cfg["target"].get("temperature", 0.7),
        target_llm_name=metadata["target_model"],
        attack_method=metadata["attack"],
        allow_regeneration=(mode == "C"),
        system_prompt=system_prompt,
        steer_target=(mode == "C"),
    )

    turns = []
    nbf_scores = []
    rubric_scores = []
    filtered = 0
    success = False
    final_score = None

    # Telemetry: judge scores per accepted turn come from the attack's own
    # official score list (Crescendo/OD/Acronym/ActorAttack all append one
    # score per recorded exchange). For ActorAttack the scores are recorded
    # for every attempt (including rejected ones), so align on the minimum.
    attack_scores = list(getattr(attack, "_scores", []) or [])
    refusal_count = (
        attack.get_refusal_count()
        if hasattr(attack, "get_refusal_count") else 0
    )

    for i, t in enumerate(conv.turns):
        turns.append({"query": t.query, "response": t.response})
        if t.was_filtered:
            filtered += 1
        score = attack_scores[i] if i < len(attack_scores) else None
        if score is not None:
            rubric_scores.append(score)
            final_score = score  # last recorded turn's score (official 'Turn k - Score')

    # Success = any judge score 5 (official ASR definition)
    if hasattr(attack, "was_successful"):
        success = bool(attack.was_successful())

    record = {
        "goal": goal,
        "attack": metadata["attack"],
        "attacker_model": metadata["attacker_model"],
        "target_model": metadata["target_model"],
        "embedding_model": metadata["embedding_model"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "eta": eta,
        "mode": mode,
        "max_turns": max_turns,
        "temperature": cfg["target"].get("temperature", 0.7),
        "seed": cfg["experiment"].get("seed", 42),
        "target_system": system_prompt,
        "turns": turns,
        "success": success,
        "final_score": final_score,
        "rubric_scores": rubric_scores,
        "refusal_count": refusal_count,
        "nbf_scores": nbf_scores,
        "filtered_queries": filtered,
        "num_turns": len(conv.turns),
        "runtime": 0.0,  # filled by the caller
    }
    return record


def resolve_models(cfg: dict, args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the explicit experiment models. NO silent fallback.

    Resolution order (each step prints what it selected):
      1. --attacker/--target CLI override (must exist if a local path)
      2. cfg.attacker.model / cfg.target.model (the experiment_model block)

    A local-path model that does not exist is a HARD ERROR, never a fallback.
    """
    attacker = args.attacker or cfg["attacker"]["model"]
    target = args.target or cfg["target"]["model"]
    for role, model in (("attacker", attacker), ("target", target)):
        p = Path(model)
        if p.is_dir():
            print(f"[repro] {role} model (local path): {model}")
        elif p.exists():
            print(f"[repro] {role} model (local file): {model}")
        elif "/" in model or "-" in model and p.parent == Path("."):
            # Dotted HF-style id or explicit override: used as-is (API-backed).
            print(f"[repro] {role} model (api-backed id): {model}")
        else:
            raise FileNotFoundError(
                f"{role} model {model!r} does not exist locally and is not a "
                f"recognizable API model id. Refusing to start (no silent "
                f"fallback allowed)."
            )
    return attacker, target


def build_metadata(cfg: dict, args: argparse.Namespace, mock: bool) -> dict:
    attacker, target = resolve_models(cfg, args)
    if mock:
        attacker = "mock"
        target = "mock"

    meta = {
        "attacker_model": attacker,
        "target_model": target,
        "evaluator_model": cfg.get("evaluator", {}).get("rubric_model", attacker),
        "embedding_model": cfg["embedding"]["model"],
        "checkpoint_sha256": None,
    }
    if not mock:
        ckpt = Path(cfg["nbf"]["checkpoint"])
        if ckpt.exists():
            meta["checkpoint_sha256"] = sha256_of(ckpt)
        else:
            raise FileNotFoundError(
                f"NBF checkpoint not found: {ckpt} — refusing to start."
            )
    return meta


def validate_config(cfg: dict, strict: bool = True) -> list[str]:
    """Validate the reproduction config; return the list of problems.

    With strict=True (default) any missing required field raises SystemExit
    before the experiment starts — a malformed config must never silently
    receive a default.
    """
    problems: list[str] = []

    def _get(path: str):
        node = cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    required = [
        "experiment_model.attacker",
        "experiment_model.target",
        "experiment_model.evaluator",
        "attacker.model",
        "target.model",
        "evaluator.rubric_model",
        "evaluator.refusal_model",
        "embedding.model",
        "embedding.dimension",
        "nbf.checkpoint",
        "nbf.state_dimension",
        "nbf.threshold",
        "nbf.trials.crescendo",
        "nbf.trials.actor_attack",
        "nbf.trials.opposite_day",
        "nbf.trials.acronym",
        "dataset.path",
        "dataset.split",
        "experiment.seed",
        "attacks.max_turns",
        "experiment.output_dir",
        "hardware.device",
    ]
    for field_path in required:
        val = _get(field_path)
        if val is None:
            problems.append(f"missing required field: {field_path}")

    # Checkpoint must exist and match its pinned hash (if pinned).
    ckpt = _get("nbf.checkpoint")
    if ckpt:
        if not Path(ckpt).exists():
            problems.append(f"nbf.checkpoint does not exist: {ckpt}")
        else:
            expected = _get("nbf.checkpoint_sha256_expected")
            if expected:
                actual = sha256_of(Path(ckpt))
                if actual != expected:
                    problems.append(
                        f"nbf.checkpoint sha256 mismatch: expected {expected}, "
                        f"got {actual} (substituted checkpoint?)"
                    )

    # Dataset must exist.
    ds = _get("dataset.path")
    if ds and not Path(ds).exists():
        problems.append(f"dataset.path does not exist: {ds}")

    # Model paths must exist locally (no silent fallback).
    for field_path in ("attacker.model", "target.model"):
        model = _get(field_path)
        if model:
            p = Path(model)
            if not p.exists() and "/" not in model:
                problems.append(
                    f"{field_path} {model!r} does not exist locally and is "
                    f"not an API model id"
                )

    if problems and strict:
        print("[repro] CONFIG INVALID — refusing to start:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)
    return problems


def write_manifest(cfg: dict, args: argparse.Namespace, out_path: Path) -> Path:
    """Write the experiment manifest (reproducibility record)."""
    import platform
    import torch
    import transformers

    def _git_rev() -> str | None:
        import subprocess
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                timeout=5, cwd=str(Path(__file__).resolve().parent.parent),
            ).stdout.strip() or None
        except Exception:
            return None

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_rev(),
        "config": {
            "path": str(args.config),
            "content": cfg,
        },
        "attacker_model": cfg["attacker"]["model"],
        "target_model": cfg["target"]["model"],
        "evaluator_model": cfg.get("evaluator", {}).get("rubric_model"),
        "embedding_model": cfg["embedding"]["model"],
        "nbf_checkpoint": cfg["nbf"]["checkpoint"],
        "nbf_checkpoint_sha256": sha256_of(Path(cfg["nbf"]["checkpoint"])),
        "dataset": cfg["dataset"]["path"],
        "dataset_split": cfg["dataset"].get("split"),
        "seed": cfg["experiment"].get("seed"),
        "temperature": cfg["target"].get("temperature"),
        "max_turns": cfg["attacks"].get("max_turns"),
        "threshold": cfg["nbf"].get("threshold"),
        "mode": args.mode,
        "limit": args.limit,
        "device": cfg["hardware"].get("device") if "hardware" in cfg else "cuda",
        "dtype": cfg["hardware"].get("dtype") if "hardware" in cfg else "bfloat16",
        "software_versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest_path


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    mode = args.mode or cfg["experiment"].get("mode", "A")
    if mode not in ("A", "B", "C"):
        raise ValueError(f"mode must be A, B, or C, got {mode!r}")

    limit = args.limit if args.limit is not None else cfg["experiment"].get("max_samples")
    start_from = args.start_from if args.start_from is not None else cfg["experiment"].get("start_from", 0)
    max_turns = args.max_turns or cfg["attacks"].get("max_turns", 8)
    eta = args.eta if args.eta is not None else cfg["nbf"].get("threshold", 0.0)

    # Config validation gate: refuse to start on any missing/invalid field.
    validate_config(cfg, strict=True)

    out_path = args.out or str(
        Path(cfg["experiment"].get("output_dir", "results/reproduction"))
        / f"repro_{mode}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    )

    goals = load_goals(OFFICIAL_DATASET, limit, start_from)
    print(f"[repro] mode={mode} goals={len(goals)} max_turns={max_turns} eta={eta}")

    enabled = args.attacks or [
        k for k, v in cfg["attacks"]["enabled"].items() if v
    ]
    if "all" in enabled:
        enabled = list(ATTACK_KEYS)

    # Modes A/B use the NBF barrier; A does not.
    use_barrier = mode in ("B", "C")

    mock = args.mock
    if args.dry_run:
        print("[repro] DRY RUN")
        for a in enabled:
            print(f"  attack={a} mode={mode} goals={len(goals)} barrier={'yes' if use_barrier else 'no'}")
        print(f"  out={out_path}")
        return

    # Print the FULL resolved configuration before execution (no-fallback
    # proof): operator must see exactly which models/checkpoint will run.
    print("[repro] resolved configuration:")
    print(f"  attacker  = {cfg['attacker']['model']}")
    print(f"  target    = {cfg['target']['model']}")
    print(f"  evaluator = {cfg.get('evaluator', {}).get('rubric_model')}")
    print(f"  embedding = {cfg['embedding']['model']}")
    print(f"  checkpoint= {cfg['nbf']['checkpoint']}")
    print(f"  dataset   = {OFFICIAL_DATASET} ({len(goals)} goals)")
    print(f"  mock      = {mock}")

    metadata = build_metadata(cfg, args, mock)

    manifest_path = write_manifest(cfg, args, Path(out_path))
    print(f"[repro] manifest: {manifest_path}")

    from guardbound.attacks.registry import get_attack

    attacker_llm = make_llm(cfg, "attacker", args, mock)
    target_llm = make_llm(cfg, "target", args, mock)
    barrier, embed_fn = make_barrier_and_embed(cfg, mock) if use_barrier else (None, None)

    print(f"[repro] attacker={metadata['attacker_model']} target={metadata['target_model']}")
    if barrier is not None:
        print(f"[repro] NBF checkpoint sha256={metadata['checkpoint_sha256']}")

    resume = not args.no_resume and cfg["experiment"].get("resume", True)
    done_goals = existing_goals(Path(out_path)) if resume else set()

    total_start = time.time()
    completed = 0
    for attack_key in enabled:
        attack = get_attack(attack_key)
        if hasattr(attack, "set_attacker_llm"):
            attack.set_attacker_llm(attacker_llm)

        for i, goal_record in enumerate(goals):
            goal = goal_record["task"]
            if (attack_key, goal) in done_goals:
                print(f"[repro] skip (resumed): {attack_key} / {goal[:60]}")
                continue

            start = time.time()
            record = await run_one(
                attack, goal_record, target_llm, barrier, embed_fn,
                cfg, args, mode, max_turns, eta, {**metadata, "attack": attack_key},
            )
            record["runtime"] = time.time() - start
            append_result(Path(out_path), record)
            completed += 1
            status = "SUCCESS" if record["success"] else "no"
            print(
                f"[repro] {attack_key} [{i+1}/{len(goals)}] turns={record['num_turns']} "
                f"success={status} filtered={record['filtered_queries']} "
                f"({record['runtime']:.1f}s)"
            )

    print(f"\n[repro] done. {completed} new conversations in {time.time() - total_start:.1f}s")
    print(f"[repro] output: {out_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.validate_config:
        cfg = load_config(args.config)
        problems = validate_config(cfg, strict=False)
        if problems:
            for p in problems:
                print(f"INVALID: {p}")
            raise SystemExit(2)
        print("CONFIG VALID")
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()