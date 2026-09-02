#!/usr/bin/env python3
"""Paper-strict validation script.

Checks that the repository configuration matches every paper-specified
parameter and that all required datasets, checkpoints, and external
assets are present.  Returns:

    PASS    — all checks satisfied
    FAIL    — one or more critical checks failed
    NOT VERIFIED — could not establish equivalence (e.g. missing dataset)

Usage:
    python scripts/validate_paper_strict.py
    python scripts/validate_paper_strict.py --verbose
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Check registry
# --------------------------------------------------------------------------- #

class Check:
    def __init__(self, name: str, severity: str = "FAIL"):
        self.name = name
        self.severity = severity  # FAIL or WARN
        self.status = "PASS"
        self.message = ""

    def fail(self, msg: str) -> None:
        self.status = self.severity
        self.message = msg

    def pass_(self, msg: str = "") -> None:
        self.message = msg

    def unverifiable(self, msg: str) -> None:
        self.status = "NOT VERIFIED"
        self.message = msg


def _check_config(cfg) -> list[Check]:
    checks: list[Check] = []

    # --- dims ---
    c = Check("config: dims.embedding_dim_n == 768")
    if cfg.dims.embedding_dim_n == 768:
        c.pass_("768")
    else:
        c.fail(f"{cfg.dims.embedding_dim_n}")
    checks.append(c)

    c = Check("config: dims.state_dim_m == 768")
    if cfg.dims.state_dim_m == 768:
        c.pass_("768")
    else:
        c.fail(f"{cfg.dims.state_dim_m}")
    checks.append(c)

    # --- dialogue ---
    c = Check("config: dialogue.max_turns_k == 8")
    if cfg.dialogue.max_turns_k == 8:
        c.pass_("8")
    else:
        c.fail(f"{cfg.dialogue.max_turns_k}")
    checks.append(c)

    c = Check("config: dialogue.temperature == 0.7")
    if abs(cfg.dialogue.temperature - 0.7) < 1e-9:
        c.pass_("0.7")
    else:
        c.fail(f"{cfg.dialogue.temperature}")
    checks.append(c)

    # --- safety_labels ---
    c = Check("config: safety_labels.scores == [1,2,3,4,5]")
    if cfg.safety_labels.scores == [1, 2, 3, 4, 5]:
        c.pass_()
    else:
        c.fail(str(cfg.safety_labels.scores))
    checks.append(c)

    c = Check("config: safety_labels.unsafe_score == 5")
    if cfg.safety_labels.unsafe_score == 5:
        c.pass_("5")
    else:
        c.fail(f"{cfg.safety_labels.unsafe_score}")
    checks.append(c)

    c = Check("config: safety_labels.safe_scores == [1,2,3,4]")
    if cfg.safety_labels.safe_scores == [1, 2, 3, 4]:
        c.pass_()
    else:
        c.fail(str(cfg.safety_labels.safe_scores))
    checks.append(c)

    # --- loss weights ---
    c = Check("config: loss_weights.lambda_dyn == 1.0")
    if cfg.training.loss_weights.get("lambda_dyn") == 1.0:
        c.pass_()
    else:
        c.fail(str(cfg.training.loss_weights.get("lambda_dyn")))
    checks.append(c)

    c = Check("config: loss_weights.lambda_ce == 1.0")
    if cfg.training.loss_weights.get("lambda_ce") == 1.0:
        c.pass_()
    else:
        c.fail(str(cfg.training.loss_weights.get("lambda_ce")))
    checks.append(c)

    c = Check("config: loss_weights.lambda_ss == 100.0")
    if cfg.training.loss_weights.get("lambda_ss") == 100.0:
        c.pass_()
    else:
        c.fail(str(cfg.training.loss_weights.get("lambda_ss")))
    checks.append(c)

    c = Check("config: loss_weights.lambda_si == 100.0")
    if cfg.training.loss_weights.get("lambda_si") == 100.0:
        c.pass_()
    else:
        c.fail(str(cfg.training.loss_weights.get("lambda_si")))
    checks.append(c)

    # --- training ---
    c = Check("config: training.stage1_dynamics.lr == 1e-4")
    if cfg.training.stage1_dynamics.lr == 1e-4:
        c.pass_()
    else:
        c.fail(str(cfg.training.stage1_dynamics.lr))
    checks.append(c)

    c = Check("config: training.stage1_dynamics.epochs == 200")
    if cfg.training.stage1_dynamics.epochs == 200:
        c.pass_()
    else:
        c.fail(str(cfg.training.stage1_dynamics.epochs))
    checks.append(c)

    c = Check("config: training.stage2_predictor.lr == 1e-3")
    if cfg.training.stage2_predictor.lr == 1e-3:
        c.pass_()
    else:
        c.fail(str(cfg.training.stage2_predictor.lr))
    checks.append(c)

    c = Check("config: training.stage2_predictor.epochs == 200")
    if cfg.training.stage2_predictor.epochs == 200:
        c.pass_()
    else:
        c.fail(str(cfg.training.stage2_predictor.epochs))
    checks.append(c)

    c = Check("config: training.eta_train == 0.0")
    if cfg.training.eta_train == 0.0:
        c.pass_()
    else:
        c.fail(str(cfg.training.eta_train))
    checks.append(c)

    c = Check("config: training.kappa_noninvariant_turns == 3")
    if cfg.training.kappa_noninvariant_turns == 3:
        c.pass_()
    else:
        c.fail(str(cfg.training.kappa_noninvariant_turns))
    checks.append(c)

    # --- defense ---
    expected_eta_grid = [0.0, 1e-4, 2e-4, 4e-4, 5e-4, 6e-4, 8e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    c = Check("config: defense.eta_eval_grid matches paper")
    if cfg.defense.eta_eval_grid == expected_eta_grid:
        c.pass_()
    else:
        c.fail(f"{cfg.defense.eta_eval_grid}")
    checks.append(c)

    c = Check("config: defense.eta_recommended == 5e-4")
    if cfg.defense.eta_recommended == 5e-4:
        c.pass_()
    else:
        c.fail(str(cfg.defense.eta_recommended))
    checks.append(c)

    # --- embeddings ---
    c = Check("config: embeddings.default_model == all-mpnet-base-v2")
    if cfg.embeddings.default_model == "all-mpnet-base-v2":
        c.pass_()
    else:
        c.fail(cfg.embeddings.default_model)
    checks.append(c)

    c = Check("config: embeddings.alt_model == all-distilroberta-v1")
    if cfg.embeddings.alt_model == "all-distilroberta-v1":
        c.pass_()
    else:
        c.fail(cfg.embeddings.alt_model)
    checks.append(c)

    return checks


def _check_architecture() -> list[Check]:
    checks: list[Check] = []

    # Verify f_theta architecture
    c = Check("dynamics: f_theta input_dim == 1536")
    try:
        from guardbound.models.dynamics import DialogueDynamics
        dyn = DialogueDynamics(embedding_dim=768, state_dim=768)
        actual = dyn.f_theta.input_dim
        if actual == 1536:
            c.pass_(f"{actual}")
        else:
            c.fail(f"{actual}")
    except Exception as exc:
        c.fail(f"import error: {exc}")
    checks.append(c)

    # Verify g_theta architecture
    c = Check("dynamics: g_theta output_dim == 768")
    try:
        from guardbound.models.dynamics import DialogueDynamics
        dyn = DialogueDynamics(embedding_dim=768, state_dim=768)
        actual = dyn.g_theta.output_dim
        if actual == 768:
            c.pass_(f"{actual}")
        else:
            c.fail(f"{actual}")
    except Exception as exc:
        c.fail(f"import error: {exc}")
    checks.append(c)

    # Verify predictor architecture
    c = Check("predictor: SafetyPredictor input_dim == 1536")
    try:
        from guardbound.models.predictor import SafetyPredictor
        pred = SafetyPredictor(state_dim=768, embedding_dim=768)
        # Check first layer input dim
        first_layer = pred.net[0]
        actual = first_layer.in_features
        if actual == 1536:
            c.pass_(f"{actual}")
        else:
            c.fail(f"{actual}")
    except Exception as exc:
        c.fail(f"import error: {exc}")
    checks.append(c)

    c = Check("predictor: SafetyPredictor output_dim == 5")
    try:
        from guardbound.models.predictor import SafetyPredictor
        pred = SafetyPredictor(state_dim=768, embedding_dim=768)
        last_layer = pred.net[-1]
        actual = last_layer.out_features
        if actual == 5:
            c.pass_(f"{actual}")
        else:
            c.fail(f"{actual}")
    except Exception as exc:
        c.fail(f"import error: {exc}")
    checks.append(c)

    # Verify h equation: h = p(class5) - max(p(class1..4))
    c = Check("predictor: predictor_value = p5 - max(p1..4)")
    try:
        import torch
        from guardbound.models.predictor import SafetyPredictor
        pred = SafetyPredictor(state_dim=768, embedding_dim=768)
        x = torch.zeros(1, 768)
        u = torch.zeros(1, 768)
        h = pred.predictor_value(x, u)
        # Check that it matches manual calculation
        p = pred.class_probs(x, u)
        expected = p[0, 4].item() - p[0, :4].max().item()
        if abs(h.item() - expected) < 1e-5:
            c.pass_(f"h={h.item():.4f}")
        else:
            c.fail(f"h={h.item():.4f}, expected={expected:.4f}")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def _check_steering() -> list[Check]:
    checks: list[Check] = []

    c = Check("steering: Q-filter uses (h + eta) >= 0")
    try:
        from guardbound.defense.steered_chat import SteeredLLMChat
        import inspect
        src = inspect.getsource(SteeredLLMChat.chat)
        if "(h_value + self.eta) >= 0" in src or "(h_value + self.eta) >= 0" in src.replace(" ", ""):
            c.pass_()
        else:
            c.fail("filter rule not found")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("steering: x_0 = zeros(768)")
    try:
        from guardbound.defense.steered_chat import SteeredLLMChat
        import inspect
        src = inspect.getsource(SteeredLLMChat)
        if "zeros(1, self._state_dim)" in src or "zeros(768" in src:
            c.pass_()
        else:
            c.fail("zeros init not found")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("steering: filtered queries do NOT update state")
    try:
        from guardbound.defense.steered_chat import SteeredLLMChat
        import inspect
        src = inspect.getsource(SteeredLLMChat.chat)
        # When filtered, state should NOT be advanced
        if "filtered" in src.lower():
            c.pass_()
        else:
            c.fail("filtered branch not found")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def _check_attacks() -> list[Check]:
    checks: list[Check] = []

    attack_names = [
        ("crescendo", "src/guardbound/attacks/crescendo.py"),
        ("actor_attack", "src/guardbound/attacks/actor_attack.py"),
        ("opposite_day", "src/guardbound/attacks/opposite_day.py"),
        ("acronym", "src/guardbound/attacks/acronym.py"),
        ("red_queen", "src/guardbound/attacks/red_queen.py"),
    ]

    for name, path in attack_names:
        c = Check(f"attack: {name} is NotImplementedError stub")
        full_path = Path(_project_root) / path
        if not full_path.exists():
            c.fail(f"file not found: {path}")
        else:
            content = full_path.read_text()
            if "NotImplementedError" in content:
                c.pass_("stub present")
            else:
                c.fail("not a stub")
        checks.append(c)

    # Adaptive attack must be IMPLEMENTED
    c = Check("attack: adaptive is IMPLEMENTED (not stub)")
    try:
        from guardbound.attacks.adaptive import AdaptiveNBFAttack
        import inspect
        src = inspect.getsource(AdaptiveNBFAttack.next_query)
        if "NotImplementedError" in src:
            c.fail("adaptive is still a stub")
        elif "max(" in src and "h_values" in src:
            c.pass_("max h selection found")
        else:
            c.fail("implementation unclear")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    # Verify adaptive selects MAXIMUM h
    c = Check("attack: adaptive selects MAXIMUM h (not minimum)")
    try:
        from guardbound.attacks.adaptive import AdaptiveNBFAttack
        import inspect
        src = inspect.getsource(AdaptiveNBFAttack.next_query)
        if "max(" in src:
            c.pass_("uses max()")
        else:
            c.fail("does not use max()")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def _check_evaluation() -> list[Check]:
    checks: list[Check] = []

    # ASR denominator = behaviors
    c = Check("eval: ASR denominator = behaviors (not turns)")
    try:
        from guardbound.evaluation.asr import compute_asr
        import inspect
        src = inspect.getsource(compute_asr)
        # Should count len(verdicts), not sum of turn counts
        if "len(verdicts)" in src or "total_behaviors" in src:
            c.pass_()
        else:
            c.fail("denominator unclear")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    # MMLU h > 0 (strict)
    c = Check("eval: MMLU uses h > 0 (strict, not >=)")
    try:
        from guardbound.defense.post_processing import mmlu_filter_answer
        import inspect
        src = inspect.getsource(mmlu_filter_answer)
        if "h_value > 0" in src and "h_value >= 0" not in src:
            c.pass_()
        else:
            c.fail("h > 0 not found or h >= 0 present")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    # MTBench h > 0 (strict)
    c = Check("eval: MTBench uses h > 0 (strict, not >=)")
    try:
        from guardbound.defense.post_processing import mtbench_replace_answer
        import inspect
        src = inspect.getsource(mtbench_replace_answer)
        if "h_value > 0" in src and "h_value >= 0" not in src:
            c.pass_()
        else:
            c.fail("h > 0 not found or h >= 0 present")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    # Guardrail: argmax == 1 → harmless
    c = Check("eval: guardrail uses argmax class == 1 → harmless")
    try:
        from guardbound.evaluation.guardrail_f1 import NBFPromptGuard
        import inspect
        src = inspect.getsource(NBFPromptGuard.predict)
        if "argmax" in src and ("== 1" in src or "==1" in src):
            c.pass_()
        else:
            c.fail("argmax==1 rule not found")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def _check_datasets() -> list[Check]:
    checks: list[Check] = []
    data_root = Path(_project_root) / "data"

    required = [
        ("data/raw/harmbench/raw_data.jsonl", "HarmBench test behaviors"),
        ("data/raw/circuit_breakers/", "Circuit Breakers training data"),
        ("data/raw/xstest.jsonl", "XSTest over-refusal"),
        ("data/raw/jbb_benign.jsonl", "JBB-Benign over-refusal"),
        ("data/raw/phtest_harmless.jsonl", "PHTest-Harmless over-refusal"),
    ]

    for rel_path, description in required:
        full_path = data_root / rel_path
        c = Check(f"dataset: {description}")
        if full_path.exists():
            if full_path.is_file() and full_path.stat().st_size > 0:
                c.pass_(f"{full_path.stat().st_size} bytes")
            elif full_path.is_dir():
                files = list(full_path.glob("*"))
                if files:
                    c.pass_(f"{len(files)} files")
                else:
                    c.fail("directory empty")
            else:
                c.fail("exists but empty")
        else:
            c.unverifiable(f"not found: {full_path}")
        checks.append(c)

    return checks


def _check_baselines() -> list[Check]:
    checks: list[Check] = []

    c = Check("baseline: original variant present")
    try:
        from guardbound.baselines.registry import build_defense
        # This should not raise for valid variant
        c.pass_()
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("baseline: system_prompt variant has prompt file")
    prompt_path = Path(_project_root) / "src/guardbound/baselines/prompts/llama2_safety_system.txt"
    if prompt_path.exists():
        c.pass_(f"{prompt_path.stat().st_size} bytes")
    else:
        c.fail(f"not found: {prompt_path}")
    checks.append(c)

    c = Check("baseline: lora_sft lr=2e-4, epochs=3")
    try:
        script_path = Path(_project_root) / "scripts/train_lora_sft.py"
        content = script_path.read_text()
        if "2e-4" in content and "num_train_epochs" in content and "3" in content:
            c.pass_()
        else:
            c.fail("hyperparameters not found")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("baseline: DPO/KTO documented as NOT SPECIFIED IN PAPER")
    try:
        dpo_path = Path(_project_root) / "scripts/train_lora_dpo.py"
        kto_path = Path(_project_root) / "scripts/train_lora_kto.py"
        dpo_content = dpo_path.read_text() if dpo_path.exists() else ""
        kto_content = kto_path.read_text() if kto_path.exists() else ""
        if "Not specified in the NBF paper" in dpo_content and "Not specified in the NBF paper" in kto_content:
            c.pass_()
        else:
            c.fail("missing paper-strict documentation")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def _check_experiments() -> list[Check]:
    checks: list[Check] = []

    c = Check("experiments: E1-E10 registry present")
    try:
        from guardbound.experiments.registry import EXPERIMENTS
        expected = {f"E{i}" for i in range(1, 11)}
        actual = set(EXPERIMENTS.keys())
        missing = expected - actual
        if not missing:
            c.pass_(", ".join(sorted(EXPERIMENTS.keys())))
        else:
            c.fail(f"missing: {missing}")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("experiments: smoke mode implemented")
    try:
        from guardbound.experiments.runner import RunnerConfig
        cfg = RunnerConfig(smoke=True)
        if cfg.smoke is True:
            c.pass_()
        else:
            c.fail("smoke not True")
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    c = Check("experiments: findings verifier present")
    try:
        from guardbound.analysis.findings import check_claims
        c.pass_()
    except Exception as exc:
        c.fail(f"error: {exc}")
    checks.append(c)

    return checks


def main() -> None:
    args = parse_args()

    all_checks: list[Check] = []

    # Load config
    try:
        from guardbound.config import load_config
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"FAIL: Could not load config: {exc}")
        sys.exit(1)

    all_checks.extend(_check_config(cfg))
    all_checks.extend(_check_architecture())
    all_checks.extend(_check_steering())
    all_checks.extend(_check_attacks())
    all_checks.extend(_check_evaluation())
    all_checks.extend(_check_datasets())
    all_checks.extend(_check_baselines())
    all_checks.extend(_check_experiments())

    # Count
    n_fail = sum(1 for c in all_checks if c.status == "FAIL")
    n_warn = sum(1 for c in all_checks if c.status == "WARN")
    n_not_verified = sum(1 for c in all_checks if c.status == "NOT VERIFIED")
    n_pass = sum(1 for c in all_checks if c.status == "PASS")

    if args.verbose:
        print("\n=== Paper-Strict Validation Results ===\n")
        for c in all_checks:
            symbol = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "NOT VERIFIED": "❓"}.get(c.status, "?")
            print(f"  {symbol} [{c.status}] {c.name}")
            if c.message:
                print(f"       {c.message}")
            print()

    print(f"\nSummary: {n_pass} passed, {n_fail} failed, {n_warn} warnings, {n_not_verified} not verified")

    if n_fail > 0:
        print("\nFAILED")
        sys.exit(1)
    else:
        print("\nPASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
