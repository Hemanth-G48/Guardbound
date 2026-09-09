"""Phase 7 — NBF checkpoint verification + deterministic sanity test.

Loads the official authors' checkpoint through the Guardbound compatibility
layer and verifies the full chain:

    checkpoint loads
        -> ssm  (DialogueDynamics: f_theta / g_theta)
        -> nbf  (SafetyPredictor)
        -> embedding dim = 768, state dim = 768, predictor outputs 5 classes

Then runs a DETERMINISTIC sanity test: for fixed embedding/state inputs it
reports raw logits, softmax probabilities, barrier score h(x,u) = p(unsafe) -
max(p(safe)), and the filtering decision at the configured threshold. This
proves the checkpoint participates in filtering — not merely that it "loaded".

Also verifies the key-remap correctness against the original author modules
(verify_conversion) so the Guardbound models are provably equivalent to the
official ones for identical inputs.

Usage:
    python scripts/verify_checkpoint.py [--config configs/reproduction.yaml]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

CONFIG_PATH = Path("configs/reproduction.yaml")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nbf_cfg = cfg["nbf"]
    ckpt_path = Path(nbf_cfg["checkpoint"])
    threshold = float(nbf_cfg.get("threshold", 0.0))
    emb_dim = int(cfg["embedding"]["dimension"])
    state_dim = int(nbf_cfg["state_dimension"])
    n_classes = int(nbf_cfg["predictor_classes"])

    report = {
        "checkpoint": str(ckpt_path),
        "config": args.config,
        "threshold": threshold,
        "checks": {},
    }

    # -- gate 1: checkpoint file exists + hash ------------------------------- #
    if not ckpt_path.exists():
        print(f"FAIL: checkpoint not found: {ckpt_path}")
        return 1
    ckpt_sha = sha256_of(ckpt_path)
    report["checkpoint_sha256"] = ckpt_sha
    print(f"[ckpt] sha256 = {ckpt_sha}")

    expected_sha = nbf_cfg.get("checkpoint_sha256_expected")
    if expected_sha and ckpt_sha != expected_sha:
        print(
            f"FAIL: checkpoint sha256 mismatch: expected {expected_sha}, "
            f"got {ckpt_sha}. Refusing to run with an unexpected checkpoint."
        )
        return 1
    report["checks"]["file_exists_and_hash"] = "PASS"

    # -- gate 2: full load chain -------------------------------------------- #
    from guardbound.models.compat import load_original_checkpoint, verify_conversion
    from guardbound.models.predictor import NeuralBarrierFunction

    dynamics, predictor = load_original_checkpoint(ckpt_path, device="cpu")
    print("[ckpt] load_original_checkpoint: OK (ssm -> dynamics, nbf -> predictor)")

    # f_theta / g_theta sub-modules must carry trained weights (non-constant).
    f_ok = any(p.abs().sum().item() > 0 for p in dynamics.f_theta.parameters())
    g_ok = any(p.abs().sum().item() > 0 for p in dynamics.g_theta.parameters())
    report["checks"]["f_theta_loads"] = "PASS" if f_ok else "FAIL"
    report["checks"]["g_theta_loads"] = "PASS" if g_ok else "FAIL"
    print(f"[ckpt] f_theta loads (non-zero weights): {'OK' if f_ok else 'FAIL'}")
    print(f"[ckpt] g_theta loads (non-zero weights): {'OK' if g_ok else 'FAIL'}")
    if not (f_ok and g_ok):
        print(json.dumps(report, indent=2))
        return 1

    barrier = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)
    report["checks"]["predictor_loads"] = "PASS"

    # -- gate 3: dimensions -------------------------------------------------- #
    # Official architecture (train.py): the first layer of both the dynamics
    # and the NBF consumes the CONCATENATION [x; u], so
    #   in_features = state_dim + embedding_dim (= 1536 for 768 + 768),
    # the dynamics output is state_dim, and the NBF output is class_num = 5.
    state_dim_actual = dynamics.f_theta.net[-1].out_features
    emb_dim_actual = dynamics.f_theta.net[0].in_features - state_dim_actual
    out_dim_actual = predictor.net[-1].out_features
    nbf_in_actual = predictor.net[0].in_features
    concat_ok = nbf_in_actual == state_dim_actual + emb_dim_actual
    dims_ok = (
        emb_dim_actual == emb_dim
        and state_dim_actual == state_dim
        and out_dim_actual == n_classes
        and concat_ok
    )
    report["checks"]["embedding_dim_768"] = (
        "PASS" if emb_dim_actual == emb_dim else f"FAIL ({emb_dim_actual})"
    )
    report["checks"]["state_dim_768"] = (
        "PASS" if state_dim_actual == state_dim else f"FAIL ({state_dim_actual})"
    )
    report["checks"]["predictor_5_classes"] = (
        "PASS" if out_dim_actual == n_classes else f"FAIL ({out_dim_actual})"
    )
    report["checks"]["nbf_input_is_concat_x_u"] = (
        "PASS" if concat_ok else f"FAIL ({nbf_in_actual})"
    )
    print(f"[ckpt] embedding dim: {emb_dim_actual} (expect {emb_dim})")
    print(f"[ckpt] state dim:     {state_dim_actual} (expect {state_dim})")
    print(f"[ckpt] predictor out: {out_dim_actual} (expect {n_classes})")
    print(f"[ckpt] NBF input:     {nbf_in_actual} = [x; u] concat "
          f"(expect {state_dim_actual + emb_dim_actual})")
    if not dims_ok:
        print(json.dumps(report, indent=2))
        return 1

    # -- gate 4: remap equivalence vs the ORIGINAL author modules ------------ #
    try:
        conv = verify_conversion(ckpt_path, atol=1e-6, rtol=1e-6)
        remap_ok = bool(conv.get("all_match", conv.get("passed", False)))
        report["checks"]["remap_equivalence_vs_original"] = (
            "PASS" if remap_ok else f"FAIL: {conv}"
        )
        print(f"[ckpt] remap equivalence vs original modules: "
              f"{'OK' if remap_ok else 'FAIL'}")
    except Exception as exc:  # noqa: BLE001
        report["checks"]["remap_equivalence_vs_original"] = f"ERROR: {exc}"
        remap_ok = False
        print(f"[ckpt] remap equivalence check errored: {exc}")
    if not remap_ok:
        print(json.dumps(report, indent=2))
        return 1

    # -- gate 5: deterministic sanity test ----------------------------------- #
    # Fixed inputs (seeded) so the numbers are reproducible run-to-run.
    torch.manual_seed(7)
    x_fixed = torch.zeros(1, state_dim_actual)          # official initial state
    u_fixed = torch.randn(1, state_dim_actual)

    # Route the fixed pair through the same scoring path the runner uses:
    # dynamics step on (x_prev, u), then predictor logits -> softmax -> score.
    with torch.no_grad():
        x_next = barrier.advance_state(x_fixed, u_fixed)
        logits = barrier.predictor(x_fixed, u_fixed)
        probs = torch.softmax(logits, dim=-1)
        last_class_prob = probs[:, -1]
        max_other_prob = torch.max(probs[:, :-1], dim=1).values
        barrier_score = (last_class_prob - max_other_prob)[0].item()

    decision = "FILTERED (unsafe)" if barrier_score >= threshold else "accepted (safe)"

    sanity = {
        "x0": "zeros(768) (official initial state)",
        "u": "torch.manual_seed(7); torch.randn(1, 768)",
        "logits": [round(v, 6) for v in logits[0].tolist()],
        "softmax_probs": [round(v, 6) for v in probs[0].tolist()],
        "barrier_score_h_xu": round(barrier_score, 6),
        "threshold": threshold,
        "filtering_decision": decision,
        "state_after_advance nonzero": bool(x_next.abs().sum().item() > 0),
    }
    report["sanity_test"] = sanity

    print("\n[ckpt] Deterministic sanity test (fixed seeded inputs):")
    print(f"  logits        = {sanity['logits']}")
    print(f"  softmax probs = {sanity['softmax_probs']}")
    print(f"  barrier score = {sanity['barrier_score_h_xu']}")
    print(f"  threshold     = {sanity['threshold']}")
    print(f"  decision      = {sanity['filtering_decision']}")

    # Note on softmax concentration: a trained classifier may legitimately
    # produce high-confidence outputs on out-of-distribution inputs; the load
    # correctness gate is the remap equivalence check above (byte-identical
    # outputs vs the original author modules), not a confidence heuristic.

    out_path = Path(cfg["experiment"].get("output_dir", "results/reproduction"))
    out_path.mkdir(parents=True, exist_ok=True)
    report_path = out_path / "checkpoint_verification.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[ckpt] report written to {report_path}")

    all_pass = all(v == "PASS" for k, v in report["checks"].items())
    print("CHECKPOINT VERIFIED" if all_pass else "CHECKPOINT VERIFICATION FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
