#!/usr/bin/env python3
"""Analyze h-values in training data to understand patterns."""
import sys
from pathlib import Path
import torch
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from guardbound.models.dynamics import DialogueDynamics, load_dynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction


def load_our_nbf(device: str = "cuda"):
    """Load our newly trained NBF."""
    dynamics_dir = Path("checkpoints/dynamics_mpnet")
    predictor_path = Path("checkpoints/nbf_mpnet/checkpoint.pt")

    dynamics = load_dynamics(dynamics_dir, device)
    dynamics = dynamics.to(device)

    ckpt = torch.load(predictor_path, map_location=device, weights_only=False)
    predictor = SafetyPredictor()
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    predictor = predictor.to(device)
    predictor.eval()
    dynamics.eval()

    return NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training data
    data_dir = Path(__file__).resolve().parent.parent / \
        "nbf_original_stuff" / "nbf_original_author"

    cb_others = torch.load(data_dir / "circuit_breakers_others.pt",
                          map_location="cpu", weights_only=False)

    print(f"Loaded {len(cb_others)} trajectories")

    # Load NBF
    print("\nLoading NBF...")
    nbf = load_our_nbf(device)
    nbf.eval()

    # Analyze h-values by score and turn position
    print("\nAnalyzing h-values by score and turn position...")
    print("=" * 70)

    turn_stats = {}  # turn_idx -> {score -> [h_values]}

    for traj in cb_others[:500]:  # Sample 500 trajectories
        state = torch.zeros(1, 768, device=device)

        for k, turn in enumerate(traj):
            score = turn['score']
            u = turn['u'].unsqueeze(0).to(device)

            with torch.no_grad():
                h_val = nbf.h(state, u)
                h_value = h_val.item()

                if k not in turn_stats:
                    turn_stats[k] = {}
                if score not in turn_stats[k]:
                    turn_stats[k][score] = []
                turn_stats[k][score].append(h_value)

                # Advance state
                state = nbf.advance_state(state, u)

    # Print statistics
    print("\nH-values by turn and score:")
    for turn_idx in sorted(turn_stats.keys()):
        print(f"\nTurn {turn_idx}:")
        for score in sorted(turn_stats[turn_idx].keys()):
            h_values = turn_stats[turn_idx][score]
            mean_h = np.mean(h_values)
            std_h = np.std(h_values)
            count = len(h_values)
            print(f"  Score {score}: n={count:4d}, mean_h={mean_h:7.4f}, std={std_h:.4f}")

    print("\n" + "=" * 70)
    print("\nKey observations:")
    print("- h > 0 means predicted unsafe (FILTERED)")
    print("- h < 0 means predicted safe (ALLOWED)")
    print("- The NBF should learn to have higher h for higher scores")


if __name__ == "__main__":
    main()
