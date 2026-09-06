#!/usr/bin/env python3
"""Convert author's data format to our training format."""
import sys
from pathlib import Path
import torch
import numpy as np

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def convert_trajectory_dataset(trajectories, max_turns=8):
    """Convert trajectory list to [U, Z, Y, MASK] format.

    Args:
        trajectories: List of trajectories, each containing turns with u, y, score
        max_turns: Maximum number of turns (K)

    Returns:
        U: User embeddings [N, K, 768]
        Z: Assistant embeddings [N, K, 768]
        Y: Scores [N, K] (1-5, 0 for padding)
        MASK: Boolean mask [N, K]
    """
    N = len(trajectories)
    K = max_turns
    D = 768  # embedding dim

    U = np.zeros((N, K, D), dtype=np.float32)
    Z = np.zeros((N, K, D), dtype=np.float32)
    Y = np.zeros((N, K), dtype=np.int64)
    MASK = np.zeros((N, K), dtype=bool)

    for n, traj in enumerate(trajectories):
        for k, turn in enumerate(traj):
            if k >= K:
                break
            U[n, k] = turn['u'].numpy()
            Z[n, k] = turn['y'].numpy()
            Y[n, k] = turn['score']  # Paper scores: 1-5
            MASK[n, k] = True

    return U, Z, Y, MASK


def main():
    data_dir = Path(__file__).resolve().parent.parent / \
        "nbf_original_stuff" / "nbf_original_author"

    output_dir = Path(__file__).resolve().parent.parent / \
        "data" / "processed" / "datasets" / "author_training_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading author's training data...")
    print("=" * 70)

    # Load training data
    cb_actorattack = torch.load(data_dir / "circuit_breakers_actorattack.pt",
                               map_location="cpu", weights_only=False)
    cb_others = torch.load(data_dir / "circuit_breakers_others.pt",
                          map_location="cpu", weights_only=False)

    # Load validation data
    harmbench_others = torch.load(data_dir / "harmbench_others.pt",
                                  map_location="cpu", weights_only=False)
    harmbench_actorattack = torch.load(data_dir / "harmbench_actorattack.pt",
                                       map_location="cpu", weights_only=False)

    # Combine data
    train_trajectories = cb_actorattack + cb_others
    val_trajectories = harmbench_others + harmbench_actorattack

    print(f"Training trajectories: {len(train_trajectories)}")
    print(f"Validation trajectories: {len(val_trajectories)}")

    # Find max turns
    max_train_turns = max(len(t) for t in train_trajectories)
    max_val_turns = max(len(t) for t in val_trajectories)
    max_turns = max(max_train_turns, max_val_turns, 8)
    print(f"Max turns: {max_turns}")

    # Analyze score distribution
    print("\n" + "=" * 70)
    print("Score distribution in training data:")
    from collections import Counter
    train_scores = []
    for traj in train_trajectories:
        for turn in traj:
            train_scores.append(turn['score'])
    score_counts = Counter(train_scores)
    total = len(train_scores)
    for score in sorted(score_counts.keys()):
        count = score_counts[score]
        print(f"  Score {score}: {count} ({count/total*100:.1f}%)")

    # Convert to our format
    print("\n" + "=" * 70)
    print("Converting training data...")
    train_U, train_Z, train_Y, train_MASK = convert_trajectory_dataset(
        train_trajectories, max_turns
    )
    print(f"  train_U: {train_U.shape}")
    print(f"  train_Z: {train_Z.shape}")
    print(f"  train_Y: {train_Y.shape}")
    print(f"  train_MASK: {train_MASK.shape}")

    print("\nConverting validation data...")
    val_U, val_Z, val_Y, val_MASK = convert_trajectory_dataset(
        val_trajectories, max_turns
    )
    print(f"  val_U: {val_U.shape}")
    print(f"  val_Z: {val_Z.shape}")
    print(f"  val_Y: {val_Y.shape}")
    print(f"  val_MASK: {val_MASK.shape}")

    # Save as npz files
    print("\n" + "=" * 70)
    print(f"Saving to {output_dir}...")

    train_path = output_dir / "train.npz"
    val_path = output_dir / "val.npz"

    np.savez(train_path,
             U=train_U, Z=train_Z, Y=train_Y, MASK=train_MASK)
    np.savez(val_path,
             U=val_U, Z=val_Z, Y=val_Y, MASK=val_MASK)

    print(f"  Saved train.npz")
    print(f"  Saved val.npz")

    # Verify saved files
    print("\nVerifying saved files...")
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    print(f"  train.npz keys: {list(train_data.keys())}")
    print(f"  val.npz keys: {list(val_data.keys())}")

    print("\n" + "=" * 70)
    print("Conversion complete!")


if __name__ == "__main__":
    main()
