#!/usr/bin/env python3
"""Load and examine the author's training data."""
import sys
from pathlib import Path
import torch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    data_dir = Path(__file__).resolve().parent.parent / \
        "nbf_original_stuff" / "nbf_original_author"

    print("Loading author's training data...")
    print("=" * 70)

    # Load training data
    cb_actorattack = torch.load(data_dir / "circuit_breakers_actorattack.pt",
                               map_location="cpu", weights_only=False)
    cb_others = torch.load(data_dir / "circuit_breakers_others.pt",
                          map_location="cpu", weights_only=False)

    print(f"circuit_breakers_actorattack.pt: {len(cb_actorattack)} trajectories")
    print(f"circuit_breakers_others.pt: {len(cb_others)} trajectories")

    # Examine a single trajectory
    print("\n" + "=" * 70)
    print("Sample trajectory from circuit_breakers_others.pt:")
    traj = cb_others[0]
    print(f"Number of turns: {len(traj)}")

    for i, turn in enumerate(traj):
        print(f"\nTurn {i}:")
        for key, value in turn.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value)}")

    # Examine labels/scores
    print("\n" + "=" * 70)
    print("Score distribution in circuit_breakers_others.pt:")
    scores = []
    for traj in cb_others[:100]:  # Sample first 100
        for turn in traj:
            scores.append(turn['score'])

    from collections import Counter
    score_counts = Counter(scores)
    for score, count in sorted(score_counts.items()):
        print(f"  Score {score}: {count} ({count/len(scores)*100:.1f}%)")

    # Load validation data
    print("\n" + "=" * 70)
    harmbench_others = torch.load(data_dir / "harmbench_others.pt",
                                  map_location="cpu", weights_only=False)
    harmbench_actorattack = torch.load(data_dir / "harmbench_actorattack.pt",
                                       map_location="cpu", weights_only=False)

    print(f"harmbench_others.pt: {len(harmbench_others)} trajectories")
    print(f"harmbench_actorattack.pt: {len(harmbench_actorattack)} trajectories")

    print("\n" + "=" * 70)
    print("Data summary:")
    print(f"  Total training trajectories: {len(cb_actorattack) + len(cb_others)}")
    print(f"  Total validation trajectories: {len(harmbench_others) + len(harmbench_actorattack)}")


if __name__ == "__main__":
    main()
