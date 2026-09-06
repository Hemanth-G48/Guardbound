#!/usr/bin/env python
"""Build training dataset from NBF pseudo-labels.

Uses the 1-5 scores from the author's pretrained NBF.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import load_conversations_jsonl


def main():
    print("Building training dataset from NBF pseudo-labels...")

    # Load scored data
    scored_path = Path("data/processed/datasets/nbf_pseudo_labels/scored_turns.jsonl")
    with open(scored_path) as f:
        scored_turns = [json.loads(line) for line in f]

    print(f"Loaded {len(scored_turns)} scored turns")

    # Load original conversations for embeddings
    convs = load_conversations_jsonl('data/processed/conversations/gpt35_circuit_breakers.jsonl')

    # Load embeddings
    cache_path = Path("data/processed/.cache/embeddings_fast/all_embeddings.npz")
    data = np.load(cache_path)
    embeddings = data['embeddings']

    meta_raw = data['meta']
    if hasattr(meta_raw, 'item') and callable(meta_raw.item):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode('utf-8')
    if isinstance(meta_raw, np.ndarray):
        meta_raw = str(meta_raw)
    meta = json.loads(meta_raw)

    emb_map = {}
    for i, (c_idx, t_idx, field, text) in enumerate(meta):
        key = (c_idx, t_idx, field)
        emb_map[key] = embeddings[i]

    # Build dataset
    max_turns = 8
    max_convs = len(convs)

    U = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Z = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Y = np.full((max_convs, max_turns), -1, dtype=np.int64)
    MASK = np.zeros((max_convs, max_turns), dtype=np.bool_)

    score_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for turn_data in scored_turns:
        c_idx = turn_data["conv_idx"]
        t_idx = turn_data["turn_idx"]
        score = turn_data["predicted_label"]

        if t_idx >= max_turns:
            continue

        query_key = (c_idx, t_idx, 'query')
        resp_key = (c_idx, t_idx, 'response')

        if query_key in emb_map and resp_key in emb_map:
            U[c_idx, t_idx] = emb_map[query_key]
            Z[c_idx, t_idx] = emb_map[resp_key]
            Y[c_idx, t_idx] = score
            MASK[c_idx, t_idx] = True
            score_dist[score] += 1

    print(f"\nDataset shapes:")
    print(f"  U: {U.shape}")
    print(f"  Z: {Z.shape}")
    print(f"  Y: {Y.shape}")
    print(f"  MASK: {MASK.shape}")
    print(f"\nValid turns: {MASK.sum()}")
    print(f"Score distribution: {score_dist}")

    # Split 90/10
    n_train = int(0.9 * max_convs)

    output_dir = Path("data/processed/datasets/nbf_labeled")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / 'train.npz',
        U=U[:n_train], Z=Z[:n_train], Y=Y[:n_train], MASK=MASK[:n_train])
    np.savez_compressed(output_dir / 'val.npz',
        U=U[n_train:], Z=Z[n_train:], Y=Y[n_train:], MASK=MASK[n_train:])

    print(f"\nSaved train.npz ({n_train} convs) and val.npz ({max_convs - n_train} convs)")
    print(f"To: {output_dir}")

    # Also save score distribution summary
    summary = {
        "total_turns": int(MASK.sum()),
        "score_distribution": score_dist,
        "percentages": {k: f"{v/MASK.sum()*100:.1f}%" for k, v in score_dist.items()},
        "source": "NBF pseudo-labels from author pretrained model",
    }
    with open(output_dir / "label_distribution.json", "w") as f:
        json.dump(summary, f, indent=2)

    return U, Z, Y, MASK


if __name__ == "__main__":
    main()
