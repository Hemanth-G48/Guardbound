#!/usr/bin/env python
"""Generate pseudo-labels for training using author's pretrained NBF.

Uses the author's NBF to generate h-values and labels for our conversations.
Then trains our own NBF on this data.

Usage:
    python scripts/retrain_with_pseudo_labels.py --nbf-dir checkpoints/author_pretrained
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import load_conversations_jsonl
from guardbound.embeddings import SentenceEmbedder
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction

def h_value_to_label(h_val: float) -> int:
    """Map h(x,u) value to paper label {1,2,3,4,5}.
    
    Paper scale:
        h < -0.5 -> 1 (very safe)
        -0.5 <= h < -0.1 -> 2 (safe)
        -0.1 <= h < 0.1 -> 3 (moderate)
        0.1 <= h < 0.5 -> 4 (unsafe)
        h >= 0.5 -> 5 (very unsafe)
    """
    if h_val < -0.5:
        return 1
    elif h_val < -0.1:
        return 2
    elif h_val < 0.1:
        return 3
    elif h_val < 0.5:
        return 4
    else:
        return 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbf-dir", default="checkpoints/author_pretrained")
    parser.add_argument("--data", default="data/processed/conversations/gpt35_circuit_breakers.jsonl")
    parser.add_argument("--embeddings", default="data/processed/.cache/embeddings_fast/all_embeddings.npz")
    parser.add_argument("--output-dir", default="data/processed/datasets/labeled_by_author")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()
    
    print("Loading author's NBF...")
    nbf_dir = Path(args.nbf_dir)
    
    dynamics = DialogueDynamics(embedding_dim=768, state_dim=768, hidden_dims=[512, 512])
    dynamics.load_state_dict(torch.load(nbf_dir / "dialogue_dynamics.pt", map_location="cpu", weights_only=False))
    dynamics.eval()
    
    predictor = SafetyPredictor(state_dim=768, embedding_dim=768)
    predictor.load_state_dict(torch.load(nbf_dir / "predictor_h.pt", map_location="cpu", weights_only=False))
    predictor.eval()
    
    nbf = NeuralBarrierFunction(dynamics, predictor)
    
    print("Loading embedder...")
    embedder = SentenceEmbedder("all-mpnet-base-v2")
    
    print("Loading pre-computed embeddings...")
    emb_data = np.load(args.embeddings)
    all_embeddings = emb_data['embeddings']
    meta_raw = emb_data['meta']
    
    # Parse meta
    if hasattr(meta_raw, 'item') and callable(meta_raw.item):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode('utf-8')
    if isinstance(meta_raw, np.ndarray):
        meta_raw = str(meta_raw)
    meta = json.loads(meta_raw)
    
    # Build embedding map
    emb_map = {}
    for i, (c_idx, t_idx, field, text) in enumerate(meta):
        key = (c_idx, t_idx, field)
        emb_map[key] = all_embeddings[i]
    
    print(f"Embeddings: {all_embeddings.shape}")
    print(f"Meta entries: {len(meta)}")
    
    print("Loading conversations...")
    convs = load_conversations_jsonl(args.data)
    print(f"Conversations: {len(convs)}")
    
    # Generate labels for each conversation
    max_turns = args.max_turns
    max_convs = len(convs)
    
    U = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Z = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Y = np.zeros((max_convs, max_turns), dtype=np.int64)  # Will be filled with labels 1-5
    MASK = np.zeros((max_convs, max_turns), dtype=np.bool_)
    
    label_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    print("Generating pseudo-labels using author's NBF...")
    for c_idx, conv in enumerate(convs):
        if c_idx % 500 == 0:
            print(f"  Processing conversation {c_idx}/{len(convs)}")
        
        state = np.zeros(768, dtype=np.float32)
        
        for t_idx, turn in enumerate(conv.turns[:max_turns]):
            query_key = (c_idx, t_idx, 'query')
            resp_key = (c_idx, t_idx, 'response')
            
            if query_key not in emb_map or resp_key not in emb_map:
                continue
            
            u_t = emb_map[query_key]
            z_t = emb_map[resp_key]
            
            # Compute h(x, u) for response
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            u_tensor = torch.from_numpy(u_t).unsqueeze(0)
            
            with torch.no_grad():
                h_val = nbf.h(state_tensor, u_tensor).item()
            
            # Map to label
            label = h_value_to_label(h_val)
            label_counts[label] += 1
            
            # Store
            U[c_idx, t_idx] = u_t
            Z[c_idx, t_idx] = z_t
            Y[c_idx, t_idx] = label
            MASK[c_idx, t_idx] = True
            
            # Advance state
            state = u_t  # Use query embedding as next state input (simplified)
    
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / max(MASK.sum(), 1)
        print(f"  Label {label}: {count} ({pct:.1f}%)")
    
    print(f"\nValid turns: {MASK.sum()}")
    
    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_dir / "train.npz", U=U, Z=Z, Y=Y, MASK=MASK)
    print(f"\nSaved labeled dataset to {output_dir}/train.npz")
    
    # Also save a copy with underscore naming for compatibility
    alt_dir = Path("data/processed/datasets/labeled_by_author")
    alt_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(alt_dir / "train.npz", U=U, Z=Z, Y=Y, MASK=MASK)
    print(f"Saved to {alt_dir}/train.npz")


if __name__ == "__main__":
    main()
