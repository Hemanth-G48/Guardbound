#!/usr/bin/env python
"""Build training datasets from embedded conversations.

Usage:
    python scripts/build_nbf_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import load_conversations_jsonl

REFUSAL_PATTERNS = ['sorry', "can't", 'cannot', 'unable', 'cannot provide', 'unable to', 'will not provide', "won't provide", 'cannot help']

def is_safe_response(response_text: str) -> bool:
    if not response_text:
        return False
    lower = response_text.lower()
    return any(pat in lower for pat in REFUSAL_PATTERNS)

def main():
    convs = load_conversations_jsonl('data/processed/conversations/gpt35_circuit_breakers.jsonl')
    print(f'Loaded {len(convs)} conversations')
    
    data = np.load('data/processed/.cache/embeddings_fast/all_embeddings.npz')
    all_embeddings = data['embeddings']
    meta_raw = data['meta']
    if hasattr(meta_raw, 'item') and callable(meta_raw.item):
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode('utf-8')
    if isinstance(meta_raw, np.ndarray):
        meta_raw = str(meta_raw)
    meta = json.loads(meta_raw)
    
    print(f'Embeddings: {all_embeddings.shape}')
    print(f'Meta entries: {len(meta)}')
    
    emb_map = {}
    for i, (c_idx, t_idx, field, text) in enumerate(meta):
        key = (c_idx, t_idx, field)
        emb_map[key] = all_embeddings[i]
    
    max_turns = 8
    max_convs = len(convs)
    
    U = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Z = np.zeros((max_convs, max_turns, 768), dtype=np.float32)
    Y = np.full((max_convs, max_turns), -1, dtype=np.int64)  # -1 for padding
    MASK = np.zeros((max_convs, max_turns), dtype=np.bool_)
    
    safe_count = 0
    unsafe_count = 0
    
    for c_idx, conv in enumerate(convs):
        for t_idx, turn in enumerate(conv.turns):
            if t_idx >= max_turns:
                break
            
            query_key = (c_idx, t_idx, 'query')
            resp_key = (c_idx, t_idx, 'response')
            
            if query_key in emb_map and resp_key in emb_map:
                U[c_idx, t_idx] = emb_map[query_key]
                Z[c_idx, t_idx] = emb_map[resp_key]
                MASK[c_idx, t_idx] = True
                # Paper labels: {1,2,3,4} = safe, 5 = unsafe
                # Use heuristic: refusal = safe (3), no refusal = unsafe (5)
                if is_safe_response(turn.response):
                    Y[c_idx, t_idx] = 3  # safe refusal
                    safe_count += 1
                else:
                    Y[c_idx, t_idx] = 5  # unsafe (provided harmful content)
                    unsafe_count += 1
    
    print(f'U shape: {U.shape}')
    print(f'Z shape: {Z.shape}')
    print(f'Y shape: {Y.shape}')
    print(f'MASK shape: {MASK.shape}')
    print(f'Valid turns: {MASK.sum()}')
    print(f'Safe turns: {safe_count}, Unsafe turns: {unsafe_count}')
    
    n_train = int(0.9 * max_convs)
    
    train_dir = Path('data/processed/datasets/all_mpnet_base_v2')
    train_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(train_dir / 'train.npz', U=U[:n_train], Z=Z[:n_train], Y=Y[:n_train], MASK=MASK[:n_train])
    np.savez_compressed(train_dir / 'val.npz', U=U[n_train:], Z=Z[n_train:], Y=Y[n_train:], MASK=MASK[n_train:])
    
    print(f'Saved train.npz ({n_train} convs) and val.npz ({max_convs - n_train} convs) to {train_dir}')

if __name__ == "__main__":
    main()
