#!/usr/bin/env python
"""Fast batch embedding - saves everything at once at the end."""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import load_conversations_jsonl
from guardbound.embeddings import SentenceEmbedder

def main():
    convs = load_conversations_jsonl('data/processed/conversations/gpt35_circuit_breakers.jsonl')
    
    embedder = SentenceEmbedder('all-mpnet-base-v2')
    embedder._get_model().to('cuda')
    
    cache_dir = Path('data/processed/.cache/embeddings_fast')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Total conversations: {len(convs)}')
    
    # Collect all texts
    all_data = []  # (conv_idx, turn_idx, field, text)
    for c_idx, conv in enumerate(convs):
        for t_idx, turn in enumerate(conv.turns):
            all_data.append((c_idx, t_idx, 'query', turn.query))
            if turn.response:
                all_data.append((c_idx, t_idx, 'response', turn.response))
    
    texts = [d[3] for d in all_data]
    print(f'Total texts: {len(texts)}')
    
    # Batch embed all at once
    print('Embedding all texts...')
    start = time.time()
    embeddings = embedder.embed(texts)
    elapsed = time.time() - start
    print(f'Embedded {len(texts)} texts in {elapsed:.1f}s ({len(texts)/elapsed:.0f}/sec)')
    
    # Save all to single npz
    output_file = cache_dir / 'all_embeddings.npz'
    np.savez_compressed(output_file, embeddings=embeddings, meta=json.dumps(all_data))
    print(f'Saved to {output_file}')
    
    print('Done!')

if __name__ == "__main__":
    main()
