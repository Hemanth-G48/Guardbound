#!/usr/bin/env python
"""Resume embedding - optimized for GPU."""
from __future__ import annotations

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.data.embedder import ConversationEmbedder
from guardbound.schemas import load_conversations_jsonl

def main():
    convs = load_conversations_jsonl('data/processed/conversations/gpt35_circuit_breakers.jsonl')
    
    embedder = ConversationEmbedder(
        model_name='all-mpnet-base-v2',
        batch_size=512,  # Very large batch for GPU
        cache_dir=Path('data/processed/.cache/embeddings')
    )
    
    # Force CUDA device
    embedder._embedder._device = 'cuda'

    print(f'Total conversations: {len(convs)}')
    print(f'Cache has {len(embedder._cache)} texts')
    print(f'Batch size: {embedder.batch_size}')
    print(f'Device: cuda')

    # Find first uncached conversation
    start_idx = 0
    for i in range(len(convs)):
        has_uncached = False
        for turn in convs[i].turns:
            if embedder._cache.get('all-mpnet-base-v2', turn.query) is None:
                has_uncached = True
                break
        if has_uncached:
            start_idx = i
            break

    print(f'Starting from conversation {start_idx}')
    total_to_process = len(convs) - start_idx
    print(f'Will process {total_to_process} conversations')
    
    start_time = time.time()
    last_save = start_time
    
    for idx in range(start_idx, len(convs)):
        results = embedder.embed_conversations([convs[idx]], dry_run=False)
        
        if (idx - start_idx + 1) % 100 == 0:
            embedder._cache.save()
            elapsed = time.time() - start_time
            rate = (idx - start_idx + 1) / elapsed * 60
            print(f'Processed {idx - start_idx + 1}/{total_to_process} ({rate:.1f}/min)')
        
        # Save every 5 minutes
        if time.time() - last_save > 300:
            embedder._cache.save()
            last_save = time.time()

    embedder._cache.save()
    elapsed = time.time() - start_time
    print(f'Done! Cache has {len(embedder._cache)} texts in {elapsed/60:.1f} min')

if __name__ == "__main__":
    main()
