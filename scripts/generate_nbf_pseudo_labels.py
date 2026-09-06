#!/usr/bin/env python
"""Generate pseudo-labels using the author's pretrained NBF.

Uses pre-computed embeddings for speed - no re-encoding needed.
"""
from __future__ import annotations

import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.schemas import load_conversations_jsonl
from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor


def load_author_nbf(dynamics_path: Path, predictor_path: Path, device: str):
    """Load the author's pretrained NBF model (in Guardbound format)."""
    dynamics = DialogueDynamics(
        embedding_dim=768,
        state_dim=768,
        hidden_dims=[512, 512],
    )
    predictor = SafetyPredictor(
        state_dim=768,
        embedding_dim=768,
    )

    dynamics.load_state_dict(torch.load(dynamics_path, map_location=device, weights_only=True))
    predictor.load_state_dict(torch.load(predictor_path, map_location=device, weights_only=True))

    dynamics = dynamics.to(device)
    predictor = predictor.to(device)
    dynamics.eval()
    predictor.eval()

    return dynamics, predictor


def load_embeddings_cache():
    """Load pre-computed embeddings from cache."""
    print("Loading embeddings cache...")
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

    print(f"Loaded {len(embeddings)} embeddings")
    return emb_map


def score_all_turns(
    dynamics,
    predictor,
    emb_map: dict,
    convs: list,
    max_turns: int = 8,
    device: str = "cuda",
) -> tuple:
    """Score all turns using pre-computed embeddings."""
    all_scores = []
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    for conv_idx, conv in enumerate(tqdm(convs, desc="Conversations")):
        dialog_embs = []

        for turn_idx, turn in enumerate(conv.turns):
            if turn_idx >= max_turns:
                break

            query_key = (conv_idx, turn_idx, 'query')
            if query_key not in emb_map:
                continue

            query_emb = emb_map[query_key]

            if turn_idx == 0:
                state = np.zeros(768, dtype=np.float32)
            else:
                state = _update_state(dynamics, state, query_emb, device)

            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(device)
                query_t = torch.from_numpy(query_emb).unsqueeze(0).to(device)
                logits = predictor(state_t, query_t)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

            predicted_label = int(probs.argmax()) + 1
            p_unsafe = probs[4]
            p_safe_max = probs[:4].max()
            h_value = float(p_unsafe - p_safe_max)

            score_info = {
                "conv_idx": conv_idx,
                "turn_idx": turn_idx,
                "query": turn.query[:200] + "..." if len(turn.query) > 200 else turn.query,
                "response": turn.response[:200] + "..." if len(turn.response) > 200 else turn.response,
                "goal": conv.goal[:200] + "..." if len(conv.goal) > 200 else conv.goal,
                "predicted_label": predicted_label,
                "h_value": h_value,
                "class_probs": {i+1: float(probs[i]) for i in range(5)},
                "confidence": float(probs.max()),
            }

            score_distribution[predicted_label] += 1
            all_scores.append(score_info)
            dialog_embs.append(query_emb)

        if (conv_idx + 1) % 500 == 0:
            print(f"\n    Distribution: {score_distribution}")

    return all_scores, score_distribution


def _update_state(dynamics, state: np.ndarray, query_emb: np.ndarray, device: str) -> np.ndarray:
    """Update state using f_theta."""
    with torch.no_grad():
        state_t = torch.from_numpy(state).unsqueeze(0).to(device)
        query_t = torch.from_numpy(query_emb).unsqueeze(0).to(device)
        f_input = torch.cat([state_t, query_t], dim=-1)
        new_state = dynamics.f_theta(f_input)
        return new_state.cpu().numpy()[0]


def main():
    print("=" * 60)
    print("NBF-Based Pseudo-Labeling (Optimized)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\n[1/4] Loading conversations...")
    convs = load_conversations_jsonl('data/processed/conversations/gpt35_circuit_breakers.jsonl')
    print(f"    {len(convs)} conversations, ~{sum(len(c.turns) for c in convs)} turns")

    print("\n[2/4] Loading embeddings cache...")
    emb_map = load_embeddings_cache()

    print("\n[3/4] Loading author pretrained NBF...")
    dynamics_path = Path("checkpoints/author_pretrained/dialogue_dynamics.pt")
    predictor_path = Path("checkpoints/author_pretrained/predictor_h.pt")

    if not dynamics_path.exists() or not predictor_path.exists():
        print(f"    ERROR: Checkpoint not found")
        return

    dynamics, predictor = load_author_nbf(dynamics_path, predictor_path, device)
    print(f"    Models loaded to {device}")

    print("\n[4/4] Scoring turns...")
    all_scores, score_distribution = score_all_turns(
        dynamics, predictor, emb_map, convs, device=device
    )

    print("\n[Saving]...")
    output_dir = Path("data/processed/datasets/nbf_pseudo_labels")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "scored_turns.jsonl", "w") as f:
        for score in all_scores:
            f.write(json.dumps(score) + "\n")

    summary = {
        "total_turns": len(all_scores),
        "score_distribution": score_distribution,
        "percentages": {k: f"{v/len(all_scores)*100:.1f}%" for k, v in score_distribution.items()},
        "model_source": "hanjianghu/NBF-LLM (author pretrained)",
    }

    with open(output_dir / "scoring_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Total: {len(all_scores)} turns")
    for score, count in score_distribution.items():
        print(f"  Score {score}: {count:6d} ({count/len(all_scores)*100:.1f}%)")
    print(f"\nSaved to: {output_dir}")

    return all_scores, score_distribution


if __name__ == "__main__":
    scores, dist = main()
