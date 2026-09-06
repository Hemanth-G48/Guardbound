#!/usr/bin/env python3
"""Adapt the paper's NBF checkpoint to this codebase's format.

The original paper's checkpoint has:
    - 'ssm.state_transition': f_theta (state transition)
    - 'ssm.observation_model': g_theta (observation model)
    - 'nbf': neural barrier function (h predictor)

This codebase expects:
    - dynamics.f_theta: state transition
    - dynamics.g_theta: observation model
    - predictor: SafetyPredictor
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def adapt_paper_checkpoint(paper_ckpt_path: str, output_path: str):
    """Convert paper's checkpoint to this codebase's format."""
    paper_ckpt = torch.load(paper_ckpt_path, map_location='cpu', weights_only=False)

    # Build dynamics state dict with renamed keys
    dynamics_state = {}
    for key, value in paper_ckpt['ssm'].items():
        if key.startswith('state_transition.'):
            new_key = 'f_theta.' + key[len('state_transition.'):]
            dynamics_state[new_key] = value
        elif key.startswith('observation_model.'):
            new_key = 'g_theta.' + key[len('observation_model.'):]
            dynamics_state[new_key] = value

    adapted = {
        'predictor_state_dict': paper_ckpt['nbf'],
        'dynamics_state_dict': dynamics_state,
        'architecture': {
            'state_dim': 768,
            'embedding_dim': 768,
        },
        'embedding_model': 'all-mpnet-base-v2',
        'training_mode': 'frozen',
        'eta': 0.001,
        'kappa': 3,
        'loss_weights': {},
        'epoch': 200,
        'seed': 0,
        'config': {},
    }

    torch.save(adapted, output_path)
    print(f"Adapted checkpoint saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='checkpoints/models_best_nbf_released.pth')
    parser.add_argument('--output', default='checkpoints/nbf_adapted.pt')
    args = parser.parse_args()

    adapt_paper_checkpoint(args.input, args.output)
