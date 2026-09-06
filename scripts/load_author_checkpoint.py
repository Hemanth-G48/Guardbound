"""Load author's pretrained NBF checkpoint.

The author's checkpoint uses different key names:
- 'ssm' -> our DialogueDynamics (f_theta, g_theta)
- 'nbf' -> our SafetyPredictor

Key mapping:
- ssm.state_transition.0.weight -> f_theta.net.0.weight
- ssm.observation_model.0.weight -> g_theta.net.0.weight
- nbf.nbf.0.weight -> net.0.weight
"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardbound.models.dynamics import DialogueDynamics
from guardbound.models.predictor import SafetyPredictor

def load_author_checkpoint(checkpoint_path, device="cpu"):
    """Load the author's pretrained checkpoint into our model structures."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    ssm_state = ckpt['ssm']  # NeuralStateSpaceModel
    nbf_state = ckpt['nbf']  # NeuralBarrierFunction
    
    # Create our models
    dynamics = DialogueDynamics(embedding_dim=768, state_dim=768, hidden_dims=[512, 512])
    predictor = SafetyPredictor(state_dim=768, embedding_dim=768)
    
    # Map SSM keys to our DialogueDynamics keys
    # ssm.state_transition -> f_theta.net
    # ssm.observation_model -> g_theta.net
    new_f_theta_state = {}
    new_g_theta_state = {}
    
    for key, value in ssm_state.items():
        if key.startswith('state_transition.'):
            new_key = key.replace('state_transition.', 'net.')
            new_f_theta_state[new_key] = value
        elif key.startswith('observation_model.'):
            new_key = key.replace('observation_model.', 'net.')
            new_g_theta_state[new_key] = value
    
    # Load f_theta and g_theta
    dynamics.f_theta.load_state_dict(new_f_theta_state)
    dynamics.g_theta.load_state_dict(new_g_theta_state)
    
    # Map NBF keys to our SafetyPredictor
    # nbf.nbf -> net
    new_nbf_state = {}
    for key, value in nbf_state.items():
        if key.startswith('nbf.'):
            new_key = key.replace('nbf.', 'net.')
            new_nbf_state[new_key] = value
    
    predictor.load_state_dict(new_nbf_state)
    
    dynamics.eval()
    predictor.eval()
    
    return dynamics, predictor


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/nbf_mpnet/models_best_nbf_released.pth")
    parser.add_argument("--output-dir", default="checkpoints/author_pretrained")
    args = parser.parse_args()
    
    dynamics, predictor = load_author_checkpoint(args.checkpoint)
    
    print(f"Dynamics loaded. Params: {dynamics.total_param_count():,}")
    print(f"Predictor loaded. Params: {predictor.param_count():,}")
    
    # Save in our format
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(dynamics.state_dict(), output_dir / "dialogue_dynamics.pt")
    torch.save(predictor.state_dict(), output_dir / "predictor_h.pt")
    
    print(f"Saved to {output_dir}")
