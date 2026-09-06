#!/usr/bin/env python3
"""Validate setup before running API-dependent tests.

This script checks that:
1. Environment variables are set
2. Python packages are available
3. NBF checkpoints exist
4. Data files exist

No API calls are made.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def check_environment():
    """Check environment variables."""
    print("\n" + "=" * 70)
    print("Checking Environment Variables")
    print("=" * 70)

    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key (required for GPT-4o)",
        "OPENAI_BASE_URL": "OpenAI base URL",
    }

    optional_vars = {
        "ANTHROPIC_API_KEY": "Anthropic API key (optional)",
        "GROQ_API_KEY": "Groq API key (optional)",
    }

    all_set = True

    print("\nRequired:")
    for var, desc in required_vars.items():
        value = __import__("os").getenv(var)
        if value and value != "your_openai_api_key_here":
            print(f"  [OK] {var} = ***{value[-4:]}")
        else:
            print(f"  [MISSING] {var} = NOT SET ({desc})")
            all_set = False

    print("\nOptional:")
    for var, desc in optional_vars.items():
        value = __import__("os").getenv(var)
        if value and value != f"your_{var.lower()}_here":
            print(f"  [OK] {var} = SET")
        else:
            print(f"  [-] {var} = NOT SET ({desc})")

    return all_set


def check_python_packages():
    """Check required Python packages."""
    print("\n" + "=" * 70)
    print("Checking Python Packages")
    print("=" * 70)

    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("sentence_transformers", "Sentence Transformers"),
        ("openai", "OpenAI"),
        ("anthropic", "Anthropic"),
        ("httpx", "HTTPX"),
    ]

    all_available = True
    for module, name in required:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [MISSING] {name} - NOT INSTALLED")
            all_available = False

    return all_available


def check_nbf_checkpoints():
    """Check NBF checkpoint files."""
    print("\n" + "=" * 70)
    print("Checking NBF Checkpoints")
    print("=" * 70)

    checkpoints = {
        "Dynamics": Path("checkpoints/dynamics_mpnet/dialogue_dynamics.pt"),
        "NBF": Path("checkpoints/nbf_mpnet/checkpoint.pt"),
        "Author's NBF": Path("nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/models_best_nbf_released.pth"),
    }

    all_exist = True
    for name, path in checkpoints.items():
        full_path = Path(_project_root) / path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name}: {size_mb:.1f} MB")
        else:
            print(f"  [MISSING] {name}: NOT FOUND at {path}")
            all_exist = False

    return all_exist


def check_dataset():
    """Check training data."""
    print("\n" + "=" * 70)
    print("Checking Training Data")
    print("=" * 70)

    datasets = {
        "Our Training Data": Path("data/processed/datasets/author_training_data/train.npz"),
        "Author's Data": Path("nbf_original_stuff/nbf_original_author/circuit_breakers_others.pt"),
    }

    all_exist = True
    for name, path in datasets.items():
        full_path = Path(_project_root) / path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {name}: {size_mb:.1f} MB")
        else:
            print(f"  [MISSING] {name}: NOT FOUND at {path}")
            all_exist = False

    return all_exist


def check_nbf_functionality():
    """Test NBF loading without API calls."""
    print("\n" + "=" * 70)
    print("Testing NBF Functionality (No API Calls)")
    print("=" * 70)

    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  [-] CUDA not available (using CPU)")

        # Load NBF
        from guardbound.models.dynamics import load_dynamics
        from guardbound.models.predictor import SafetyPredictor, NeuralBarrierFunction

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load dynamics
        dynamics_dir = Path(_project_root) / "checkpoints/dynamics_mpnet"
        dynamics = load_dynamics(dynamics_dir, device)
        print(f"  [OK] Dynamics loaded: {sum(p.numel() for p in dynamics.parameters())} params")

        # Load NBF
        predictor_path = Path(_project_root) / "checkpoints/nbf_mpnet/checkpoint.pt"
        ckpt = torch.load(predictor_path, map_location=device, weights_only=False)
        predictor = SafetyPredictor()
        predictor.load_state_dict(ckpt['predictor_state_dict'])
        predictor = predictor.to(device)
        print(f"  [OK] Predictor loaded: {predictor.param_count()} params")

        # Test NBF forward pass
        nbf = NeuralBarrierFunction(dynamics=dynamics, predictor=predictor)
        state = torch.zeros(1, 768, device=device)
        u = torch.zeros(1, 768, device=device)

        with torch.no_grad():
            h_val = nbf.h(state, u)
            h_value = h_val.item()

        print(f"  [OK] NBF forward pass: h={h_value:.4f}")

        return True

    except Exception as e:
        print(f"  [FAIL] NBF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("GUARDBOUND SETUP VALIDATION")
    print("=" * 70)

    results = {
        "Environment Variables": check_environment(),
        "Python Packages": check_python_packages(),
        "NBF Checkpoints": check_nbf_checkpoints(),
        "Training Data": check_dataset(),
        "NBF Functionality": check_nbf_functionality(),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] All checks passed! Ready to run API tests.")
        print("\nNext step: Provide API key and run:")
        print("  py scripts/run_evaluation.py --num-tasks 5 --dry-run  # Validate setup")
        print("  py scripts/run_evaluation.py --num-tasks 100  # Full evaluation")
    else:
        print("[FAIL] Some checks failed. Please fix the issues above.")
        print("\nTo set environment variables:")
        print("  Windows: set OPENAI_API_KEY=your_key")
        print("  Linux/Mac: export OPENAI_API_KEY=your_key")
        print("\nOr add them to a .env file (see .env.example)")

    print("=" * 70)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
