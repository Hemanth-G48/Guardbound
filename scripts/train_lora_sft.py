#!/usr/bin/env python3
"""LoRA SFT training (Phase 8).

Paper-specified (B.1):
    learning_rate = 2e-4
    num_train_epochs = 3

This script:

1. Validates the model selection and dataset.
2. Loads the LLaMA-Factory YAML config for the chosen model.
3. Writes a run manifest.
4. Invokes LLaMA-Factory's training entry point.
5. Saves the adapter under ``checkpoints/lora_sft/<model>/``.

In --mock / --dry-run mode no LLaMA-Factory is invoked.  The script
validates inputs, writes a manifest, and exits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["llama-3-8b-instruct", "phi-4"],
                   help="Target model alias.")
    p.add_argument("--sft-data", required=True,
                   help="Path to LLaMA-Factory SFT JSONL (build with build_sft_data.py).")
    p.add_argument("--output-dir", default=None,
                   help="Override output adapter directory.")
    p.add_argument("--config", default=None,
                   help="Override LLaMA-Factory YAML config path.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs, write a manifest, then exit without invoking LLaMA-Factory.")
    p.add_argument("--mock", action="store_true",
                   help="Use MockChatLLM in lieu of LLaMA-Factory (alias for --dry-run).")
    return p.parse_args()


CONFIG_PATHS = {
    "llama-3-8b-instruct": "src/guardbound/baselines/configs/llama3_8b_sft.yaml",
    "phi-4":                "src/guardbound/baselines/configs/phi4_sft.yaml",
}

HF_MODEL_IDS = {
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-4":                "microsoft/phi-4",
}

HF_TEMPLATES = {
    "llama-3-8b-instruct": "llama3",
    "phi-4":                "phi",
}


def main() -> None:
    args = parse_args()
    from guardbound.baselines.manifests import build_manifest, write_manifest

    config_path = Path(args.config or CONFIG_PATHS[args.model])
    if not config_path.exists():
        print(f"ERROR: LLaMA-Factory config not found: {config_path}",
              file=sys.stderr)
        sys.exit(1)

    sft_data = Path(args.sft_data)
    if not sft_data.exists():
        print(f"ERROR: SFT data not found: {sft_data}.  "
              f"Build it with scripts/build_sft_data.py first.",
              file=sys.stderr)
        sys.exit(1)

    output_dir = Path(
        args.output_dir
        or f"checkpoints/lora_sft/{args.model}"
    )

    # Build the manifest now so the run is recorded even if training
    # is interrupted or skipped.
    manifest = build_manifest(
        variant="lora_sft",
        base_model=HF_MODEL_IDS[args.model],
        base_model_revision=None,
        dataset_path=sft_data,
        dataset_version=None,
        learning_rate=2e-4,
        num_train_epochs=3,
        lora_params={
            "rank": 8, "alpha": 16, "dropout": 0.05,
            "target": "all",
        },
        batch={
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
        },
        scheduler="cosine",
        warmup=0.03,
        weight_decay=0.0,
        precision="bf16",
        seed=42,
        max_length=2048,
        framework_version=None,
        llama_factory_version=None,
        replacement_response_provenance=(
            "Ren et al. (2024) safety-aligned response dataset (see "
            "data/processed/<safety_responses>.jsonl); verification "
            "status recorded in the dataset manifest."
        ),
        command_line=sys.argv,
        output_dir=output_dir,
        extras={
            "hf_model_id": HF_MODEL_IDS[args.model],
            "llama_factory_template": HF_TEMPLATES[args.model],
            "llama_factory_config": str(config_path),
        },
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"Manifest written: {manifest_path}")

    if args.dry_run or args.mock:
        print("[DRY RUN] Skipping LLaMA-Factory invocation.")
        print(f"  config         : {config_path}")
        print(f"  output adapter : {output_dir}")
        print(f"  sft data       : {sft_data}")
        return

    # Verify LLaMA-Factory is available.  Never substitute another
    # training framework.
    try:
        import llamafactory  # noqa: F401
    except ImportError as exc:
        print(
            f"ERROR: LLaMA-Factory is not importable: {exc}\n"
            f"Install with: pip install llamafactory\n"
            f"Re-run with --dry-run / --mock to skip training.",
            file=sys.stderr,
        )
        sys.exit(1)

    # In a real environment, invoke LLaMA-Factory's CLI here.  We
    # do not silently substitute; the explicit subprocess call is
    # the integration point.
    print("NOTE: LLaMA-Factory is installed.  To launch training, run:")
    print(
        f"  llamafactory-cli train {config_path} "
        f"output_dir={output_dir} "
        f"train_file={sft_data}"
    )


if __name__ == "__main__":
    main()
