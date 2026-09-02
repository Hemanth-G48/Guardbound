#!/usr/bin/env python3
"""LoRA KTO training (Phase 8).

Not specified in the NBF paper.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["llama-3-8b-instruct", "phi-4"])
    p.add_argument("--kto-data", required=True,
                   help="Path to LLaMA-Factory KTO JSONL (build with build_preference_data.py).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mock", action="store_true")
    return p.parse_args()


CONFIG_PATHS = {
    "llama-3-8b-instruct": "src/guardbound/baselines/configs/llama3_8b_kto.yaml",
    "phi-4":                "src/guardbound/baselines/configs/phi4_kto.yaml",
}
HF_MODEL_IDS = {
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "phi-4":                "microsoft/phi-4",
}


def main() -> None:
    args = parse_args()
    from guardbound.baselines.manifests import build_manifest, write_manifest

    config_path = Path(args.config or CONFIG_PATHS[args.model])
    if not config_path.exists():
        print(f"ERROR: KTO config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    kto_data = Path(args.kto_data)
    if not kto_data.exists():
        print(f"ERROR: KTO data not found: {kto_data}", file=sys.stderr)
        sys.exit(1)
    output_dir = Path(args.output_dir or f"checkpoints/lora_kto/{args.model}")

    manifest = build_manifest(
        variant="lora_kto",
        base_model=HF_MODEL_IDS[args.model],
        dataset_path=kto_data,
        learning_rate=2e-4,
        num_train_epochs=3,
        lora_params={"rank": 8, "alpha": 16, "dropout": 0.05, "target": "all"},
        batch={"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8},
        scheduler="cosine",
        warmup=0.03,
        weight_decay=0.0,
        precision="bf16",
        seed=42,
        max_length=2048,
        chosen_provenance=(
            "Ren et al. (2024) safety-aligned response dataset (positive "
            "label); verification status recorded in dataset manifest."
        ),
        rejected_provenance=(
            "Not specified in the NBF paper — locally constructed from "
            "the Phase 2 corpus: 'undesirable' = the original jailbreak "
            "response."
        ),
        command_line=sys.argv,
        output_dir=output_dir,
        extras={"llama_factory_config": str(config_path)},
    )
    write_manifest(manifest, output_dir / "manifest.json")
    print(f"Manifest written: {output_dir / 'manifest.json'}")

    if args.dry_run or args.mock:
        print("[DRY RUN] Skipping LLaMA-Factory invocation.")
        return

    try:
        import llamafactory  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: LLaMA-Factory unavailable: {exc}", file=sys.stderr)
        sys.exit(1)

    print("NOTE: LLaMA-Factory is installed.  To launch training, run:")
    print(
        f"  llamafactory-cli train {config_path} "
        f"output_dir={output_dir} train_file={kto_data}"
    )


if __name__ == "__main__":
    main()
