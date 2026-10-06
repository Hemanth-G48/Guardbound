"""Phase 17 Stage 3 — Parts 7/8/9: architecture audit and native loader probe.

Reads the *published* Qwen3.8-27B checkpoint configuration from the local
snapshot and establishes, by measurement rather than by name-matching:

  * what the checkpoint declares (architecture, text/vision sub-configs,
    processor class, context length, EOS set);
  * which AutoClass and which concrete class each candidate resolution path
    produces for that config;
  * the parameter-name structure each candidate class expects, compared against
    the checkpoint's own ``model.safetensors.index.json`` — the structural test
    that Stage 2 could not make (its raw key comparison was confounded by the
    CausalLM path's key mapping).

No weight shard is read by this script: everything comes from config files and
from tiny synthetic instances built on the meta device.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoConfig, AutoProcessor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage3_qwen38_native"

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Auto-classes that could plausibly resolve a `qwen3_5` config.
CANDIDATE_AUTO_CLASSES = (
    "AutoModelForImageTextToText",
    "AutoModelForMultimodalLM",
    "AutoModelForCausalLM",
)


def snapshot_path() -> Path:
    path = HF_CACHE / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    for candidate in path.iterdir():
        if candidate.is_dir() and candidate.name == REVISION:
            return candidate
    raise FileNotFoundError(f"revision {REVISION} not found under {path}")


def shrink(config):
    """A tiny same-class config, for enumerating expected parameter names."""
    text = config.text_config
    text.num_hidden_layers = 2
    text.hidden_size = 32
    text.intermediate_size = 64
    text.num_attention_heads = 2
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.linear_num_key_heads = 2
    text.linear_num_value_heads = 2
    text.linear_key_head_dim = 8
    text.linear_value_head_dim = 8
    text.linear_conv_kernel_dim = 4
    text.vocab_size = 128
    text.layer_types = ["linear_attention", "full_attention"]
    text.mtp_num_hidden_layers = 1
    vision = config.vision_config
    vision.depth = 1
    vision.hidden_size = 16
    vision.intermediate_size = 32
    vision.num_heads = 2
    vision.out_hidden_size = 32
    vision.num_position_embeddings = 64
    return config


def expected_structure(auto_name: str, config) -> dict:
    """Parameter-name structure a given AutoClass produces for this config."""
    auto = getattr(transformers, auto_name, None)
    if auto is None:
        return {"auto_class": auto_name, "available": False}
    try:
        with torch.device("meta"):
            model = auto.from_config(shrink(config))
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        return {
            "auto_class": auto_name,
            "available": True,
            "instantiated": False,
            "error": f"{type(exc).__name__}: {exc}"[:400],
        }
    keys = list(model.state_dict().keys())
    prefixes: dict[str, int] = {}
    for key in keys:
        prefix = ".".join(key.split(".")[:3])
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    return {
        "auto_class": auto_name,
        "available": True,
        "instantiated": True,
        "class": type(model).__name__,
        "tensor_count": len(keys),
        "prefixes": dict(sorted(prefixes.items())),
        "has_vision_tower": any(p.startswith(("model.visual", "visual")) for p in prefixes),
        "text_prefix_sample": [k for k in keys if "layers.0." in k][:4],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snap = snapshot_path()

    raw_config = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    preprocessor = json.loads((snap / "preprocessor_config.json").read_text(encoding="utf-8"))
    video_preprocessor = json.loads(
        (snap / "video_preprocessor_config.json").read_text(encoding="utf-8")
    )
    generation_config = json.loads(
        (snap / "generation_config.json").read_text(encoding="utf-8")
    )
    tokenizer_config = json.loads(
        (snap / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    index = json.loads((snap / "model.safetensors.index.json").read_text(encoding="utf-8"))
    checkpoint_keys = list(index["weight_map"])

    checkpoint_prefixes: dict[str, int] = {}
    for key in checkpoint_keys:
        prefix = ".".join(key.split(".")[:3])
        checkpoint_prefixes[prefix] = checkpoint_prefixes.get(prefix, 0) + 1

    config = AutoConfig.from_pretrained(str(snap))
    config_class = type(config).__name__

    audit: dict = {
        "phase": "17",
        "stage": "3",
        "deliverable": "architecture_audit.json",
        "scope": "Qwen/Qwen3.8-27B only",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "snapshot_path": str(snap),
        "checkpoint_config": {
            "architectures": raw_config.get("architectures"),
            "model_type": raw_config.get("model_type"),
            "language_model_only": raw_config.get("language_model_only"),
            "tie_word_embeddings": raw_config.get("tie_word_embeddings"),
            "image_token_id": raw_config.get("image_token_id"),
            "video_token_id": raw_config.get("video_token_id"),
            "vision_start_token_id": raw_config.get("vision_start_token_id"),
            "vision_end_token_id": raw_config.get("vision_end_token_id"),
            "transformers_version": raw_config.get("transformers_version"),
            "text_config": {
                "model_type": raw_config["text_config"].get("model_type"),
                "hidden_size": raw_config["text_config"].get("hidden_size"),
                "num_hidden_layers": raw_config["text_config"].get("num_hidden_layers"),
                "intermediate_size": raw_config["text_config"].get("intermediate_size"),
                "num_attention_heads": raw_config["text_config"].get("num_attention_heads"),
                "num_key_value_heads": raw_config["text_config"].get("num_key_value_heads"),
                "vocab_size": raw_config["text_config"].get("vocab_size"),
                "max_position_embeddings": raw_config["text_config"].get("max_position_embeddings"),
                "mtp_num_hidden_layers": raw_config["text_config"].get("mtp_num_hidden_layers"),
                "layer_type_histogram": {
                    t: raw_config["text_config"]["layer_types"].count(t)
                    for t in sorted(set(raw_config["text_config"]["layer_types"]))
                },
            },
            "vision_config": raw_config.get("vision_config"),
            "generation_config": generation_config,
            "processor_class_declared": preprocessor.get("processor_class"),
            "image_processor_type": preprocessor.get("image_processor_type"),
            "video_processor_type": video_preprocessor.get("video_processor_type"),
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "model_max_length": tokenizer_config.get("model_max_length"),
            "chat_template_present": bool(tokenizer_config.get("chat_template")),
        },
        "checkpoint_tensors": {
            "count": len(checkpoint_keys),
            "declared_total_size_bytes": index["metadata"].get("total_size"),
            "name_roots": sorted({k.split(".")[0] for k in checkpoint_keys}),
            "prefixes": dict(sorted(checkpoint_prefixes.items())),
        },
        "auto_config": {"class": config_class, "model_type": getattr(config, "model_type", None)},
        "auto_class_resolution": {},
        "transformers_version": transformers.__version__,
    }

    for auto_name in CANDIDATE_AUTO_CLASSES:
        auto = getattr(transformers, auto_name, None)
        entry: dict = {"available": auto is not None}
        if auto is not None:
            mapping = getattr(auto, "_model_mapping", None)
            mapped = None
            if mapping is not None:
                try:
                    mapped = mapping.get(type(config), None)
                except Exception:  # noqa: BLE001
                    mapped = None
            entry["maps_config_to"] = getattr(mapped, "__name__", None)
            entry["structure"] = expected_structure(auto_name, config)
        audit["auto_class_resolution"][auto_name] = entry

    # Processor resolution (config/tokenizer files only; no weights).
    try:
        processor = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        audit["processor_probe"] = {
            "loaded": True,
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(tokenizer).__name__ if tokenizer else None,
            "image_processor_class": type(getattr(processor, "image_processor", None)).__name__,
            "video_processor_class": type(getattr(processor, "video_processor", None)).__name__,
            "declared_processor_class": preprocessor.get("processor_class"),
            "matches_declared": type(processor).__name__ == preprocessor.get("processor_class"),
            "has_chat_template": bool(
                getattr(tokenizer, "chat_template", None) or tokenizer_config.get("chat_template")
            ),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
            "pad_token": getattr(tokenizer, "pad_token", None),
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        }
    except Exception as exc:  # noqa: BLE001
        audit["processor_probe"] = {
            "loaded": False,
            "error": f"{type(exc).__name__}: {exc}"[:400],
        }

    # The structural verdict this stage exists to establish.
    native = audit["auto_class_resolution"].get("AutoModelForImageTextToText", {})
    native_structure = native.get("structure") or {}
    causal = audit["auto_class_resolution"].get("AutoModelForCausalLM", {})
    causal_structure = causal.get("structure") or {}
    checkpoint_has_vision = any(
        p.startswith("model.visual") for p in checkpoint_prefixes
    )
    audit["structural_verdict"] = {
        "checkpoint_declares": raw_config.get("architectures"),
        "checkpoint_contains_vision_tower": checkpoint_has_vision,
        "native_class": native_structure.get("class"),
        "native_class_expects_vision_tower": native_structure.get("has_vision_tower"),
        "native_text_prefix_sample": native_structure.get("text_prefix_sample"),
        "causallm_class": causal_structure.get("class"),
        "causallm_expects_vision_tower": causal_structure.get("has_vision_tower"),
        "causallm_text_prefix_sample": causal_structure.get("text_prefix_sample"),
        "interpretation": (
            "A class whose expected parameter names match the checkpoint's own "
            "index can consume the published checkpoint as-is. A class that expects "
            "a different naming structure is a substitution, whatever transformers "
            "reports about missing keys."
        ),
    }

    path = OUT_DIR / "architecture_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"declared architecture     : {raw_config.get('architectures')}")
    print(f"model_type                : {raw_config.get('model_type')}  "
          f"(vision tower in checkpoint: {checkpoint_has_vision})")
    print(f"processor class (declared): {preprocessor.get('processor_class')}")
    print(f"processor class (loaded)  : {(audit.get('processor_probe') or {}).get('processor_class')}")
    for auto_name, entry in audit["auto_class_resolution"].items():
        structure = entry.get("structure") or {}
        print(f"  {auto_name:30s} -> {entry.get('maps_config_to')!s:34s} "
              f"instantiated={structure.get('instantiated')} "
              f"class={structure.get('class')} vision={structure.get('has_vision_tower')}")
    print(f"wrote {path.relative_to(REPO_ROOT)}")

    # Loader probe artifact: which path is the native one, and does it match.
    loader_probe = {
        "phase": "17",
        "stage": "3",
        "deliverable": "native_loader_probe.json",
        "model_id": MODEL_ID,
        "revision": REVISION,
        "question": (
            "Which loading API resolves this checkpoint to its declared "
            "Qwen3_5ForConditionalGeneration class, and does that class's expected "
            "parameter structure match the published checkpoint?"
        ),
        "results": {
            name: {
                "maps_config_to": entry.get("maps_config_to"),
                "instantiated_class": (entry.get("structure") or {}).get("class"),
                "expects_vision_tower": (entry.get("structure") or {}).get("has_vision_tower"),
                "text_prefix_sample": (entry.get("structure") or {}).get("text_prefix_sample"),
            }
            for name, entry in audit["auto_class_resolution"].items()
        },
        "checkpoint_prefixes": dict(sorted(checkpoint_prefixes.items())),
        "selection": {
            "chosen_auto_class": "AutoModelForImageTextToText",
            "why": (
                "It is the auto-class for image-text-to-text checkpoints (the "
                "checkpoint's own pipeline_tag and its declared "
                "Qwen3VLProcessor), and it resolves Qwen3_5Config to "
                "Qwen3_5ForConditionalGeneration directly. "
                "AutoModelForMultimodalLM resolves identically in this "
                "transformers version; AutoModelForCausalLM resolves a different "
                "class and is not the native path."
            ),
            "not_chosen": {
                "AutoModelForCausalLM": "resolves Qwen3_5ForCausalLM — the Stage 2 substitution",
                "AutoModelForMultimodalLM": "resolves identically here; kept as the "
                                            "existing project seam used by the Ornith backend",
            },
        },
        "native_architecture_supported": native.get("maps_config_to")
        == "Qwen3_5ForConditionalGeneration",
    }
    probe_path = OUT_DIR / "native_loader_probe.json"
    probe_path.write_text(json.dumps(loader_probe, indent=2), encoding="utf-8")
    print(f"native architecture supported by installed transformers: "
          f"{loader_probe['native_architecture_supported']}")
    print(f"wrote {probe_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
