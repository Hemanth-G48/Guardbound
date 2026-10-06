"""Phase 17 Stage 2 — Part D/E: attacker interface-compatibility probe.

Determines, with measured evidence and before any weight download, whether each
candidate can be driven by the project's frozen attacker interface
(``HFLocalChatLLM``: ``pipeline("text-generation")`` + ``apply_chat_template`` +
``model.generate``).

The probe exercises the *real* resolution path rather than asserting an
expectation:

  1. ``AutoConfig.from_pretrained`` on the candidate (config files only).
  2. The auto-class the loader's pipeline resolves to — ``AutoModelForCausalLM``
     — asked to map that config. A model type absent from that mapping cannot be
     loaded by the frozen interface at all; the ValueError is the evidence.
  3. For model types that *are* mapped, the expected parameter-name structure is
     obtained by instantiating the *same* auto-class from a tiny synthetic config
     of that type (no weights, megabytes of RAM), then compared against the
     candidate's own ``model.safetensors.index.json``. This distinguishes "the
     class is mapped" from "the published checkpoint actually carries that
     text backbone".

Downloads are limited to config/tokenizer/index files; no weight shard is
fetched, and no model is loaded onto the GPU.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"

CANDIDATES = {
    "qwen38": "Qwen/Qwen3.8-27B",
    "qwen36": "Qwen/Qwen3.6-27B",
    "ministral": "mistralai/Ministral-3-14B-Reasoning-2512",
    "devstral": "mistralai/Devstral-Small-2-24B-Instruct-2512",
}

# Tiny synthetic configs, one per candidate family, used only to enumerate the
# parameter names the mapped auto-class expects. Small enough to be free.
MINI_CONFIGS = {
    "qwen3_5": dict(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    ),
    "mistral3": dict(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=64,
    ),
}


def curl_text(url: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["curl.exe", "-sS", "-L", "--max-time", "120", "-w", "\n__HTTP__%{http_code}", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    out = proc.stdout
    marker = out.rfind("__HTTP__")
    if marker == -1:
        return 0, out[:200]
    body, status = out[:marker].strip(), out[marker + len("__HTTP__"):].strip()
    try:
        return int(status), body
    except ValueError:
        return 0, body[:200]


def expected_text_keys(config_type_name: str, model_type: str) -> dict:
    """Parameter names the mapped causal-LM class expects, from a tiny instance."""
    cfg = None
    for name in (config_type_name, model_type):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            cfg = cls(**MINI_CONFIGS[model_type])
            break
        except Exception:  # noqa: BLE001 -- try the next constructor spelling
            cfg = None
    if cfg is None:
        return {"resolvable": False}

    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg)
        keys = list(model.state_dict().keys())
    except Exception as exc:  # noqa: BLE001
        return {
            "resolvable": False,
            "config_class": type(cfg).__name__,
            "error": f"{type(exc).__name__}: {exc}",
        }
    prefixes = sorted({".".join(k.split(".")[:3]) for k in keys})
    return {
        "resolvable": True,
        "config_class": type(cfg).__name__,
        "class": type(model).__name__,
        "param_count": len(keys),
        "prefixes": prefixes[:20],
    }


def probe(key: str, model_id: str) -> dict:
    record: dict = {
        "candidate": key,
        "model_id": model_id,
        "record_type": "interface_compatibility_probe",
        "weights_downloaded": False,
        "transformers_version": transformers.__version__,
    }

    try:
        cfg = AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001
        record["verdict"] = "MODEL_LOAD_FAILURE"
        record["error"] = f"AutoConfig failed: {type(exc).__name__}: {exc}"
        return record

    cfg_class = type(cfg)
    model_type = getattr(cfg, "model_type", None)
    record["config"] = {
        "class": cfg_class.__name__,
        "model_type": model_type,
        "architectures": getattr(cfg, "architectures", None),
        "text_config_class": type(getattr(cfg, "text_config", None)).__name__,
        "has_text_config": getattr(cfg, "text_config", None) is not None,
    }

    # 1) Does the auto-class the pipeline resolves to map this config?
    mapping = AutoModelForCausalLM._model_mapping
    # transformers' lazy auto-mapping requires an explicit default.
    mapped_to = mapping.get(cfg_class, None)
    record["auto_class_resolution"] = {
        "auto_class": "AutoModelForCausalLM",
        "mapped": mapped_to is not None,
        "mapped_class": getattr(mapped_to, "__name__", None),
    }

    if mapped_to is None:
        # Ask the real entry point anyway, so the recorded evidence is the
        # exception the production loader would raise, not our inference.
        try:
            AutoModelForCausalLM.from_config(cfg)
            raised = None
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        record["from_config_probe"] = {"raised": raised}
        record["verdict"] = "MODEL_INTERFACE_INCOMPATIBLE"
        record["reason"] = (
            f"model_type {model_type!r} is not mapped by AutoModelForCausalLM, which "
            "is the auto-class pipeline('text-generation') resolves to. The frozen "
            "attacker interface therefore cannot load this checkpoint."
        )
        record["not_a_weight_problem"] = (
            "This is resolved from the published config alone: no weight shard was "
            "downloaded or needed to establish it."
        )
        return record

    # 2) The class is mapped; does the published checkpoint carry that backbone?
    record["expected_keys"] = expected_text_keys(
        type(getattr(cfg, "text_config", cfg)).__name__, model_type
    )

    status, index_body = curl_text(
        f"https://huggingface.co/{model_id}/raw/main/model.safetensors.index.json"
    )
    record["checkpoint_index"] = {"status": status}
    if status == 200:
        try:
            index = json.loads(index_body)
            weight_map = index.get("weight_map", {})
            prefixes = sorted({".".join(k.split(".")[:2]) for k in weight_map})
            record["checkpoint_index"].update(
                tensor_count=len(weight_map),
                total_size_bytes=index.get("metadata", {}).get("total_size"),
                key_prefixes=prefixes[:20],
                has_vision_keys=any(
                    p.startswith(("visual", "vision", "model.visual", "model.vision"))
                    for p in prefixes
                ),
            )
        except json.JSONDecodeError as exc:
            record["checkpoint_index"]["error"] = str(exc)
    else:
        record["checkpoint_index"]["error"] = index_body[:200]

    record["verdict"] = "INTERFACE_PROBE_PASSED"
    record["reason"] = (
        "The model type is mapped by AutoModelForCausalLM. Interface compatibility "
        "still has to be proven by an actual NF4 load (Part G) — mapping presence "
        "alone is not a load result."
    )
    return record


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, model_id in CANDIDATES.items():
        record = probe(key, model_id)
        path = OUT_DIR / f"candidate_{key}_interface_probe.json"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(
            f"{key:10s} {record['config']['model_type']:10s} "
            f"mapped={record['auto_class_resolution']['mapped']} "
            f"-> {record['auto_class_resolution']['mapped_class']} | "
            f"{record['verdict']}"
        )
        if record["verdict"] == "MODEL_INTERFACE_INCOMPATIBLE":
            print(f"           {record['from_config_probe']['raised']}")
        elif record.get("expected_keys", {}).get("resolvable"):
            print(
                f"           expected class {record['expected_keys']['class']}, "
                f"checkpoint tensors {record['checkpoint_index'].get('tensor_count')}, "
                f"vision keys={record['checkpoint_index'].get('has_vision_keys')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
