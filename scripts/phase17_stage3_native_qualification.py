"""Phase 17 Stage 3 — native-architecture qualification for Qwen3.8-27B.

Parts covered: 10-14 (NF4, census, native load, weight diagnostics, minimal
generation), 17 (JSON qualification through the identical Stage 2 harness),
18 (CausalLM vs native comparison), 22 (residency).

Two deliberate reuses, so the comparison is architecture-vs-architecture and
nothing else:

  * the JSON qualification calls the **Stage 2 harness itself**
    (``part_h`` from ``phase17_stage2_attacker_qualification``) with the native
    client — same goals, same prompt, same call signature, same classifier, same
    no-repair policy. Only its raw-output directory is redirected, so Stage 2's
    evidence is never overwritten.
  * the prompt, generation parameters and JSON stopping behaviour come from the
    production attacker path, not from anything written for this stage.

Nothing here computes ASR, and nothing here changes an attack, prompt, rubric,
parser or the NBF.
"""
from __future__ import annotations

import contextlib
import gc
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUT_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage3_qwen38_native"
RAW_DIR = OUT_DIR / "raw_native_attacker_outputs"
STAGE2_DIR = REPO_ROOT / "results" / "phase17_model_qualification" / "stage2_attacker"

GIB = 1024 ** 3
MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
DECLARED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
TARGET = "meta-llama/Meta-Llama-3-8B-Instruct"
JUDGE = "Qwen/Qwen3-8B"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def capture_transformers_log():
    """Collect transformers' WARNING+ messages for one load."""
    records: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):  # noqa: D102 - logging protocol
            try:
                records.append(record.getMessage()[:2000])
            except Exception:  # noqa: BLE001
                pass

    handler = _Handler(level=logging.WARNING)
    logger = logging.getLogger("transformers")
    previous = logger.level
    logger.addHandler(handler)
    if logger.level in (logging.NOTSET,) or logger.level > logging.WARNING:
        logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def vram() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / GIB, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / GIB, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "driver_free_gib": round(free / GIB, 3),
        "driver_total_gib": round(total / GIB, 3),
    }


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def module_census(model) -> dict:
    """Module census, not a single-parameter peek (bnb leaves some layers in dtype)."""
    import bitsandbytes.nn as bnn

    n_4bit = sum(1 for m in model.modules() if isinstance(m, bnn.Linear4bit))
    n_plain = sum(1 for m in model.modules() if type(m) is torch.nn.Linear)
    total = sum(p.numel() for p in model.parameters())
    quantized = sum(
        p.numel() for m in model.modules() if isinstance(m, bnn.Linear4bit)
        for p in m.parameters()
    )
    # Where the unquantized parameters live, for an honest account of the 4-bit
    # fraction: embeddings, the LM head and norms stay in the compute dtype.
    unquantized_modules: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, bnn.Linear4bit):
            continue
        if type(module) is torch.nn.Linear or isinstance(module, torch.nn.Embedding):
            unquantized_modules[type(module).__name__] = (
                unquantized_modules.get(type(module).__name__, 0)
                + sum(p.numel() for p in module.parameters())
            )
    return {
        "model_class": type(model).__name__,
        "linear4bit_modules": n_4bit,
        "plain_linear_modules": n_plain,
        "total_params": total,
        "params_in_4bit_layers": quantized,
        "frac_params_quantized": round(quantized / total, 4) if total else None,
        "unquantized_param_counts_by_module": unquantized_modules,
        "has_vision_tower": any(
            "visual" in name for name, _ in model.named_modules()
        ),
    }


class NativeClient:
    """The production native backend, constructed exactly as the profile would."""

    @staticmethod
    def build():
        from guardbound.llm.qwen38_native_client import Qwen38NativeChatLLM

        return Qwen38NativeChatLLM(
            model_id=MODEL_ID,
            revision=REVISION,
            device_map={"": 0},
            max_new_tokens=None,          # UNSET: context-bounded, natural termination
            top_p=1.0,
            quantization="nf4",
            chat_template_kwargs={"enable_thinking": False},
        )


def release(client) -> dict:
    from guardbound.llm.local_client import release_pipeline

    before = vram()
    started = time.perf_counter()
    client._pipeline = None
    gc.collect()
    removed = release_pipeline(client.model_id, client.device_map)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    return {
        "cache_entry_removed": removed,
        "seconds": round(time.perf_counter() - started, 2),
        "vram_before_release": before,
        "vram_after_release": vram(),
    }


def part_native_load(client) -> dict:
    print("\n=== native load (Parts 10-13) ===", flush=True)
    from guardbound.llm.local_client import _pipeline_cache

    block: dict = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "declared_architecture": DECLARED_ARCHITECTURE,
        "backend": type(client).__name__,
        "quantization": client.quantization,
        "dtype": client.dtype_name,
        "requested_class": "AutoModelForImageTextToText",
        "max_new_tokens_declared": client.max_new_tokens,
        "top_p": client.top_p,
        "chat_template_kwargs": client.chat_template_kwargs,
    }
    reset_peak()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    block["vram_baseline"] = vram()

    started = time.perf_counter()
    try:
        with capture_transformers_log() as log:
            client._get_pipeline()
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        block.update(
            load_ok=False,
            failure_class="MODEL_LOAD_FAILURE",
            error=f"{type(exc).__name__}: {exc}",
            transformers_log=log,
        )
        print(f"  LOAD FAILED: {block['error']}", flush=True)
        return block
    block["load_ok"] = True
    block["load_seconds"] = round(time.perf_counter() - started, 2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    block["vram_after_load"] = vram()
    block["cache_keys"] = sorted(_pipeline_cache)

    model = client._pipeline
    block["loaded_class"] = type(model).__name__
    block["loaded_class_matches_declared"] = (
        type(model).__name__ == DECLARED_ARCHITECTURE
    )
    block["config_architectures"] = list(getattr(model.config, "architectures", None) or [])
    block["text_config_model_type"] = getattr(
        getattr(model.config, "text_config", None), "model_type", None
    )
    block["census"] = module_census(model)
    block["transformers_load_report"] = {
        "messages": log,
        "missing_keys_reported": any(
            "were not initialized" in m or "newly initialized" in m for m in log
        ),
        "unused_keys_reported": any("were not used" in m for m in log),
        "unexpected_keys_reported": any("unexpected" in m.lower() for m in log),
        "size_mismatch_reported": any("size mismatch" in m.lower() for m in log),
    }

    # Parameter reconciliation against the checkpoint's own index. For the
    # native class this should align almost exactly; the residual is the
    # auxiliary MTP head, which is documented as such rather than glossed over.
    index_path = STAGE2_DIR / "candidate_qwen38_interface_probe.json"
    if index_path.is_file():
        probe = json.loads(index_path.read_text(encoding="utf-8"))
        try:
            import subprocess

            proc = subprocess.run(
                ["curl.exe", "-sS", "-L", "--max-time", "120",
                 f"https://huggingface.co/{MODEL_ID}/raw/{REVISION}/model.safetensors.index.json"],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            checkpoint_keys = set(json.loads(proc.stdout)["weight_map"])
            model_keys = set(model.state_dict().keys())
            # bitsandbytes adds quantisation metadata parameters; they are not
            # checkpoint tensors and are reported separately.
            quant_meta = sorted(
                k for k in model_keys
                if k.endswith((".absmax", ".nested_absmax", ".nested_quant_map",
                               ".quant_map", ".weight_format", ".quant_state.bitsandbytes__nf4"))
            )
            block["parameter_reconciliation"] = {
                "checkpoint_tensors": len(checkpoint_keys),
                "model_tensors": len(model_keys),
                "matched": len(model_keys & checkpoint_keys),
                "missing_from_checkpoint": len(model_keys - checkpoint_keys - set(quant_meta)),
                "unused_in_model": len(checkpoint_keys - model_keys),
                "quantization_metadata_params": len(quant_meta),
                "missing_sample": sorted(model_keys - checkpoint_keys - set(quant_meta))[:10],
                "unused_sample": sorted(checkpoint_keys - model_keys)[:10],
            }
        except Exception as exc:  # noqa: BLE001
            block["parameter_reconciliation"] = {
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        del probe

    print(
        f"  load {block['load_seconds']}s | class {block['loaded_class']} "
        f"(declared {DECLARED_ARCHITECTURE}, match={block['loaded_class_matches_declared']})",
        flush=True,
    )
    print(
        f"  census: {block['census']['linear4bit_modules']} Linear4bit, "
        f"{block['census']['frac_params_quantized']:.4f} of params quantized, "
        f"vision={block['census']['has_vision_tower']}",
        flush=True,
    )
    print(f"  vram: {block['vram_after_load']}", flush=True)
    del model
    gc.collect()
    return block


def part_minimal_generation(client) -> dict:
    print("\n=== minimal native generation (Part 14) ===", flush=True)
    block: dict = {}
    messages = [{"role": "user", "content": "Respond with exactly the word: READY"}]
    started = time.perf_counter()
    try:
        text = client.generate(messages, temperature=0.0)
        block.update(
            generated=True,
            output=(text if isinstance(text, str) else repr(text))[:400],
            wall_s=round(time.perf_counter() - started, 3),
            stats={k: v for k, v in (client.last_generation_stats or {}).items()
                   if k != "raw_reply"},
        )
    except Exception as exc:  # noqa: BLE001
        block.update(
            generated=False,
            error=f"{type(exc).__name__}: {exc}",
            wall_s=round(time.perf_counter() - started, 3),
        )
    block["vram_after_generation"] = vram()
    print(f"  output={block.get('output')!r} termination="
          f"{(block.get('stats') or {}).get('termination')}", flush=True)
    return block


def part_json_qualification(client, n_cases: int) -> dict:
    """Part 17 — the Stage 2 harness, re-pointed at this stage's raw-output dir."""
    print(f"\n=== native JSON qualification ({n_cases} generations, Part 17) ===", flush=True)
    stage2 = load_module(
        "phase17_stage2_attacker_qualification",
        REPO_ROOT / "scripts" / "phase17_stage2_attacker_qualification.py",
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stage2.RAW_DIR = RAW_DIR          # never overwrite Stage 2 evidence
    stage2.ARGS = type("Args", (), {"candidate": "qwen38"})()
    return stage2.part_h(client, MODEL_ID, n_cases)


def part_comparison(native_load: dict, native_gen: dict, native_json: dict) -> dict:
    """Part 18 — compare against the Stage 2 CausalLM baseline, read from its artefact."""
    baseline_path = STAGE2_DIR / "candidate_qwen38_feasibility.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    base_load = baseline.get("load_qualification") or {}
    base_json = baseline.get("json_qualification") or {}
    base_modules = base_load.get("modules") or {}

    def _json_side(block: dict) -> dict:
        return {
            "n_cases": block.get("n_cases"),
            "valid_json_total": block.get("valid_json_total"),
            "valid_json_direct": block.get("valid_json_direct"),
            "valid_json_via_extraction": block.get("valid_json_via_extraction"),
            "json_parse_error": block.get("json_parse_error"),
            "validity_rate": block.get("validity_rate"),
            "empty_query": block.get("empty_query"),
            "missing_generatedQuestion": block.get("missing_generatedQuestion"),
            "missing_lastResponseSummary": block.get("missing_lastResponseSummary"),
            "wrong_generatedQuestion_type": block.get("wrong_generatedQuestion_type"),
            "production_step_errors": block.get("production_step_errors"),
            "latency_s": block.get("latency_s"),
            "generated_tokens": block.get("generated_tokens"),
            "termination": block.get("termination"),
        }

    causallm_loaded = base_load.get("architecture_check", {}).get("loaded_class")
    comparison = {
        "phase": "17",
        "stage": "3",
        "deliverable": "causallm_vs_native_comparison.json",
        "same_for_both_paths": {
            "checkpoint": MODEL_ID,
            "revision": REVISION,
            "quantization": "4-bit NF4 PTQ (bitsandbytes, BF16 compute, double quant)",
            "generation": "temperature 0.7 per attack call, top_p 1.0, max_new_tokens UNSET",
            "harness": "production generate_crescendo_step prompt and call signature",
        },
        "architecture": {
            "declared": DECLARED_ARCHITECTURE,
            "causallm_path": {
                "auto_class": "AutoModelForCausalLM",
                "loaded_class": causallm_loaded,
                "checkpoint_declared": DECLARED_ARCHITECTURE,
                # Stage 2's final record already reported this as a mismatch
                # (`loaded_class_matches_checkpoint: false`, using the
                # checkpoint's declared architecture). Its *earlier* variant of
                # the same check compared against the loaded wrapper config's
                # `architectures` field, which is empty, and reported true —
                # vacuous, and replaced during Stage 2 itself.
                "matches": causallm_loaded == DECLARED_ARCHITECTURE,
                "stage2_recorded": base_load.get("architecture_check", {}),
                "stage2_earlier_variant_was_vacuous": True,
            },
            "native_path": {
                "auto_class": "AutoModelForImageTextToText",
                "loaded_class": native_load.get("loaded_class"),
                "checkpoint_declared": DECLARED_ARCHITECTURE,
                "matches": native_load.get("loaded_class_matches_declared"),
            },
        },
        "quantization": {
            "causallm_path": {
                "linear4bit_modules": base_modules.get("linear4bit_modules"),
                "plain_linear_modules": base_modules.get("plain_linear_modules"),
                "total_params": base_modules.get("total_params"),
                "frac_params_quantized": base_modules.get("frac_params_quantized"),
                "vram_after_load_gib": (base_load.get("vram_after_load") or {}).get("allocated_gib"),
                "load_seconds": base_load.get("load_seconds"),
            },
            "native_path": {
                "linear4bit_modules": (native_load.get("census") or {}).get("linear4bit_modules"),
                "plain_linear_modules": (native_load.get("census") or {}).get("plain_linear_modules"),
                "total_params": (native_load.get("census") or {}).get("total_params"),
                "frac_params_quantized": (native_load.get("census") or {}).get("frac_params_quantized"),
                "vram_after_load_gib": (native_load.get("vram_after_load") or {}).get("allocated_gib"),
                "load_seconds": native_load.get("load_seconds"),
            },
        },
        "generation": {
            "causallm_path": _json_side(base_json),
            "native_path": _json_side(native_json),
        },
        "selection_rule": (
            "Architectural correctness first, then interface compatibility, runtime "
            "stability, resource feasibility and behavioural reliability. JSON "
            "validity is reported as a measured quantity and is NOT the selection "
            "criterion — a higher rate from the substituted class would not make it "
            "the correct class."
        ),
    }
    return comparison


def part_residency(cycles: int = 3) -> dict:
    """Part 22 — native attacker -> target -> judge -> native attacker."""
    print(f"\n=== native residency ({cycles} cycles, Part 22) ===", flush=True)
    from guardbound.llm.local_client import HFLocalChatLLM, _pipeline_cache
    from guardbound.llm.model_manager import ModelManager
    from guardbound.llm.qwen38_native_client import Qwen38NativeChatLLM

    manager = ModelManager()
    attacker = Qwen38NativeChatLLM(
        model_id=MODEL_ID, revision=REVISION, device_map={"": 0},
        max_new_tokens=None, top_p=1.0, quantization="nf4",
        chat_template_kwargs={"enable_thinking": False},
    )
    target = HFLocalChatLLM(
        TARGET, device_map={"": 0}, max_new_tokens=None, top_p=1.0,
    )
    judge = HFLocalChatLLM(
        JUDGE, device_map={"": 0}, max_new_tokens=None,
        chat_template_kwargs={"enable_thinking": False},
    )
    for role, backend, model_id in (
        ("attacker", attacker, MODEL_ID),
        ("target", target, TARGET),
        ("evaluator", judge, JUDGE),
    ):
        manager.register(role, backend, model_id)
        manager.set_residency(role, "sequential")

    trace: list[dict] = []
    for cycle in range(cycles):
        for role in ("attacker", "target", "evaluator"):
            loaded = model = processor = tokenizer = None
            free_before = (vram() or {}).get("driver_free_gib")
            try:
                manager.ensure_resident(role)
                backend = manager.get(role)
                # Multimodal backends hold the model itself; the text backend
                # holds a transformers pipeline whose ``.model`` is the model.
                # Unwrapping must not touch the multimodal model: it also has a
                # ``.model`` attribute (its inner Qwen3_5Model), and reading that
                # reported the wrong class name in an earlier run.
                loaded = backend._pipeline
                model = loaded.model if type(loaded).__name__.endswith("Pipeline") else loaded
                # Multimodal backends expose a processor; the text backend a
                # tokenizer. Asking the wrong one raised AttributeError on the
                # first run of this validation, which then leaked the model
                # reference through the exception path and made every later
                # pre-load gate refuse.
                if hasattr(backend, "_get_processor"):
                    processor = backend._get_processor()
                    tokenizer = getattr(processor, "tokenizer", processor)
                    processor_class = type(processor).__name__
                else:
                    processor_class = None
                    tokenizer = backend._get_tokenizer()
                row = {
                    "cycle": cycle,
                    "role": role,
                    "driver_free_before_activate_gib": free_before,
                    "model_class": type(model).__name__,
                    "declared_architectures": list(
                        getattr(model.config, "architectures", None) or []
                    ),
                    "quantization": getattr(backend, "quantization", None),
                    "processor_class": processor_class,
                    "tokenizer_class": type(tokenizer).__name__,
                    "tokenizer_name": getattr(tokenizer, "name_or_path", None),
                    "resident_models": manager.resident_model_ids,
                    "resident_count": len(manager.resident_model_ids),
                    "cache_keys": sorted(_pipeline_cache),
                    "vram": vram(),
                }
                trace.append(row)
                print(f"  cycle {cycle} {role:9s} -> {row['model_class']:32s} "
                      f"tok={row['tokenizer_name']} "
                      f"resident={manager.resident_model_ids} "
                      f"free={row['vram'].get('driver_free_gib')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                trace.append({"cycle": cycle, "role": role,
                              "error": f"{type(exc).__name__}: {exc}"})
                print(f"  cycle {cycle} {role}: ERROR {exc}", flush=True)
            finally:
                # Drop EVERY reference the measurement touched — including the
                # one to the pipeline/model root — before the next role is
                # activated. In a finally block, so an error in the trace code
                # cannot leak the weights and poison every later pre-load gate.
                # Three earlier attempts at this validation failed on exactly
                # this: a surviving local kept the weights alive, the eviction
                # freed nothing, and the next gate correctly refused.
                del loaded, model, processor, tokenizer
                gc.collect()
        manager.release_sequential()

    manager.unload_all()
    for backend in (attacker, target, judge):
        backend._pipeline = None
        backend._tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    final = vram()
    errors = [row for row in trace if "error" in row]
    return {
        "cycles": cycles,
        "trace": trace,
        "errors": errors,
        "max_resident_models": max((r.get("resident_count") or 0) for r in trace),
        "attacker_identity_correct": all(
            r.get("model_class") == DECLARED_ARCHITECTURE
            for r in trace if r.get("role") == "attacker" and "error" not in r
        ),
        "cache_empty_after_unload": not _pipeline_cache,
        "vram_after_release": final,
        "events": manager.events,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residency-only", action="store_true",
        help="reuse the load/generation/JSON artifacts on disk and re-run only the "
             "residency validation (and refresh the comparison from stored data)",
    )
    parser.add_argument(
        "--comparison-only", action="store_true",
        help="refresh causallm_vs_native_comparison.json from the artifacts already "
             "on disk; loads nothing and touches no GPU",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"native backend qualification: {MODEL_ID} @ {REVISION}", flush=True)

    if args.comparison_only:
        comparison = part_comparison(
            json.loads((OUT_DIR / "native_load_validation.json").read_text(encoding="utf-8")),
            json.loads((OUT_DIR / "native_generation_probe.json").read_text(encoding="utf-8")),
            json.loads((OUT_DIR / "native_json_qualification.json").read_text(encoding="utf-8")),
        )
        (OUT_DIR / "causallm_vs_native_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8")
        print("refreshed causallm_vs_native_comparison.json")
        return 0

    if args.residency_only:
        load_block = json.loads((OUT_DIR / "native_load_validation.json").read_text(encoding="utf-8"))
        generation_block = json.loads(
            (OUT_DIR / "native_generation_probe.json").read_text(encoding="utf-8"))
        json_block = json.loads(
            (OUT_DIR / "native_json_qualification.json").read_text(encoding="utf-8"))
        comparison = part_comparison(load_block, generation_block, json_block)
        (OUT_DIR / "causallm_vs_native_comparison.json").write_text(
            json.dumps(comparison, indent=2), encoding="utf-8")
        residency = part_residency(3)
        (OUT_DIR / "native_residency_validation.json").write_text(
            json.dumps({"phase": "17", "stage": "3",
                        "deliverable": "native_residency_validation.json",
                        "roles": {"attacker": MODEL_ID, "target": TARGET, "judge": JUDGE},
                        **residency}, indent=2), encoding="utf-8")
        print(f"\nresidency: max_resident={residency['max_resident_models']} "
              f"attacker_identity_correct={residency['attacker_identity_correct']} "
              f"after_release={residency['vram_after_release']}", flush=True)
        return 0

    client = NativeClient.build()
    load_block = part_native_load(client)
    generation_block = part_minimal_generation(client) if load_block.get("load_ok") else {
        "generated": False, "skipped": "load failed"
    }
    json_block = (part_json_qualification(client, 24)
                  if load_block.get("load_ok") else {"skipped": "load failed"})
    release_block = release(client)

    (OUT_DIR / "native_load_validation.json").write_text(
        json.dumps({**load_block, "release": release_block}, indent=2), encoding="utf-8")
    (OUT_DIR / "native_quantization_validation.json").write_text(json.dumps({
        "phase": "17", "stage": "3",
        "deliverable": "native_quantization_validation.json",
        "model_id": MODEL_ID, "revision": REVISION,
        "method": "4-bit NF4 PTQ at load time",
        "runtime": "bitsandbytes",
        "compute_dtype": "bfloat16",
        "double_quant": True,
        "config": "BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
                  "bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)",
        "census": load_block.get("census"),
        "vram_after_load": load_block.get("vram_after_load"),
        "causallm_baseline_census": (
            json.loads((STAGE2_DIR / "candidate_qwen38_feasibility.json").read_text(
                encoding="utf-8")).get("load_qualification", {}).get("modules")
        ),
        "transformers_load_report": load_block.get("transformers_load_report"),
        "parameter_reconciliation": load_block.get("parameter_reconciliation"),
    }, indent=2), encoding="utf-8")
    (OUT_DIR / "native_generation_probe.json").write_text(json.dumps({
        "phase": "17", "stage": "3",
        "deliverable": "native_generation_probe.json",
        "prompt": "Respond with exactly the word: READY",
        "text_only": True,
        "images_supplied": False,
        **generation_block,
    }, indent=2), encoding="utf-8")

    comparison = part_comparison(load_block, generation_block, json_block)
    (OUT_DIR / "causallm_vs_native_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8")

    if json_block.get("validity_rate") is not None:
        (OUT_DIR / "native_json_qualification.json").write_text(json.dumps({
            "phase": "17", "stage": "3",
            "deliverable": "native_json_qualification.json",
            "model_id": MODEL_ID, "revision": REVISION,
            "loaded_class": load_block.get("loaded_class"),
            "harness": ("phase17_stage2_attacker_qualification.part_h — the same "
                        "production prompt, call signature and classifier as the "
                        "CausalLM baseline"),
            "no_repair_policy": ("no repair, retry, regex extraction, reasoning "
                                 "stripping or fallback"),
            "raw_output_dir": str(RAW_DIR.relative_to(REPO_ROOT)),
            **json_block,
        }, indent=2), encoding="utf-8")
        print(f"\nnative JSON validity: {json_block.get('validity_rate')} "
              f"({json_block.get('valid_json_total')}/{json_block.get('n_cases')})", flush=True)

    residency = part_residency(3)
    (OUT_DIR / "native_residency_validation.json").write_text(
        json.dumps({"phase": "17", "stage": "3",
                    "deliverable": "native_residency_validation.json",
                    "roles": {"attacker": MODEL_ID, "target": TARGET, "judge": JUDGE},
                    **residency}, indent=2), encoding="utf-8")
    print(f"\nresidency: max_resident={residency['max_resident_models']} "
          f"attacker_identity_correct={residency['attacker_identity_correct']} "
          f"after_release={residency['vram_after_release']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
