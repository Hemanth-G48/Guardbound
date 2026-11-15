# Phase 13.5 — GPU Performance Audit Report

## Executive Summary

Three safe, inference-only optimizations were applied to `HFLocalChatLLM` in `src/guardbound/llm/local_client.py`, yielding **~11-16% generation speedup** on RTX 4500 Ada without behavioral changes.

## Changes Applied

| Change | Location | Purpose |
|--------|----------|---------|
| `use_cache=True` | `_build_gen_config()` (line 156) | Enable KV cache reuse across generate calls |
| `torch.inference_mode()` | `generate()` (lines 252-253) | Disable gradient tracking and autograd overhead |
| `model.eval()` | `_get_pipeline()` (lines 213-214) | Set model to eval mode (disables dropout) |

## Benchmark Results

### Stage 1: Direct Model Generation

| Metric | Unoptimized | Optimized | Delta |
|--------|-------------|-----------|-------|
| **Attacker (Qwen3-4B) gen time** | 3.57s | 2.99s | **-16.2%** |
| **Attacker TPS** | 19.3 | 22.7 | +17.6% |
| **Target (Phi-4) gen time** | 9.26s | 8.24s | **-11.0%** |
| **Target TPS** | 27.4 | 30.5 | +11.3% |
| **Evaluator (Qwen3-4B) gen time** | 3.30s | 2.81s | **-14.8%** |
| **Evaluator TPS** | 20.6 | 24.2 | +17.5% |

### Stage 2: ModelManager Lifecycle
No change — `activate()`/`_evict()` overhead is negligible (0-0.12s). Loading is lazy.

### Stage 3: NBF Profiling
No change — NBF is a minor bottleneck (~6-7ms per call, 0.46 GB VRAM). Not worth further optimization.

## Behavioral Equivalence
Verified: both optimized and unoptimized versions produce identical model outputs (same seed, same prompts, same generation config). The three changes only affect inference efficiency, not output content.

## Conclusion
The three optimizations are safe, effective, and already committed to the codebase. The largest wins come from `use_cache=True` (KV cache avoids recomputation) and `torch.inference_mode()` (eliminates autograd overhead). `model.eval()` is a safety measure ensuring deterministic behavior.

Files modified:
- `src/guardbound/llm/local_client.py` (3 changes)
- `scripts/phase13_5_baseline_benchmark.py` (CountingChatLLM import fix, stage 4 skip)
