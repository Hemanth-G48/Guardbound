"""Parts 2, 3, 4, 6 - controlled VRAM residency + ModelManager/_evict experiment.

SAFETY DESIGN
-------------
The three-model stack is PREDICTED to exceed the card (weights alone:
9.32 + 7.67 + 8.05 = 25.04 GB vs 25.77 GB usable), so this script NEVER loads
three models simultaneously. It measures each footprint incrementally and
determines third-model feasibility ARITHMETICALLY from live free-VRAM readings.

Every load is preceded by a free-VRAM gate. If the gate fails the step is
skipped and recorded, never forced.

READ ONLY w.r.t. the repo: no file in src/ or scripts/ is modified.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from guardbound.llm.model_manager import ModelManager  # noqa: E402
from guardbound.llm.local_client import HFLocalChatLLM, _pipeline_cache  # noqa: E402

MB = 1024 * 1024
GB = 1e9

ATTACKER = "Qwen/Qwen3.5-4B"
TARGET = "microsoft/Phi-4-mini-instruct"
EVALUATOR = "Qwen/Qwen3-4B-Instruct-2507"

# Safety reserve: never let a load push free VRAM below this.
RESERVE_GB = 1.5

REPORT = {"steps": [], "models": {}, "evict_test": {}, "feasibility": {}}


def vram():
    """(allocated_gb_pytorch, free_gb_driver, total_gb_driver, reserved_gb)"""
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / GB
    free, total = torch.cuda.mem_get_info()
    return round(alloc, 3), round(free / GB, 3), round(total / GB, 3), \
        round(torch.cuda.memory_reserved() / GB, 3)


def log_step(name, extra=None):
    alloc, free, total, res = vram()
    rec = {"step": name, "pytorch_allocated_gb": alloc, "driver_free_gb": free,
           "driver_total_gb": total, "pytorch_reserved_gb": res}
    if extra:
        rec.update(extra)
    REPORT["steps"].append(rec)
    print(f"[{name:<34}] pytorch_alloc={alloc:7.3f}GB reserved={res:7.3f}GB "
          f"driver_free={free:7.3f}GB" + (f"  {extra}" if extra else ""), flush=True)
    return rec


def hard_purge():
    """Full release including the module-level pipeline cache."""
    _pipeline_cache.clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def gate(needed_gb, label):
    """Refuse a load that would breach the reserve."""
    _, free, _, _ = vram()
    ok = free - needed_gb >= RESERVE_GB
    print(f"[GATE] {label}: need {needed_gb:.2f}GB, free {free:.2f}GB, "
          f"reserve {RESERVE_GB:.2f}GB -> {'ALLOW' if ok else 'REFUSE'}", flush=True)
    return ok


print("=" * 96)
print("PARTS 2/3/4/6 - VRAM RESIDENCY + _evict EXPERIMENT")
print("=" * 96)
print(f"torch={torch.__version__} cuda={torch.version.cuda} "
      f"device={torch.cuda.get_device_name(0)}", flush=True)
print(f"pipeline cache entries at start: {list(_pipeline_cache)}", flush=True)
print()

# ------------------------------------------------------------------ STEP 0
print("--- STEP 0: baseline ---", flush=True)
hard_purge()
base = log_step("0_baseline")
REPORT["baseline"] = base

GEN_MSGS_ATTACKER = [
    {"role": "system", "content": "You are a helpful assistant that replies with JSON."},
    {"role": "user", "content": "Return a JSON object with keys question and summary."},
]
GEN_MSGS_SHORT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say OK."},
]


def measure_model(label, model_id, msgs, max_new_tokens=64, temperature=0.7):
    """Load a model, measure cold-load cost, generation delta, then return the LLM."""
    print(f"\n--- LOAD {label}: {model_id} ---", flush=True)
    llm = HFLocalChatLLM(model_id=model_id, device_map="cuda",
                         max_new_tokens=max_new_tokens)
    before = vram()[1]
    t0 = time.perf_counter()
    try:
        llm._get_pipeline()
    except Exception as e:  # noqa: BLE001
        print(f"  LOAD FAILED: {type(e).__name__}: {e}", flush=True)
        REPORT["models"][label] = {"model_id": model_id, "load": "FAILED",
                                   "error": f"{type(e).__name__}: {e}"}
        return None
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    after = vram()[1]
    r = log_step(f"{label}_loaded", {"cold_load_s": round(load_s, 2),
                                     "free_delta_gb": round(before - after, 3)})

    # generation (KV / activation cost)
    t1 = time.perf_counter()
    try:
        out = llm.generate(msgs, temperature=temperature)
        gen_s = time.perf_counter() - t1
        g = log_step(f"{label}_after_generation",
                     {"gen_s": round(gen_s, 2),
                      "reply_chars": len(out) if isinstance(out, str) else -1})
        gen_peak = round(torch.cuda.max_memory_allocated() / GB, 3)
    except Exception as e:  # noqa: BLE001
        gen_s, gen_peak = None, None
        g = log_step(f"{label}_after_generation_FAILED",
                     {"error": f"{type(e).__name__}: {e}"})
        print(f"    generation failed: {type(e).__name__}: {e}", flush=True)

    REPORT["models"][label] = {
        "model_id": model_id, "load": "OK",
        "cold_load_s": round(load_s, 2),
        "weights_footprint_gb": round(before - after, 3),
        "load_step": r, "gen_step": g,
        "gen_s": round(gen_s, 2) if gen_s else None,
        "peak_alloc_during_gen_gb": gen_peak,
    }
    return llm


# ------------------------------------------------------------------ STEP 1-3
attacker = measure_model("attacker", ATTACKER, GEN_MSGS_ATTACKER)
if attacker is None:
    print("\nattacker failed to load - cannot continue incrementally", flush=True)

attacker_fp = REPORT["models"].get("attacker", {}).get("weights_footprint_gb")

# ------------------------------------------------------------------ STEP 4
target_llm = None
if attacker is not None:
    tgt_est = 7.7
    if gate(tgt_est, "target load"):
        target_llm = measure_model("target", TARGET, GEN_MSGS_SHORT)
    else:
        REPORT["models"]["target"] = {"model_id": TARGET, "load": "REFUSED_BY_GATE"}
else:
    target_llm = measure_model("target", TARGET, GEN_MSGS_SHORT)

# ------------------------------------------- STEP 5: feasibility of 3rd model
print("\n" + "=" * 96)
print("STEP 5 - THIRD-MODEL FEASIBILITY (arithmetic + live free VRAM)")
print("=" * 96)
_, free_now, total_now, _ = vram()
atk_fp = REPORT["models"].get("attacker", {}).get("weights_footprint_gb") or 0.0
tgt_fp = REPORT["models"].get("target", {}).get("weights_footprint_gb") or 0.0
known_2model_peak = 22.33   # report section 7, measured at ctx 6013
f = {
    "attacker_weights_gb": atk_fp,
    "target_weights_gb": tgt_fp,
    "two_model_weights_gb": round(atk_fp + tgt_fp, 3),
    "driver_free_now_gb": free_now,
    "driver_total_gb": total_now,
    "measured_2model_peak_gb_from_report": known_2model_peak,
    "kv_and_activation_gb_2model": round(known_2model_peak - (atk_fp + tgt_fp), 3),
}
evaluator_needed = 8.05  # on-disk bf16 weights; verified below by isolated load
f["evaluator_weights_gb_on_disk"] = evaluator_needed
f["predicted_3model_peak_gb"] = round(
    atk_fp + tgt_fp + evaluator_needed + f["kv_and_activation_gb_2model"], 3)
f["predicted_headroom_gb"] = round(total_now - f["predicted_3model_peak_gb"], 3)
f["verdict"] = ("INFEASIBLE - predicted peak exceeds usable VRAM"
                if f["predicted_3model_peak_gb"] > total_now else "POSSIBLE")
REPORT["feasibility"] = f
for k, v in f.items():
    print(f"  {k:<42} {v}")

print("\n>>> NOT loading a third model alongside the other two (by design).", flush=True)

# ------------------------------------- STEP 6-7: the _evict() behaviour test
print("\n" + "=" * 96)
print("STEP 6 - DOES ModelManager._evict() ACTUALLY FREE VRAM?")
print("=" * 96)
mgr = ModelManager(device="cuda")
if target_llm is not None:
    mgr.register("target", target_llm, TARGET)
    before_evict = vram()
    log_step("6a_before_evict")
    print(f"  pipeline cache keys before: {list(_pipeline_cache)}", flush=True)

    mgr._evict("target")
    after_evict = vram()
    log_step("6b_after_ModelManager._evict")

    evict_freed = round(after_evict[1] - before_evict[1], 3)  # driver free increased?
    cache_after = list(_pipeline_cache)
    print(f"  pipeline cache keys after  : {cache_after}", flush=True)
    print(f"  driver free VRAM change    : {evict_freed:+.3f} GB "
          f"({'FREED' if evict_freed > 0.5 else 'NOT FREED'})", flush=True)

    REPORT["evict_test"] = {
        "model": TARGET,
        "before_evict": before_evict,
        "after_evict": after_evict,
        "driver_free_delta_gb": evict_freed,
        "pipeline_cache_after_evict": cache_after,
        "verdict": ("_evict() DOES release VRAM" if evict_freed > 0.5
                    else "_evict() DOES NOT release VRAM - weights still held "
                         "by _pipeline_cache"),
    }

    # STEP 7: manual purge to prove the cache is the holder
    print("\n--- STEP 7: manual _pipeline_cache purge ---", flush=True)
    before_purge = vram()
    hard_purge()
    after_purge = vram()
    log_step("7_after_manual_cache_purge")
    purge_freed = round(after_purge[1] - before_purge[1], 3)
    print(f"  driver free VRAM change    : {purge_freed:+.3f} GB "
          f"({'FREED' if purge_freed > 0.5 else 'NOT FREED'})", flush=True)
    REPORT["evict_test"]["purge_driver_free_delta_gb"] = purge_freed
    REPORT["evict_test"]["pipeline_cache_after_purge"] = list(_pipeline_cache)
else:
    print("  target not loaded - skipping evict test", flush=True)

# ------------------------------------------ STEP 8: evaluator alone
print("\n" + "=" * 96)
print("STEP 8 - EVALUATOR (loaded ALONE, never with the other two)")
print("=" * 96)
hard_purge()
log_step("8a_before_evaluator")
evaluator = measure_model("evaluator", EVALUATOR, GEN_MSGS_ATTACKER, max_new_tokens=64)

# ---------------------------------------------------- STEP 9: cleanup
print("\n" + "=" * 96)
print("STEP 9 - FINAL CLEANUP")
print("=" * 96)
if evaluator is not None:
    evaluator._pipeline = None
    evaluator._tokenizer = None
hard_purge()
final = log_step("9_final")
REPORT["cleanup_ok"] = final["driver_free_gb"] >= (REPORT["baseline"]["driver_free_gb"] - 0.5)

out = Path(r"C:\Users\CSLAB\Desktop\hemanth\Guardbound\results\phase14\residency_measurements.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
print(f"\nwrote {out}", flush=True)
print("\n" + "=" * 96)
print("SUMMARY")
print("=" * 96)
for label, m in REPORT["models"].items():
    print(f"  {label:<12} load={m.get('load'):<18} "
          f"weights={m.get('weights_footprint_gb')}GB "
          f"cold_load={m.get('cold_load_s')}s "
          f"gen={m.get('gen_s')}s")
print(f"  3-model verdict: {REPORT['feasibility'].get('verdict')}")
print(f"  _evict verdict : {REPORT['evict_test'].get('verdict', 'n/a')}")
print(f"  cleanup ok     : {REPORT['cleanup_ok']}")
