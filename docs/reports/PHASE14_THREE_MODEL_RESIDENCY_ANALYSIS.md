# Phase 14 — Three-Model Residency Analysis

**Date:** 2026-09-23
**Machine:** `DESKTOP-PVCRPVT` — HP Z2 Tower G1i Workstation
**GPU:** NVIDIA RTX 4500 Ada Generation — `24570 MiB` = **25.763 GB** decimal, **24.385 GB free at idle**
**Driver:** 595.79 · **torch:** 2.7.1+cu118 · **CUDA runtime:** 11.8 · **Python:** 3.13.9 (anaconda3 base)
**Scope:** architecture investigation ONLY — no production code modified, no Phase 14 / Batch 0 launched.
**Related:** `crash_investigation/PHASE14_CRASH_ROOT_CAUSE.md`

---

## 0. Headline

1. **`Qwen/Qwen3.5-4B-Instruct` does not exist.** The Qwen3.5 family has no `-Instruct` suffix; `Qwen/Qwen3.5-4B` **is** the instruction-tuned model and `Qwen/Qwen3.5-4B-Base` is the base. The instruction "use the Instruct variant, do NOT substitute base `Qwen3.5-4B`" is based on a naming misreading — `Qwen3.5-4B` is already the correct, non-base model. See §1.
2. **All three resident is arithmetically impossible.** Measured weight footprints sum to **24.118 GB** against **24.385 GB** free at idle. With the KV/workspace a real run needs (2.680 GB measured), the predicted peak is **26.798 GB** — over by **~1.04 GB**. This independently reproduces the crash report's "~1.46 GB over" figure.
3. **`ModelManager._evict()` frees exactly 0.000 GB.** Proven empirically, not inferred. A real eviction requires **two** releases (backend attribute *and* pipeline-cache entry), and only then does VRAM drop — at the cost of a full cold reload. See §7.
4. **No architecture in the brief is viable as literally specified.** Option A is infeasible; Options B and C are infeasible in their "keep two resident, add the third" form. What works is a **2-slot rotation**, and the cheapest one is: **pin the attacker, rotate target ↔ evaluator** — measured **22.15 s (warm) to 36.43 s (cold) per turn**.
5. **`device_map="cpu"` does not keep a model off the GPU** — it silently loads onto `cuda:0`. Only `device="cpu"` works. This is a latent trap and it invalidated one of my own earlier measurements; see §6.

---

## 1. Intended model stack — verification

### 1.1 The requested attacker id does not exist

| Check | Result |
|---|---|
| `GET /api/models/Qwen/Qwen3.5-4B-Instruct` | **HTTP 403** (gated or non-existent) |
| Hub search `Qwen3.5-4B` (30 results) | no `-Instruct` variant |
| `GET /api/models?author=Qwen&search=Qwen3.5` (full official line) | `Qwen3.5-{0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B}` + `-Base` variants. **No `-Instruct` anywhere.** |

**The Qwen3.5 naming convention differs from Qwen3.** In Qwen3, `Qwen3-4B-Instruct-2507` exists alongside a base. In Qwen3.5 the released conversational model carries **no suffix**, and the base carries `-Base`:

```
Qwen/Qwen3.5-4B          <- instruction-tuned / conversational  (cardData.license apache-2.0)
   base_model: ["Qwen/Qwen3.5-4B-Base"]
   tags: transformers, safetensors, qwen3_5, image-text-to-text, conversational
Qwen/Qwen3.5-4B-Base     <- the BASE model
```

The Hub API tag `base_model:finetune:Qwen/Qwen3.5-4B-Base` on `Qwen/Qwen3.5-4B` is the authoritative marker that `Qwen3.5-4B` is the fine-tuned (i.e. instruct) derivative.

**Consequence:** the requirement "the attacker MUST be the Instruct variant; do NOT substitute base `Qwen3.5-4B`" is satisfied by `Qwen/Qwen3.5-4B`. Requesting a literal `Qwen/Qwen3.5-4B-Instruct` would require downloading a nonexistent repo. **No config change is needed** — `configs/reproduction_three_model.yaml:69` already declares `Qwen/Qwen3.5-4B`.

### 1.2 A second, unreported property of the attacker model

`Qwen/Qwen3.5-4B` is **multimodal**, not a text-only LLM:

```
architectures                ['Qwen3_5ForConditionalGeneration']
model_type                   qwen3_5
pipeline_tag                 image-text-to-text
transformersInfo.auto_model  AutoModelForMultimodalLM
files                        preprocessor_config.json, video_preprocessor_config.json
```

The repo loads it through `transformers.pipeline("text-generation", ...)`. It *does* load and generate (measured, §2), but it carries a vision tower that the attack pipeline never uses, and it contributes to the 8.454 GB footprint. This was not mentioned in any prior report and is worth recording as a known property of the substitute model.

### 1.3 Verified inventory

| Role | Exact model id | Local snapshot | Architecture | dtype | Params | Weights on disk | Cached? |
|---|---|---|---|---|---|---|---|
| **attacker** | `Qwen/Qwen3.5-4B` | `…\models--Qwen--Qwen3.5-4B\snapshots\851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `Qwen3_5ForConditionalGeneration` | bf16 (BF16 4,659,861,248 + F32 3,840) | **4.66 B** | **9.343 GB** | **yes** |
| **target** | `microsoft/Phi-4-mini-instruct` | `…\models--microsoft--Phi-4-mini-instruct\snapshots\cfbefacb99257ffa30c83adab238a50856ac3083` | `Phi3ForCausalLM` | bf16 | **3.836 B** | **7.694 GB** | **yes** |
| **evaluator** | `Qwen/Qwen3-4B-Instruct-2507` | `…\models--Qwen--Qwen3-4B-Instruct-2507\snapshots\cdbee75f17c01a7cc42f958dc650907174af0554` | `Qwen3ForCausalLM` | bf16 | **4.023 B** | **8.061 GB** | **yes** |
| embedding | `sentence-transformers/all-mpnet-base-v2` | (Phase 1) | — | fp32 | — | — | yes |

All three are fully cached with a valid `refs/main` and **zero `.incomplete` files**. No download required.

Per-model detail:

| | attacker | target | evaluator |
|---|---|---|---|
| hidden_size | — (config.json omits; VLM) | 3072 | 2560 |
| num_hidden_layers | — | 32 | 36 |
| vocab_size | — | 200064 | 151936 |
| tie_word_embeddings | true | true | true |
| safetensors shards | 2 | 2 | 3 |
| tokenizer_config.json | present | present | present |
| chat template | present | present | present |
| generation_config.json | **absent** | present | present |

Two notes that matter operationally:
* The attacker has **no `generation_config.json`** — the pipeline must rely on the chat template and on explicitly-built `GenerationConfig`. `HFLocalChatLLM._build_gen_config` does build one explicitly, so this is handled, but it is the reason the thinking-channel suppression must come from the template.
* The attacker's config.json omits `torch_dtype`. The loader's non-local-path branch hardcodes `dtype=torch.bfloat16`; the local-path branch uses `torch_dtype="auto"`. Both resolve to bf16 in practice (measured footprint matches bf16 exactly).

---

## 2. Actual VRAM measurements

Method: `torch.cuda.mem_get_info()` free-VRAM delta at the driver level (not just PyTorch's allocator), synchronized. Full JSON: `results/phase14/residency_measurements.json`, `results/phase14/evict_mechanism_confirmation.json`.

**Safety:** the three-model stack was **never loaded simultaneously**. Each load was preceded by a free-VRAM gate; the third model was measured in isolation after a full purge. No OOM was induced.

### 2.1 Capacity

| Quantity | GB (decimal) | MiB |
|---|---:|---:|
| `nvidia-smi` total | 25.763 | 24570 |
| **Free at idle, desktop running** | **24.385** | 24193 |
| Consumed by display/compositor/desktop | 1.378 | — |

All figures below are decimal GB.

### 2.2 Incremental residency ramp

| Step | PyTorch `memory_allocated` | driver free | driver delta observed |
|---|---:|---:|---:|
| 0 · baseline (clean) | 0.000 | 24.385 | — |
| 1 · + **attacker** (`Qwen/Qwen3.5-4B`) | 8.412 | 15.931 | **−8.454** |
| 2 · attacker after one generation | 8.420 | 15.828 | −0.103 |
| 3 · + **target** (`Phi-4-mini-instruct`) | 16.092 | **8.209** | **−7.619** |
| 4 · target after one generation | 16.092 | 8.209 | 0.000 |
| 5 · **evaluator, loaded ALONE** (others purged) | 16.465 | 7.867 | **−8.045** |
| 6 · evaluator after one generation | 16.465 | 7.826 | −0.041 |
| 7 · final cleanup | 0.000 | 24.353 | fully released ✓ |

### 2.3 Measured weight footprints and their sum

| Model | Measured footprint |
|---|---:|
| `Qwen/Qwen3.5-4B` (attacker) | **8.454 GB** |
| `microsoft/Phi-4-mini-instruct` (target) | **7.619 GB** |
| `Qwen/Qwen3-4B-Instruct-2507` (evaluator) | **8.045 GB** |
| **Sum of all three** | **24.118 GB** |
| Free VRAM available | 24.385 GB |
| **Slack before any KV cache / activations** | **0.267 GB** |

This sum **independently reproduces** the crash report's `24.13 GB` cumulative figure to within 0.05 GB — two separate measurements, two different methods.

### 2.4 KV / activation requirement (derived from real runs)

| Quantity | Value | Source |
|---|---:|---|
| 2-model weights (attacker + target) | 16.073 GB | measured, §2.3 |
| Highest `peak_vram_gb` in any **real** Phase 14 run record | **18.753 GB** | `results/phase14/probe_json_compliance.jsonl` |
| KV + activations + workspace | **2.680 GB** | 18.753 − 16.073 |

Real records inspected (all 2-model, frozen config):

```
results/phase14/batch00_pre14_1.jsonl        18.325  18.301
results/phase14/probe_json_compliance.jsonl  18.325  18.753   <- worst observed
results/phase13/nbf_off.jsonl                15.726 (x9)
results/phase13/nbf_on.jsonl                 16.177 (x5), 8.505 (x4)
results/phase12/smoke_A_nbf_off.jsonl        15.726, 15.726
```

> **Correction to a prior figure.** `crash_investigation/PHASE14_CRASH_ROOT_CAUSE.md` §14.3 quotes a batch0 `peak_vram_gb` range of *"18.3-24.2"*. The **24.2 GB figure does not appear in any run record.** The worst measured peak in the frozen 2-model configuration is **18.753 GB**. The 22.33 GB and 27.23 GB values in that report came from standalone VRAM probes with artificially long contexts (`ctx 6013` / `ctx 4087`), not from the frozen pipeline. The probes are valid experiments, but they are **not** what the frozen study does, and the 24.2 GB number should not be used for capacity planning.

---

## 3. Is three-resident safe? — Part 3 verdict

**No. It is not "risky". It is arithmetically impossible.** Two independent estimates:

| Method | Predicted 3-model peak | Over/under 24.385 GB free |
|---|---:|---:|
| Measured weights (24.118) + measured real-run KV (2.680) | **26.798 GB** | **−2.413 GB** |
| Crash report's own probe measurements (27.23 GB @ ctx 4087) | 27.23 GB | −2.85 GB |
| Optimistic: weights alone, zero KV, zero activations | 24.118 GB | −0.267 GB |

Even the physically impossible zero-KV case leaves 0.267 GB — less than a single forward pass's workspace (measured 0.084 GB at ctx 1539, larger at longer contexts). **There is no context budget, precision setting, or batch tweak that makes three resident models fit.**

> Note the third row is a *lower bound that cannot be achieved*; it is listed only to show the margin is negative under every assumption.

---

## 4. Part 4 — architecture analysis

Throughout, "free" = `mem_get_info().free`, which already excludes the 1.378 GB the desktop consumes.

### OPTION A — all three resident

```
Qwen3.5-4B 8.454  +  Phi-4-mini 7.619  +  Qwen3-4B-Instruct 8.045  = 24.118 GB  >  usable
```

| | |
|---|---|
| VRAM required | 26.798 GB (measured + measured KV) vs 24.385 GB free |
| Startup | 3 cold loads: 23.53 + 18.67 + 17.76 = **59.96 s** (first) / 13.12 + 10.89 + 11.25 = **35.26 s** (warm) |
| Per-run overhead | **0 s** — no swaps |
| Stability | **Unrunnable.** The third load cannot be satisfied from VRAM; on WDDM the driver would satisfy the excess from host memory over PCIe — the exact condition present at the 2026-09-22 bugcheck. |
| **Verdict** | **INFEASIBLE — rejected on measurement, not on caution.** |

### OPTION B — attacker + target resident, evaluator swapped in

| | |
|---|---|
| Attacker + target resident | 16.073 GB → measured free **8.209 GB** (§2.2 step 3) |
| Free at the **worst real-run peak** (18.753 GB) | **7.010 GB** |
| Evaluator weights alone | **8.045 GB** |
| **Shortfall** | **1.035 GB at the worst peak**, before any evaluator KV |
| Per design (evict after use) | requires 2 loads × 17.6 evaluator calls = 35 loads/run |
| **Verdict** | **INFEASIBLE as written.** The evaluator does not fit alongside both. |

The brief's own instruction is correct and important here: *"Do not assume `del model; torch.cuda.empty_cache()` is sufficient."* It is not — §7 quantifies exactly why. But even a *correct* eviction does not rescue Option B, because the arithmetic fails before eviction is relevant.

### OPTION C — target + evaluator resident, attacker swapped

| | |
|---|---|
| Target + evaluator resident | 15.664 GB → free ≈ 8.721 GB |
| Attacker weights | **8.454 GB** — fits, with ~0.27 GB spare before KV |
| Frequency | attacker **15.9 calls/run** (crescendo, the most of any role) and the **slowest to load (23.53 s cold)** |
| **Verdict** | **Feasible only as a rotation, and the worst-performing one.** Pinning the evaluator forces the most-frequently-called and slowest-loading model through a load on every use. |

### OPTION D — 2-slot rotation, attacker pinned *(derived from measurement; not in the brief)*

Since 2 slots are available and 3 models are needed, a rotation is **unavoidable**. The only free choice is *which* model is the "rotating pair". Choosing the two **cheapest to load** minimises cost:

| Pairing | Rotating pair | Loads per turn | Cost/turn (warm) | Cost/turn (cold) |
|---|---|---:|---:|---:|
| **attacker pinned** | target ↔ evaluator | 2 | 10.89 + 11.25 = **22.15 s** | 18.67 + 17.76 = **36.43 s** |
| evaluator pinned | attacker ↔ target | 2 | 13.12 + 10.89 = 24.01 s | 23.53 + 18.67 = 42.20 s |
| target pinned | attacker ↔ evaluator | 2 | 13.12 + 11.25 = 24.37 s | 23.53 + 17.76 = 41.29 s |

**Cheapest = pin the attacker (largest weights, longest load), rotate target ↔ evaluator.** It is also the safest pairing by VRAM, because `attacker + evaluator = 16.499 GB` is the largest resident pair and still leaves 7.886 GB free.

| | |
|---|---|
| VRAM (worst resident pair) | 16.499 GB weights + ~2.7 GB KV ≈ **19.2 GB** → **~5.2 GB headroom** |
| Startup | 2 loads (attacker + evaluator) ≈ 35.3 s cold |
| Per-run overhead | **2 loads × turns**, see §9 |
| Stability | runnable; every load gated (§8) |
| **Verdict** | **FEASIBLE — recommended.** |

### OPTION E — evaluator on CPU

Measured, `results/phase14/cpu_evaluator_real.json`:

| | |
|---|---|
| Placement with `device="cpu"` | **cpu** (4,022,468,096 params), **GPU delta 0.000 GB** ✓ |
| Load | 2.95 s (from page cache) |
| Generation, ctx 643 | 4.65 s for 6 new tokens → **~1.3 tok/s** |
| Estimated per evaluator call (real shape: ctx ~900, 50–200 new tokens) | **~13–42 s** |
| Per run (16.4 calls, acronym) | **~490 s** on a 289 s base → **+170%** |
| Per run (17.6 calls, crescendo) | **~530 s** on a 511 s base → **+104%** |
| Semantics | identical weights, identical dtype → **identical numerics**, only slower |
| **Verdict** | **FEASIBLE and the safest option w.r.t. GPU risk, but ~2× the runtime cost of Option D.** Requires the `device_map="cpu"` fix (§6, §10-C6). |

### OPTION F — partial offload (accelerate `max_memory`)

Keep most evaluator layers on GPU, spill ~1.1 GB to CPU. Would close the 1.035 GB gap with a fraction of Option E's penalty. **Not measured** — listed for completeness only; would need its own experiment before adoption.

---

## 5. Part 5 — evaluator call frequency

Extracted from 37 existing records across Phase 12/13/14 (`results/phase14/*.jsonl`, `results/phase13/*.jsonl`, `results/phase12/*.jsonl`).

### Per-run mean call counts (completed runs only)

| Attack | runs | attacker | target | **evaluator** | turns | runtime (s) | eval/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| `acronym` | 9 | 11.3 | 8.4 | **16.4** | 3.7 | 289.4 | 16.4 |
| `crescendo_paper` | 9 | 15.9 | 11.0 | **17.6** | 6.6 | 510.7 | 17.6 |
| `opposite_day` | 9 | 9.4 | 7.7 | **13.9** | 6.2 | 238.0 | 13.9 |

**The evaluator is the most frequently invoked role in every attack** — more than the attacker. Maximum single-run evaluator count observed: **37**.

### Evaluator call breakdown by purpose

| Attack | `refusal_or_disclaimer` | `rubric` | total |
|---|---:|---:|---:|
| `acronym` | 12.78 (max 29) | 3.67 (max 8) | 16.45 |
| `crescendo_paper` | 11.00 (max 15) | 6.56 (max 8) | 17.56 |
| `opposite_day` | 7.67 (max 13) | 6.22 (max 8) | 13.89 |

`refusal_or_disclaimer` dominates, and it is **consumed inside the attack loop** — it drives backtracking (`results/phase14/batch00.log:104`: *"Refusal detected at round 2, backtracking (attempt 1)"*). This is decisive for architecture:

> **Evaluator calls cannot be deferred, batched, or hoisted out of the loop without changing attack behaviour** — which Part 7 forbids. Any architecture must serve an evaluator call *in place*.

### Swap-cost arithmetic

Using measured cold loads (attacker 23.53 s, target 18.67 s, evaluator 17.76 s) and warm loads (13.12 / 10.89 / 11.25 s):

```
Option D (attacker pinned, rotate target <-> evaluator):
    loads per turn   = 1 x target + 1 x evaluator
    cost per turn    = 22.15 s (warm)  ..  36.43 s (cold)
    runs per run     = turns
    acronym    3.7 turns ->   82 .. 135 s  added to 289 s  ->  +28% .. +47%
    crescendo  6.6 turns ->  146 .. 240 s  added to 511 s  ->  +29% .. +47%
    opposite   6.2 turns ->  137 .. 226 s  added to 238 s  ->  +58% .. +95%
```

> The 2-loads-per-turn figure assumes one target call and one evaluator call per turn, which matches the observed loop `attacker → target → evaluator → ...`. It is an **estimate**; the exact interleaving is not recoverable from the stored records, which carry counts but not call order. Labelled as an estimate in §9.

---

## 6. Part 6 — ModelManager / cache audit

### 6.1 Code audit (static)

`src/guardbound/llm/local_client.py`:

| Line | Code | Role |
|---|---|---|
| 55 | `_pipeline_cache: dict[str, Any] = {}` | **module-level dict holding strong references to every loaded pipeline** |
| 56 | `_cache_lock = threading.Lock()` | — |
| 201–208 | `_get_pipeline()` reuse branch | returns the cached pipeline |
| 255–256 | `_pipeline_cache[cache_key] = self._pipeline` | **stores, never removes** |
| 164–165 | `self._pipeline / _tokenizer = None` | constructor only |

`src/guardbound/llm/model_manager.py`:

| Line | Code | Problem |
|---|---|---|
| 14–16 | docstring: *"unloading evicts it from cache AND from GPU/`CPU` memory"* | **not implemented** |
| 156–160 | comment: *"KEEP the pipeline cache entry… Models loaded with `device_map` "cuda" stay resident on GPU; only the Python handle is cleared"* | **accurate** — and it contradicts the module docstring |
| 161–166 | `for attr in ("_pipeline","_tokenizer","_model","_client"): setattr(backend, attr, None)` | nulls the *backend's* attributes only |
| 170–171 | `torch.cuda.empty_cache()` | returns only **unused** allocator blocks; live tensors untouched |

**A repo-wide search confirms: `model_manager.py` contains no reference to `_pipeline_cache` at all.** Nothing pops or clears it. The only working purge in the repository is `scripts/phase13_5_baseline_benchmark.py:105,154,171,212` — a sibling script already solved this problem correctly.

### 6.2 Empirical proof (`results/phase14/evict_mechanism_confirmation.json`)

| Step | PyTorch alloc | driver free | cache entries |
|---|---:|---:|---:|
| baseline | 0.000 | 24.385 | 0 |
| + attacker (cold 13.1 s) | 8.412 | 15.931 | 1 |
| + target (cold 10.9 s) | 16.084 | 8.237 | 2 |
| **(1) `ModelManager._evict('target')`** | **16.084** | **8.237** | **2** |
| **(2) re-activate "evicted" target** | 16.084 | 8.237 | 2 |
| **(3) `cache.clear()`, attacker handle kept** | **8.412** | 15.931 | 0 |
| (4) + attacker handle nulled | 8.412 | 15.931 | 0 |
| **(4b) + stray local handle released** | **0.000** | **24.353** | 0 |
| (5) genuine **RELOAD** of target | 7.672 | 16.678 | 1 |
| final cleanup | 0.000 | 24.353 | 0 |

**Four conclusions, all measured:**

1. **`_evict()` frees exactly 0.000 GB.** `free` went 8.237 → 8.237. Not "a little" — nothing.
2. **Re-activation after `_evict()` took 0.00 s** → the model **never left the GPU**. `evict()` followed by `activate()` is a no-op pair: it costs nothing and frees nothing. This is why the three-model configuration could never have worked through `ModelManager`, and why the config comment *"The ModelManager keeps only one physical model resident at a time"* is false.
3. **`cache.clear()` frees only models whose backend attribute was already nulled** (step 3 freed exactly the target's 7.672 GB, because `_evict` had nulled `target._pipeline`; the attacker — whose handle was still set — stayed).
4. **The weights are held by two independent references, and *both* must be released**: `_pipeline_cache[cache_key]` **and** the backend's `._pipeline`. Step (4b) shows the attacker released only once *both* were gone (including a stray local reference in my own probe, which is exactly the class of bug that makes this fragile in practice).
5. **A genuine eviction costs a full cold reload: 11.25 s (warm page cache) to 23.53 s (cold).**

> **The central trade-off.** `_evict()` as written lets a model be re-activated for free *because it never left*. Fixing it so VRAM is actually freed makes every reload cost 11–24 s. **You can have the free reload or the freed VRAM — not both.** Every architecture in §4 is an instance of choosing between those two.

### 6.3 Additional integration defects found (not previously reported)

These matter for any implementation and are recorded for Part 10:

* **`ManagedLocalChatLLM.generate` silently drops `structured_output_mode`** (`model_manager.py:266–283`). It accepts the parameter then calls `self._inner.generate(messages, temperature, max_turns_context, json_format)` — omitting it. Routing Phase 14 through `ModelManager` would **silently disable Phase 14.1 constrained JSON decoding** and change attack reliability.
* **`provider_factory.build_role_llm` also drops it** (`provider_factory.py:106–111`): it forwards `chat_template_kwargs` (only when truthy) but never `structured_output_mode`.
* **`ManagedLocalChatLLM` is not used by Phase 14 at all.** `scripts/phase14_full_reproduction.py::make_models` (313–360) constructs three bare `HFLocalChatLLM`s, returns `None` for the manager slot, and never registers or activates anything. The `ModelManager` path is live only in `run_reproduction.py:960–968`, `provider_factory.build_three_model_stack`, and `phase13_5_baseline_benchmark.py`.
* **`device_map="cpu"` is silently ignored** — see below.

### 6.4 `device_map="cpu"` does not work (new, and it invalidated one of my own measurements)

`results/phase14/cpu_placement_probe.json`:

| kwargs passed to `pipeline()` | resolved `pipeline.device` | actual placement | GPU delta |
|---|---|---|---:|
| `device_map="cpu"` | **`cuda:0`** | all 4,022,468,096 params on `cuda:0` | **+8.103 GB** |
| `device="cpu"` | `cpu` | all params on `cpu` | **0.000 GB** |
| `device_map="cpu", device="cpu"` | `cpu` | all params on `cpu` | **0.000 GB** |

`HFLocalChatLLM._get_pipeline` only ever passes `device_map=`. So `device_map="cuda"` happens to work, while `device_map="cpu"` silently puts the model on the GPU.

**This is a hazard in its own right** — anyone attempting CPU offload through the existing backend would get the opposite of what they asked for, with no error. It also means my *first* CPU measurement (`results/phase14/cpu_evaluator_measurement.json`, reporting 1.5 s generations) was **wrong**: those were GPU timings. The corrected numbers are in §4 Option E / `cpu_evaluator_real.json`. Recorded here rather than silently overwritten.

---

## 7. Part 7 — experimental-semantics preservation

No architecture in §4 requires changing any of the following, and none was changed during this investigation:

| Preserved | Status |
|---|---|
| attack prompts, attack algorithms | untouched — no file in `src/guardbound/attacks/` was modified |
| full history, backtracking | untouched |
| refusal retry limits (`OFFICIAL_MAX_REFUSAL_RETRIES`) | untouched |
| max rounds (`max_turns=8`) | untouched |
| NBF, NBF threshold (`eta=0.0`) | untouched |
| generation temperatures (atk 0.7 / tgt 0.7 / eval 0.0) | untouched |
| top_p (1.0), max_new_tokens (256) | untouched |
| evaluator semantics (rubric / refusal / disclaimer / step judge) | untouched |
| target behaviour | untouched |
| constrained JSON behaviour | **at risk** — see §6.3; `ModelManager` and `provider_factory` currently drop `structured_output_mode`. Any implementation MUST carry it through, or Phase 14.1 is silently disabled. |

No history truncation, no reduced retries, no reduced rounds, no skipped evaluator calls.

---

## 8. Part 8 — VRAM safety guard

### Evidence chain

| Quantity | Value | Source |
|---|---:|---|
| Card total | 25.763 GB | `nvidia-smi` |
| Free at idle (desktop running) | **24.385 GB** | measured |
| Desktop/display consumption | 1.378 GB | derived |
| Largest resident pair (attacker + evaluator) | 16.499 GB | measured |
| Max KV/workspace in a **real stable run** | **2.680 GB** | 18.753 − 16.073, measured |
| **Lowest free VRAM in a completed, non-crashing run** | **7.010 GB** | 25.763 − 18.753, measured |
| Genuine reload cost | 11.25 – 23.53 s | measured |

### Recommended guard — derived, not chosen

All thresholds are expressed against `torch.cuda.mem_get_info().free`, which **already** accounts for the desktop, so no 1.378 GB term is needed.

| # | Check | Threshold | Derivation |
|---|---|---|---|
| G1 | **Pre-load gate** — before loading any model | `free_now − weights_of_incoming ≥ 3.0 GB` | 2.680 GB is the measured max KV/workspace in a stable run; rounded up to 3.0 GB. Below this, the load is REFUSED. |
| G2 | **Post-load verify** — immediately after a load | assert `free ≥ 1.5 GB`; else evict at once and abort the batch | catches a mis-estimated footprint before generation begins |
| G3 | **Per-run pre-flight** — before each run | abort batch if `free < 6.0 GB` | the lowest free seen in a completed stable run is 7.010 GB; 6.0 GB trips below anything the frozen study has been observed to survive |
| G4 | **Hard stop** — at every generation checkpoint | abort batch if `memory_reserved() > 22.0 GB` | 22.0 GB reserved leaves ~2.4 GB free, i.e. 3.25 GB above the worst observed legitimate run (18.753 GB) |

**G1 is the load that matters.** The 2026-09-22 failure was a *load/allocation* that could not be satisfied from VRAM; WDDM then satisfied it from host memory over PCIe, which is what stalled the Intel IOMMU and killed the hypervisor. A guard that only watches steady-state occupancy (G3/G4) acts too late; G1 prevents the over-commit from ever being attempted.

**Not recommended:** any guard expressed against `torch.cuda.memory_allocated()`. That counts only PyTorch's own allocations and is blind to the ≥1.378 GB the desktop and compositor hold, and to any second process. `mem_get_info()` is the correct primitive.

---

## 9. Part 9 — performance analysis

**MEASURED** values are from instrumented runs in this investigation. **ESTIMATED** values are arithmetic on measured inputs and are marked as such.

| Architecture | VRAM (resident) | Startup | Per-run overhead | Stability |
|---|---:|---:|---:|---|
| **A · all three resident** | **26.798 GB required** vs 24.385 GB free — **INFEASIBLE** | 59.96 s cold / 35.26 s warm *(measured)* | 0 s *(measured)* | **cannot run** — would spill to host memory over PCIe (the crash condition) |
| **B · atk+tgt resident, eval added** | free at worst real peak = **7.010 GB**; eval needs **8.045 GB** → **INFEASIBLE** | n/a | n/a | **cannot run** as specified |
| **C · tgt+eval resident, atk swapped** | 15.664 GB resident, ~3.8 GB headroom *(measured weights)* | 2 loads | **24.01 s/turn warm · 42.20 s/turn cold** *(estimated)* | runnable; pins the most-called and slowest-loading model |
| **D · attacker pinned, target↔eval rotated** | **16.499 GB** resident — ~5.2 GB headroom *(measured)* | 2 loads ≈ 35.3 s cold | **22.15 s/turn warm · 36.43 s/turn cold** *(estimated)* | **FEASIBLE — recommended** |
| **E · evaluator on CPU** | 16.073 GB GPU + 8.05 GB host RAM | 2.95 s CPU *(measured)* | **~490–530 s/run** *(estimated from measured 1.3 tok/s)* | feasible; **zero** GPU risk for the evaluator |
| **F · partial offload** | unmeasured | unmeasured | unmeasured | **not evaluated** |

### Projected per-run overhead, Option D

| Attack | turns | added (warm) | added (cold) | base runtime | overhead |
|---|---:|---:|---:|---:|---:|
| `acronym` | 3.7 | 82 s | 135 s | 289 s | **+28% … +47%** |
| `crescendo_paper` | 6.6 | 146 s | 240 s | 511 s | **+29% … +47%** |
| `opposite_day` | 6.2 | 137 s | 226 s | 238 s | **+58% … +95%** |

### Projected cost for the full frozen study

200 goals × 3 attacks × 2 conditions = **1200 runs**.

| | Value |
|---|---:|
| Mean base runtime (measured) | 346 s |
| Base total | 415,200 s ≈ **115 h** |
| Mean added by Option D (estimated, ~160 s/run) | ≈ **53 h** |
| **Option D total** | ≈ **168 h** |
| Option E total (estimated, ~510 s/run added) | ≈ **285 h** |

> **These are estimates and should be treated as such.** The two dominant uncertainties are (a) the number of role switches per turn, which cannot be recovered from stored records, and (b) page-cache warmth, which spans 11 s to 24 s per load — a factor of 2 on the added time. A 40-run instrumented pilot is the cheap way to replace both.

---

## 10. Part 11 §10 — exact code changes that would eventually be required

**None of these were implemented.** Each is stated so it can be reviewed before anything is touched.

---

**C1 — `_evict()` must actually release the weights**

```
File:               src/guardbound/llm/local_client.py
Function:           _get_pipeline (and a new release helper)
Current behavior:   _pipeline_cache[cache_key] = pipeline is written and never removed.
                    A repo-wide search shows nothing ever pops or clears it.
Required change:    Add a release path, e.g. release_pipeline(model_id, device_map)
                    that pops _pipeline_cache[cache_key] under _cache_lock, and is
                    the ONLY supported way to unload.
Experimental risk:  None to attack/NBF semantics — it changes only residency.
Performance impact: None on its own. It is the enabler for C2; without it no eviction
                    can free VRAM.
Validation:         Load a model, call release, assert mem_get_info().free returns to
                    baseline. Measured reference: purge delta = +7.682 GB for Phi-4-mini.
```

**C2 — `_evict()` must call the release path**

```
File:               src/guardbound/llm/model_manager.py
Function:           ModelManager._evict  (lines 133-172)
Current behavior:   Nulls the backend's _pipeline/_tokenizer/_model/_client attributes and
                    calls torch.cuda.empty_cache(). MEASURED to free 0.000 GB. The model
                    remains GPU-resident and re-activates in 0.00 s, so evict+activate is
                    a no-op pair.
Required change:   After nulling the attributes, also pop the corresponding
                    _pipeline_cache entry (via C1). Both releases are required:
                    cache-only leaves the backend holding it; backend-only leaves the
                    cache holding it. MEASURED: both are needed (steps 3 and 4b).
                    Optionally add a siblings check so a shared model id is not released
                    while another role still references it (preserve existing intent).
Experimental risk:  None to semantics. Behavioural change: roles no longer silently share
                    resident weights, so a run that previously "worked" only because
                    weights never left the GPU may now genuinely swap (and be slower).
Performance impact: Every subsequent re-activation costs a real reload: 11.25 s (warm)
                    to 23.53 s (cold), measured.
Validation:         Repeat this investigation's probe: _evict must reduce mem_get_info().free
                    by the model's measured footprint (7.619 GB for Phi-4-mini).
```

**C3 — `ManagedLocalChatLLM.generate` must forward `structured_output_mode`**

```
File:               src/guardbound/llm/model_manager.py
Function:           ManagedLocalChatLLM.generate  (lines 266-283)
Current behavior:   Accepts structured_output_mode, then calls
                    self._inner.generate(messages, temperature, max_turns_context, json_format)
                    WITHOUT it. Phase 14.1 constrained JSON decoding is silently disabled.
Required change:    Forward structured_output_mode to the inner backend.
Experimental risk:  HIGH if left unfixed — silently changes attacker/evaluator JSON
                    reliability, i.e. attack behaviour. Fixing it preserves current
                    Phase 14.1 behaviour.
Performance impact: None (constrained decoding is a logits processor).
Validation:         Run the existing Phase 14.1 structured-output test
                    (scripts/phase14_1_test_structured_output.py) through the manager and
                    confirm identical parse-failure counts.
```

**C4 — `build_role_llm` must forward `structured_output_mode` (and not drop falsy kwargs)**

```
File:               src/guardbound/llm/provider_factory.py
Function:           build_role_llm  (lines 94-115)
Current behavior:   Forwards chat_template_kwargs only when truthy (`extra = {...} if ctk else {}`)
                    and never forwards structured_output_mode.
Required change:    Forward structured_output_mode; treat an explicitly-empty
                    chat_template_kwargs as intentional rather than absent.
Experimental risk:  Same class as C3.
Performance impact: None.
Validation:         Assert a stack built via build_three_model_stack produces non-zero
                    constrained-JSON parse successes on the Phase 14.1 fixture.
```

**C5 — `make_models()` must use the three-model stack properly**

```
File:               scripts/phase14_full_reproduction.py
Function:           make_models  (lines 313-360)
Current behavior:   Builds three bare HFLocalChatLLM objects with device_map="cuda",
                    returns None for the manager slot, registers nothing, activates
                    nothing. Docstring states "We bypass the ModelManager eviction/swapping
                    entirely". Verified by log: results/phase14/batch00.log:7 shows
                    attacker == evaluator == Qwen/Qwen3-4B-Instruct-2507 (a TWO-model run),
                    consistent with the default CONFIG_PATH = configs/reproduction_phase12.yaml.
Required change:    Build attacker/target/evaluator through the provider factory with a
                    shared ModelManager (the pattern already used at
                    run_reproduction.py:960-968), supply the REAL three-model config, and
                    return the manager so per-run VRAM telemetry is live.
                    ALSO fix the docstring, which currently contradicts both the module
                    docstring and the measured behaviour.
Experimental risk:  HIGH — this is the change that switches Phase 14 from 2 distinct models
                    to 3, i.e. from a config that runs to one that cannot run resident and
                    must rotate. It changes runtime by ~+30-95% and changes the study's
                    declared model stack.
Performance impact: See §9.
Validation:         Assert the log line reports three DISTINCT model ids, and that
                    manager.summary()["roles"] has three distinct model_id values.
```

**C6 — CPU placement must use `device=`, not `device_map=`** *(only if Option E is chosen)*

```
File:               src/guardbound/llm/local_client.py
Function:           _get_pipeline  (lines 201-259)
Current behavior:   Passes device_map=self.device_map to pipeline(). MEASURED:
                    device_map="cpu" resolves to cuda:0 and puts all 4.02 B params on the
                    GPU (+8.103 GB) with no error; only device="cpu" yields real CPU
                    placement (0.000 GB).
Required change:    Pass device= for a pure-CPU role, or pass both, and assert afterwards
                    that the placement matches the request.
Experimental risk:  Low for the GPU path (device_map="cuda" already works). The risk is
                    leaving the trap in place: a silent 8 GB GPU allocation that the
                    operator believes is on CPU.
Performance impact: If Option E is chosen, ~490-530 s added per run (measured 1.3 tok/s).
Validation:         Sum parameter element counts by device and assert none land on cuda:0.
```

**C7 — new VRAM guard (G1–G4 from §8)**

```
File:               new module, e.g. src/guardbound/llm/vram_guard.py
Function:           n/a (new)
Current behavior:   No pre-flight gate, no ceiling, no mem_get_info use anywhere in
                    scripts/ or src/. Only peak_vram_gb REPORTING exists
                    (phase14_full_reproduction.py lines 772, 791).
Required change:    Implement G1 (pre-load gate, free - incoming >= 3.0 GB),
                    G2 (post-load verify, free >= 1.5 GB), G3 (pre-run, free >= 6.0 GB),
                    G4 (hard stop, memory_reserved() > 22.0 GB). Refuse and record; never
                    force a load.
Experimental risk:  A too-tight ceiling aborts legitimate runs. The thresholds above are
                    derived from measurement and leave >= 3.25 GB above the worst
                    observed legitimate peak.
Performance impact: Negligible (a few microseconds per check).
Validation:         Unit-test each gate with synthetic free-VRAM values, including the
                    boundary, and assert the pre-load gate REFUSES a third model in the
                    frozen three-model configuration.
```

---

## 11. Final recommendation

**Option D — pin the attacker, rotate target ↔ evaluator — is the recommended architecture.**

Rationale, all from measurement:

1. Three resident is impossible by ~1.04–2.41 GB, so a rotation is mandatory (§3).
2. Options B and C fail the arithmetic *before* eviction is even relevant (§4).
3. Option D minimises reload cost by pinning the model with the **largest weights (8.454 GB)** and the **longest load time (23.53 s cold)**.
4. It has the **largest safety margin** of any rotation: 5.2 GB headroom at its worst resident pair.
5. Its estimated cost, **+28% to +95% per run**, is roughly **half** of Option E's.

Option E (evaluator on CPU) is the right choice **only if GPU stability is prioritised over runtime** — it removes the evaluator from the GPU entirely, which is attractive given this machine's history. It costs about twice as much and requires the C6 fix.

**Blocking prerequisites, in order:**

| Order | Item | Why |
|---|---|---|
| 1 | **C1 + C2** (real eviction) | Without these, *no* rotation frees VRAM. Option D is unimplementable. |
| 2 | **C7** (VRAM guard) | The guard is what makes the rotation safe to run. |
| 3 | **C3 + C4** (forward `structured_output_mode`) | Otherwise Phase 14.1 constrained JSON is silently disabled and the attack behaviour changes. |
| 4 | **C5** (`make_models` + correct config) | This is the change that actually instantiates three models. Last, because it is the one that makes the run dangerous. |
| 5 | **C6** | Only if Option E is selected. |

**Before any of it:** resolve the two open questions that change the plan materially —

* **Q1.** Is `Qwen/Qwen3.5-4B` acceptable as the attacker? It is the instruct/conversational model, and `Qwen/Qwen3.5-4B-Instruct` does not exist (§1.1). If a *different* model was intended, that changes every number in this report.
* **Q2.** Should the evaluator be a **separate physical model at all**? The frozen `results/phase14_manifest.json` and `configs/reproduction_phase12.yaml` both set attacker == evaluator, and `attacks/base.py:41` records that *"Officially the attacker model doubles as the evaluator."* Three distinct models is therefore a deliberate **deviation** from the paper's own semantics as captured in this repo. It is a legitimate choice — but it should be a recorded one, because it is what forces the entire rotation architecture and its ~53 h cost.

---

## 12. Evidence and provenance

### Artifacts produced (all read-only w.r.t. the repo's production code)

| File | Contents |
|---|---|
| `results/phase14/residency_measurements.json` | VRAM ramp, per-model footprints, load times, feasibility arithmetic |
| `results/phase14/evict_mechanism_confirmation.json` | The four-step proof that `_evict()` frees nothing and why |
| `results/phase14/cpu_evaluator_verification.json` | Proof that `device_map="cpu"` lands on `cuda:0` |
| `results/phase14/cpu_placement_probe.json` | Isolation: `device_map="cpu"` vs `device="cpu"` vs both |
| `results/phase14/cpu_evaluator_real.json` | Corrected CPU timings (1.3 tok/s, 0.000 GB GPU) |
| `results/phase14/cpu_evaluator_measurement.json` | **SUPERSEDED — wrong.** GPU timings mislabelled as CPU. Retained for traceability; see §6.4. |

Scratch scripts (session scratchpad, not in the repo): `p1_model_verify.py`, `p5_call_frequency.py`, `p234_residency.py`, `p6_confirm.py`, `p_cpu_eval.py`, `p_cpu_verify.py`, `p_cpu_isolate.py`, `p_cpu_real.py`.

### What was NOT done

* No production code modified — `src/guardbound/llm/model_manager.py`, `local_client.py`, `provider_factory.py` and `scripts/phase14_full_reproduction.py` are byte-identical to before.
* No change to `ModelManager`, cache behaviour, eviction, swapping, generation, attacks, or prompts.
* **Phase 14 was not launched. Batch 0 was not launched.**
* A third model was never loaded alongside the other two. No OOM was induced.
* GPU left clean: `nvidia-smi` reports **0 MiB used, 24193 MiB free** after every run.

### Caveats stated honestly

1. **The rotation overhead (§9) is an estimate.** The per-turn switch count is an assumption (2 loads/turn, from the observed `attacker → target → evaluator` loop) because stored records carry call *counts*, not call *order*. Load time also spans 11–24 s with page-cache warmth. A 40-run pilot would replace both uncertainties with measurements.
2. **Option F (partial offload) is unevaluated.** It may dominate Option D on cost; it is not measured and must not be assumed.
3. **CPU throughput was measured at ctx 643 with 6 generated tokens** (the model answered briefly). The per-call estimate for real evaluator calls (ctx ~900, 50–200 new tokens) is extrapolated from that single measurement on a `refusals`-style prompt; a 10-call sample on real evaluator prompts would firm it up.
4. **The 22.33 GB / 27.23 GB probe figures** in `PHASE14_CRASH_ROOT_CAUSE.md` §7 come from standalone probes with artificial contexts (`ctx 6013` / `ctx 4087`), not the frozen pipeline. They are useful as upper bounds but should not be quoted as the study's behaviour (§2.4).

---

## STOP

Per Part 10 and the final gate: this is an **architecture investigation only**. No implementation has been started, no architecture selected for implementation, and Phase 14 / Batch 0 remain unlaunched. Awaiting explicit approval before any change in §10 is made.
