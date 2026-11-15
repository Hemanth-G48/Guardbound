# PHASE 17 — Attacker Model Qualification: Stage 0 Report

**Status:** **STAGE 0 COMPLETE — BEHAVIOURAL QUALIFICATION NOT STARTED**
**Contract:** TARGET `Llama 3 8B Instruct` (FIXED) · JUDGE `Qwen/Qwen3-8B` (FIXED, re-resolved) · ATTACKER = 5 candidates (VARIABLE)
**Closing requirement:** `PHASE 17 INCOMPLETE — STAGE 0 COMPLETE — NO CANDIDATE FEASIBLE AT REFERENCE PRECISION — 90-RUN PILOT NOT LAUNCHED`

This is an **interim** report. Stage A (structured output), Stage B (multi-turn), GPU qualification, and judge pre-qualification have not run. It is issued now because Stage 0 reached a decisive stopping conclusion and because three contract-level defects were found that had to be resolved before any behavioural work could begin.

---

## 1. Executive summary

Five attacker candidates were specified. **None of them can run on this machine at reference precision.** This was determined from authoritative HuggingFace metadata and measured machine budgets, without downloading a single weight.

The blockers are independent and compound: the 27B candidates are *dense* (not MoE), so a 27B model is ~52 GiB in BF16 regardless of marketing; the only official quantized checkpoints are FP8 at ~28.8 GiB, which still exceeds a 24 GiB card; and the community 4-bit ecosystem is dominated by two formats that **cannot execute on this hardware at all** — MLX (Apple-silicon only) and NVFP4 (Blackwell-class, while this is an Ada sm_89 card).

Three contract defects were also found and resolved:

1. The fixed judge id `Qwen/Qwen3-8B-Instruct-2507` **does not exist** (HTTP 401).
2. Two of the five attacker ids do not resolve as written; the exact ids differ.
3. The candidate `prism-ml/bonsai-27b` is ambiguous between two model generations and has no CUDA runtime path.

The phase can proceed only via **4-bit post-training quantization at load time** on the official weights, which was verified working end-to-end on this machine. Under §22 that caps every runnable candidate at **CONDITIONALLY QUALIFIED**, by construction, before a single token is generated.

No ASR was computed. The 90-run pilot and the 1,200-run study were not launched. No frozen artifact was modified.

---

## 2. Machine budgets (measured, not assumed)

| Budget | Value | Source |
|---|---|---|
| GPU | NVIDIA RTX 4500 Ada Generation | `nvidia-smi` |
| VRAM total | 24570 MiB = **23.99 GiB** | `nvidia-smi` |
| Compute capability | **sm_89 (Ada)** | `torch.cuda.get_device_capability(0)` |
| Driver | 595.79 | `nvidia-smi` |
| Runtime allowance applied | 3.0 GiB (KV cache + activations + CUDA context) | stated assumption; §11 replaces it with a measured peak |
| Disk free, before | 28.86 GiB | `shutil.disk_usage` |
| Disk free, after eviction | **147.96 GiB** | measured |
| HF cache, before | 193.48 GiB | measured |

Environment: Python 3.13.9 · transformers 5.16.1 · torch 2.7.1+**cu118** · accelerate 1.14.0 · bitsandbytes 0.50.2 (installed this phase).

---

## 3. Stage 0 — feasibility results

### 3.1 Candidate verdicts

| Role | Model | Reference precision | Footprint | Verdict |
|---|---|---:|---:|---|
| A | `prism-ml` Bonsai 27B | 1-bit/ternary (native) | n/a | **BLOCKED** — MLX only, no CUDA runtime |
| B | `Qwen/Qwen3.8-27B` | BF16 | 51.75 GiB | **INFEASIBLE** (VRAM −7.7 GiB, disk) |
| B′ | `Qwen/Qwen3.8-27B-FP8` | FP8 | 28.74 GiB | **INFEASIBLE** (VRAM −7.8 GiB) |
| C | `Qwen/Qwen3.6-27B` | BF16 | 51.75 GiB | **INFEASIBLE** (VRAM −7.8 GiB, disk) |
| C′ | `Qwen/Qwen3.6-27B-FP8` | FP8 | 28.75 GiB | **INFEASIBLE** (VRAM −7.8 GiB) |
| D | `mistralai/Ministral-3-14B-Reasoning-2512` | BF16 | 25.97 GiB | **INFEASIBLE** (VRAM −5.0 GiB) |
| E | `mistralai/Devstral-Small-2-24B-Instruct-2512` | FP8 (native) | 24.02 GiB | **INFEASIBLE** (VRAM −3.0 GiB) |
| TARGET | `meta-llama/Meta-Llama-3-8B-Instruct` | BF16 | 14.96 GiB | **FEASIBLE** (already cached) |

Zero of five attacker candidates fit. The target is the only role that runs as specified.

### 3.2 Official quantized availability — exhausted

Enumerating the publisher orgs directly (rather than guessing at naming conventions):

| Base model | Official repos | Size |
|---|---|---|
| Qwen3.8-27B | BF16, **FP8** | 51.77 / 28.77 GiB |
| Qwen3.6-27B | BF16, **FP8** | 51.77 / 28.77 GiB |
| Ministral-3-14B-Reasoning | BF16, GGUF | 51.98 / 55.99 GiB |
| Devstral-Small-2-24B | FP8 only | 48.08 GiB |

**No official 4-bit checkpoint exists for any candidate.** `-AWQ` and `-GPTQ-Int4` were probed explicitly and both return 401. The only official quantization is FP8, and FP8 does not fit.

### 3.3 Community 4-bit — mostly unusable on this hardware

A sweep of 4-bit-shaped repos found 49 variants for Qwen3.8-27B and 37 for Qwen3.6-27B. The majority cannot execute here:

- **MLX** (e.g. `lmstudio-community/Qwen3.8-27B-MLX-4bit`, `mlx-community/*`) — Apple-silicon array framework, **no CUDA backend**. Includes the specific repo queried during this phase.
- **NVFP4** — Blackwell-class FP4; this card is Ada **sm_89**. Unusable.
- **compressed-tensors / AutoRound** — vLLM-oriented, and **vLLM has no Windows support**, so these are unreachable on this host regardless of size.
- **GGUF** — llama.cpp runtime, not the project's transformers backend.

Genuinely Ada-loadable 4-bit exists only as AWQ or GPTQ (17.3–20.4 GiB), each needing a runtime that is not installed, on Windows + cu118 where `autoawq` support is generally absent.

---

## 4. Verification performed

### 4.1 NF4 runtime — verified by experiment, not by install

A successful `pip install` proves nothing, so the runtime was exercised end-to-end on the smallest cached model.

| Measurement | Result |
|---|---|
| `Linear4bit` modules | **252** |
| Plain `Linear` modules | 253 (vision tower / embeddings / LM head, deliberately unquantized by bnb) |
| Parameters in 4-bit layers | **82.4%** |
| VRAM allocated, NF4 | **2.495 GiB** |
| VRAM allocated, same model BF16 (control) | **7.501 GiB** |
| Reduction factor | **3.01×** |
| Generation | correct (`'READY'`) |
| Load time | 7.23 s |

The BF16 control is the decisive evidence: without it, an unquantized load would have produced a superficially similar-looking success.

**One correction worth recording.** An earlier version of this check inspected `next(model.parameters())` and reported `quantized=False` — a false negative, because bitsandbytes intentionally leaves embeddings in the compute dtype. The check was replaced with a module census. Had it stood, it would have misreported successful quantization as a failure.

### 4.2 Section 4/5 role verification

**Target — `meta-llama/Meta-Llama-3-8B-Instruct`**

| Field | Value |
|---|---|
| Revision | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| Architecture | `LlamaForCausalLM`, 32 layers, hidden 4096, vocab 128256 |
| Context length | 8192 |
| dtype / quantization | bfloat16 / none |
| EOS | `[128001, 128009]`; `pad_token` is **null** on disk |
| `generation_config.json` | temperature 0.6, top_p 0.9, do_sample true, max_length 4096 |
| Contract generation | temperature 0.7, top_p 1.0 (explicit) |
| `max_new_tokens` | **NOT SPECIFIED** — see §6 |

**Judge — `Qwen/Qwen3-8B`**

| Field | Value |
|---|---|
| Revision | `b968826d9c46dd6066d109eabc6255188de91218` |
| Architecture | `Qwen3ForCausalLM`, 36 layers, hidden 4096, vocab 151936 |
| Context length | 40960 |
| dtype / quantization | bfloat16 / none |
| `generation_config.json` | temperature 0.6, top_p 0.95, **top_k 20**, do_sample true |
| Chat template | contains `enable_thinking` → **hybrid thinking model** |

Config hashes (sha256 over canonical JSON of each role block):

```
target_config_hash : ec657141be3ce3b3977b840c6baf943f3474cd7407613f5d942f904b7520c679
judge_config_hash  : f2433e3b2a25bde9e98a1736503015a31b82134d472a50a42018f8e4ea352357
```

Both must remain identical across every attacker candidate (§4).

---

## 5. Frozen-stack integrity

All four frozen constants re-verified unchanged, and every historical results directory confirmed intact.

| Artifact | SHA-256 | Match |
|---|---|---|
| Track A, 180 records | `FCB952F1598B697B033D605B0791CFBB64A010F8E43CD1ADF3D0EDC359D8A092` | ✅ |
| Track A, 108 snapshot | `C12A562A9454D83B6CC0EE93B2653F4AA5AB058C5AC3E50CEED4EF7A8EC57FB3` | ✅ |
| Frozen config | `27B660A7052054521A94E3430013D9863964E825157056D3861228250F19A358` | ✅ |
| NBF checkpoint | `CEA1A75BCEF4FC515814B69C42541C95114F587ABCC4505C9B096BBFA2A136FE` | ✅ (both copies byte-identical) |

Historical directories untouched: `results/phase14` (27 files), `results/phase15_pilot_30` (4), `results/phase16_5_forensics` (11), `results/phase16_6_phi4_target` (8).

Git: `b46d6fd7ed37b7a6ee9e643476ae2b2263b85624` on `main`, working tree dirty (8 modified tracked files, recorded in the manifest rather than cleaned).

---

## 6. Contract defects found and resolved

| # | Defect | Resolution |
|---|---|---|
| 1 | Judge id `Qwen/Qwen3-8B-Instruct-2507` **returns 401 — does not exist**. The 2507 refresh covered 4B / 30B-A3B / 235B-A22B, not 8B. | Re-resolved to `Qwen/Qwen3-8B` (**user-approved**), downloaded and verified. |
| 2 | `mistralai/Ministral-3-14B-Reasoning` → 404. Exact id is `…-Reasoning-**2512**`. | Exact id approved by the user. |
| 3 | `mistralai/Devstral-Small-2-2512` → wrong. Exact id is `mistralai/Devstral-Small-2-**24B-Instruct**-2512`. | Exact id approved by the user. |
| 4 | `prism-ml/bonsai-27b` is ambiguous between two generations (Bonsai 27B on Qwen3.6; Ternary Bonsai 2 27B on Qwen3.8) and ships MLX + a custom llama.cpp fork only. vLLM support exists solely via a third-party reverse-engineered `PTQ1_0` kernel. | **BLOCKED** (**user-approved**), on both ambiguity and §10's prohibition on ad-hoc conversions. |

**Note on defect 1:** the assistant offered the non-existent id as an option in an earlier question. The error was caught by an existence check before any download or trust was placed in it.

**Documented contract break.** Replacing the frozen target (Phi-4-mini) and judge (Llama-3.2-3B) means any later ASR is **not comparable** to the 180 Track A records. This is deliberate: the previous target never self-terminated (95.0% cap-hit, 1/8 natural stops), which made attack success unmeasurable. It is recorded as a break, not discovered later.

**`max_new_tokens` is deliberately NOT SPECIFIED for the target.** Phase 16.5 established that a fixed 256-token budget truncated 95.0% of responses with 0% natural EOS. Re-introducing a ceiling silently would repeat that defect; §16 requires an inherited value to be declared, so none is set. **This is an open decision for the user.**

---

## 7. Actions taken on shared state

**Cache eviction — user-approved, recorded before execution.** 9 entries, **135.76 GiB** freed:

| Entry | GiB | Reason |
|---|---:|---|
| `unsloth/gemma-3-12b-it` | 22.74 | unused by any config |
| `google/gemma-4-12B-it` | 22.31 | unused |
| `meta-llama/Llama-3.1-8B-Instruct` | 20.78 | not in the Phase 17 contract |
| `zai-org/GLM-4.6V-Flash` | 19.19 | Phase 15: NOT QUALIFIED |
| `ornith-ai/Ornith-1.5-9B` | 18.00 | Phase 16.3: pilot NOT AUTHORIZED |
| `google/gemma-4-12B-it-qat-w4a16-ct` | 9.59 | unused |
| `Qwen/Qwen3.5-4B` | 8.70 | superseded attacker |
| `google/gemma-3-4b-it` | 8.05 | unused |
| `microsoft/phi-4` | 6.40 | Phase 16.6: not feasible |

**Retained deliberately:** `Meta-Llama-3-8B-Instruct` (target), `Llama-3.2-3B-Instruct` (frozen Track A judge), `Qwen3-4B-Instruct-2507` (frozen incumbent attacker), `Phi-4-mini-instruct` (frozen Track A target), `all-mpnet-base-v2` (NBF embedder), `hanjianghu/NBF-LLM` (author reference).

All evicted entries are public, re-downloadable weights that no frozen configuration references.

**Runtime dependency added:** `bitsandbytes` 0.50.2 — required by the approved quantization path, and recorded as a deviation.

---

## 8. Section 28 questions — status

| # | Question | Status |
|---|---|---|
| 1 | Can each candidate run on the RTX 4500? | **ANSWERED** — none at reference precision |
| 2 | Which candidates require quantization? | **ANSWERED** — all four non-BLOCKED candidates |
| 3 | What exact quantization was used? | **PARTIAL** — NF4 verified as the method; not yet applied per candidate |
| 4 | Does each candidate reliably produce attacker JSON? | **PENDING** — Stage A |
| 5 | Does each candidate survive multi-turn generation? | **PENDING** — Stage B |
| 6 | Are reasoning models compatible with the frozen interface? | **PENDING** — risk identified, not measured |
| 7 | Are model switching and VRAM residency safe? | **PENDING** — §11 |
| 8 | Is Qwen3-8B a reliable fixed judge? | **PENDING** — §6 |
| 9 | Qualification decisions per candidate | **PARTIAL** — A = BLOCKED; B–E capped at CONDITIONALLY QUALIFIED |
| 10 | Exact attacker configuration to freeze | **PENDING** |

**No ASR conclusion is provided.** Per §2, this phase is not an ASR experiment and no attack success was measured.

---

## 9. Evidence index

| Artifact | Path |
|---|---|
| Baseline manifest | `results/phase17_model_qualification/baseline_manifest.json` |
| Stage 0 feasibility | `results/phase17_model_qualification/feasibility_results.json` |
| Cache eviction record | `results/phase17_model_qualification/cache_eviction.json` |
| NF4 runtime verification | `results/phase17_model_qualification/quant_runtime_verification.json` |
| Target/judge verification | `results/phase17_model_qualification/target_verification.json` |
| Feasibility script | `scripts/phase17_stage0_feasibility.py` |
| Cache eviction script | `scripts/phase17_evict_cache.py` |
| Runtime verification script | `scripts/phase17_verify_quant_runtime.py` |
| Role verification script | `scripts/phase17_verify_roles.py` |

---

## 10. Caveats and limitations

- **Every candidate will be CONDITIONALLY QUALIFIED at best.** All four runnable candidates require quantization, which §22 classifies as a documented limitation. This is unavoidable on this hardware, not a judgement about any model's capability.
- **A 4-bit attacker is an additional capability deviation** stacked on top of the existing local-for-GPT-4o substitution.
- **The 3.0 GiB runtime allowance is an assumption**, not a measurement. §11's load tests replace it with observed peaks. Any verdict within ~3 GiB of the boundary (notably Devstral, which misses by exactly 3.0 GiB) could shift once measured.
- **Blocking-reason arithmetic is exact but the margins are thin** for Devstral and Ministral.
- **`usedStorage` is not a download size.** An early version of the feasibility script used it and reported Llama-3-8B as 57.88 GiB (it sums all revisions and format copies). The script now sums main-revision file sizes only; the earlier figures were discarded.
- **Console capture is unreliable in this environment** (a documented cluster quirk); every verdict above is read from a written JSON artifact, not from stdout.
- **Download throughput is ~4 MB/s**, so the four remaining candidate downloads are many hours of transfer.

---

## 11. Open decisions for the user

1. **Target `max_new_tokens`** — currently deliberately unset. Must be decided explicitly before any Stage B run.
2. **Quantization path** — runtime NF4 on official BF16 weights (recommended, one method for all candidates) versus community AWQ/GPTQ (smaller downloads, three runtimes, Windows/cu118 risk).
3. **Candidate subset** — whether to qualify all four runnable candidates or a cheaper subset first, given ~4 MB/s transfer.
4. **Judge thinking mode** — whether to disable `enable_thinking` for the judge, or measure reasoning-leakage as a §6 failure mode first.

---

## 12. What was not done

- No ASR computed. No NBF effect measured. No attack-success ranking produced.
- The 90-run pilot was **not** launched.
- The 1,200-run experiment was **not** launched.
- No frozen artifact was modified; no prompt, parser, threshold, or generation default was edited.
- Stage A, Stage B, §6 judge pre-qualification, §11 GPU qualification, §19/§20 model-switching validation were **not** run.

---

```
PHASE 17 INCOMPLETE — STAGE 0 COMPLETE — NO CANDIDATE FEASIBLE AT REFERENCE PRECISION — 90-RUN PILOT NOT LAUNCHED
```
