# PHASE 17 — STAGE 3: Qwen3.8-27B Native Architecture Qualification

**Status:** **STAGE 3 COMPLETE — NATIVE ARCHITECTURE LOADED AND EXECUTED — SEE §15 FOR THE TWO ANSWERS**
**Scope:** `Qwen/Qwen3.8-27B` only. Target `meta-llama/Meta-Llama-3-8B-Instruct` and judge `Qwen/Qwen3-8B` unchanged.

No ASR was computed. No 90-run pilot, no 1,200-run reproduction and no NBF ON/OFF comparison was launched. Nothing here measures attack effectiveness.

---

## 1. Executive summary

The question this stage exists to answer — *can the declared `Qwen3_5ForConditionalGeneration` architecture be loaded and executed on this machine under the approved 4-bit NF4 runtime?* — has a measured answer: **yes.**

| Step | Result |
|---|---|
| Declared architecture | `Qwen3_5ForConditionalGeneration` (from the checkpoint's own `config.json`) |
| Native auto-class | `AutoModelForImageTextToText` → **`Qwen3_5ForConditionalGeneration`** |
| Native processor | **`Qwen3VLProcessor`** (the class the checkpoint declares) |
| NF4 load | **PASS** — 98.9 s, **16.688 GiB** resident, 606 `Linear4bit`, 82.95 % of parameters quantized |
| Weight consumption | **1184 of 1199** checkpoint tensors consumed, **0 missing**, 15 unused = the auxiliary `mtp.*` head |
| Text-only generation | **PASS** — `'READY'`, 2 tokens, natural EOS, no images supplied |
| JSON contract | **22/24 = 0.9167** (21 direct + 1 via the frozen extraction), 2 prose failures |
| Multi-turn | **executes** — 12/15 attacker replies valid, target EOS 12/12, evaluator 14/14; all 3 runs ended on a prose reply |
| Residency | **PASS** — 3 cycles × 3 roles, max resident 1, zero errors, 0.0 GiB after release |
| **Final native status** | **NATIVE CONDITIONALLY QUALIFIED** (§14) |

The structural point that Stage 2 could not establish, and that this stage does, is in §5: the native class expects **exactly** the parameter names the checkpoint publishes (`model.language_model.*`, `model.visual.*`), while `Qwen3_5ForCausalLM` expects `model.layers.*` and has no vision tower at all. The Stage 2 path was therefore a genuine substitution, not a naming detail — and it is now avoidable.

Nothing in this stage changes an attack, a prompt, the rubric, the parser or the NBF. The code changes are additive, opt-in and listed in `stage3_code_changes.md`.

---

## 2. What the repository and transformers already provide (Part 6)

Before writing anything, the existing implementations were inspected:

- `multimodal_client.MultimodalChatLLM` already implements the native pipeline shape — `processor.apply_chat_template(..., tokenize=True, return_dict=True)` → `<native class>.generate(...)` → `processor.batch_decode(...)` — including the **frozen** JSON stopping criterion and structural extraction, imported from `local_client` rather than re-implemented.
- `ornith15_client.Ornith15ChatLLM` is an existing native-path adapter for a *`Qwen3_5ForConditionalGeneration`* checkpoint (`ornith-ai/Ornith-1.5-9B`), with an architecture assertion that refuses a class the checkpoint does not declare.
- `transformers 5.16.1` maps `qwen3_5` natively: `AutoModelForImageTextToText` and `AutoModelForMultimodalLM` both resolve `Qwen3_5ForConditionalGeneration`.

**Decision taken from that inspection:** reuse `MultimodalChatLLM` (the whole pipeline) and add a thin `Qwen38NativeChatLLM` for the loader only. Not a new pipeline, not a per-model message format, and not a clone of the Ornith module — a Qwen backend whose error type and label name another model would be worse than ten lines of loader code.

## 3. Architecture audit (Part 7)

Read from the local snapshot's own files, not from the model name (`architecture_audit.json`).

| Field | Value |
|---|---|
| model_id / revision | `Qwen/Qwen3.8-27B` / `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| `architectures` | `["Qwen3_5ForConditionalGeneration"]` |
| `model_type` | `qwen3_5` (wrapper; `language_model_only: false`) |
| Language sub-config | `qwen3_5_text`, hidden 5120, **64 layers**, 24 heads / 4 KV heads, intermediate 17408, vocab 248320, `max_position_embeddings` **262144**, layer types 48 × `linear_attention` + 16 × `full_attention`, 1 MTP layer |
| Vision sub-config | depth 27, hidden 1152, patch 16, spatial merge 2, `out_hidden_size` 5120 |
| Processor (declared) | `Qwen3VLProcessor` (`image_processor_type: Qwen2VLImageProcessorFast`, video: `Qwen3VLVideoProcessor`) |
| Generation config | `do_sample: true`, **`eos_token_id: [248046, 248044]`** |
| Checkpoint tensors | 1199, 55,562,855,904 bytes — roots `lm_head` (1), `model` (1183), `mtp` (15); `model.visual.*` **333** tensors confirm a populated vision tower |

## 4. Processor and text-only interaction (Part 9)

| Check | Result |
|---|---|
| `AutoProcessor.from_pretrained(snapshot)` | loaded **`Qwen3VLProcessor`** — the class the checkpoint declares (`matches_declared: true`) |
| Tokenizer | present via the processor; chat template present |
| EOS / PAD | `eos_token_id 248046`, pad token present |
| Text-only generation | the attacker experiment sends text-only messages, and §11 shows the model generating from exactly that — **no image, placeholder or image token is fabricated** |

The processor is a multimodal processor by class, but nothing in the attacker path requires an image: the chat template renders the text messages and the model generates from `input_ids` alone. The alternative — inventing a placeholder image — was not needed and was not done.

## 5. Native loader discovery and the structural proof (Part 8)

Three auto-classes were asked to resolve the checkpoint's own config, and each was instantiated from a *tiny* config of the same class on the meta device to enumerate the parameter names it expects (`native_loader_probe.json`):

| Auto-class | Resolves to | Expects vision tower | Text prefix it wants |
|---|---|---|---|
| `AutoModelForImageTextToText` | **`Qwen3_5ForConditionalGeneration`** | **yes** | `model.language_model.layers.*` |
| `AutoModelForMultimodalLM` | **`Qwen3_5ForConditionalGeneration`** | **yes** | `model.language_model.layers.*` |
| `AutoModelForCausalLM` | `Qwen3_5ForCausalLM` | **no** | `model.layers.*` |

Checkpoint publishes: `lm_head.weight`, `model.language_model.{embed_tokens,layers,norm}`, `model.visual.{blocks,merger,patch_embed,pos_embed}`, `mtp.*`.

So the native class's expected structure **is** the checkpoint's structure, and the CausalLM class's is not. This is the distinction Stage 2 flagged as unresolved and could not test: there, a raw key comparison reported 1 of 3331 matched, which was *evidence of a mismatch* that the library's own "no missing keys" report obscured by remapping the names. Both statements are now reconciled — transformers remaps, so the weights are consumed, but the class is still not the declared architecture.

`AutoModelForImageTextToText` was chosen: it is the auto-class for image-text-to-text checkpoints (the checkpoint's `pipeline_tag` and its declared `Qwen3VLProcessor`), and it resolves the declared class directly. `AutoModelForMultimodalLM` resolves identically in this version and remains the project's existing seam for the Ornith backend.

## 6. NF4 implementation and census (Parts 10/11)

The approved configuration, unchanged from Stage 2 and applied to the native path:

```text
official Qwen/Qwen3.8-27B weights
        ↓  BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=bfloat16, bnb_4bit_use_double_quant=True)
4-bit NF4 PTQ at load time (bitsandbytes, BF16 compute)
```

| Census | CausalLM baseline (Stage 2) | Native (Stage 3) | Difference |
|---|---|---|---|
| `Linear4bit` modules | 496 | **606** | +110 — the **vision tower's** linears |
| plain `Linear` modules | 1 | 1 | the LM head, deliberately unquantized by bitsandbytes |
| total parameters | 14,720,720,384 | **14,953,501,936** | +232,781,552 — the vision tower |
| params in 4-bit layers | 12,175,278,080 | 12,403,508,080 | +228,230,000 |
| fraction quantized | 0.8271 | **0.8295** | +0.0024 |
| VRAM after load | 16.456 GiB | **16.688 GiB** | +0.23 GiB |
| load time | 100.44 s | 98.91 s | — |

The unquantized remainder is accounted for, not assumed: embeddings 1,274,052,608 parameters and the LM head 1,271,398,400 — the two modules bitsandbytes intentionally leaves in the compute dtype.

The difference is exactly what the architecture predicts: the native class carries the vision tower, the substituted text class does not. Nothing "disappeared" through quantization, and nothing extra was silently dropped.

## 7. Native load and weight consumption (Parts 12/13)

| Measurement | Value |
|---|---|
| Load | **ok**, 98.91 s |
| Requested class | `AutoModelForImageTextToText` |
| **Loaded class** | **`Qwen3_5ForConditionalGeneration`** — matches the checkpoint's declaration |
| `config.architectures` on the loaded model | `["Qwen3_5ForConditionalGeneration"]` |
| transformers load report | **no** missing, unused, unexpected or size-mismatch messages |
| VRAM after load | 16.688 GiB allocated, 16.877 GiB reserved, peak 16.728 GiB; 5.804 GiB driver-free |
| Parameter reconciliation | **1184 / 1199** checkpoint tensors consumed; **0 missing**; **15 unused** |
| The 15 unused | `mtp.*` — the multi-token-prediction head (auxiliary; not part of the generation path and not consumed by either class) |
| Quantization metadata | 3030 bitsandbytes parameters (`absmax`, `nested_absmax`, `quant_map`, …) — not checkpoint tensors |
| Release | 16.696 GiB → **0.008 GiB** allocated; driver-free back to **22.544 GiB** |

The instruction not to treat "no missing weights" as proof is respected in the other direction too: the class identity was checked first, and only then the weights. Here both agree — the class is the declared one **and** the parameter names line up, which is a strictly stronger statement than the CausalLM path could make.

## 8. Minimal native generation (Part 14)

Text-only prompt `"Respond with exactly the word: READY"`, greedy:

| Metric | Value |
|---|---|
| Output | **`READY`** |
| Latency | 4.68 s |
| Prompt / generated tokens | 19 / **2** |
| Termination | **`eos`** (last token 248046 = `<|im_end|>`) |
| Images supplied | **none** |
| VRAM during | 16.696 GiB allocated |
| Backend telemetry | `backend: Qwen38NativeChatLLM`, `quantization: nf4`, `max_new_tokens_declared: null`, `top_p: 1.0`, `do_sample: false` |

That is the whole chain the stage asked for, end to end: native load → native processor → native generation.

## 9. Attacker interface adapter (Parts 15/16)

Only the loader is model-specific; the attack layer sees an ordinary `ChatLLM`:

```text
messages  ->  Qwen3VLProcessor.apply_chat_template(add_generation_prompt=True,
                                                   tokenize=True, return_dict=True)
          ->  Qwen3_5ForConditionalGeneration.generate(...)
          ->  processor.batch_decode(new_tokens)
          ->  the caller's parser (unchanged: json.loads, then the frozen
              _extract_json_block, else the raw string)
```

No system prompt added, no user prompt rewritten, no reasoning instruction, no examples, no JSON repair, no reasoning stripping, no retry, no temperature or sampling change, no output truncation. The full change list — five files, two of them new, with the non-goals enumerated — is in `stage3_code_changes.md`.

## 10. JSON qualification (Part 17)

24 generations through the **production** attacker step (`generate_crescendo_step`, `json_format=True`, temperature 0.7, `top_p` 1.0, `max_new_tokens` UNSET), classified with the project's own parse chain and nothing else. The harness is the Stage 2 code itself, re-pointed at this stage's raw-output directory, so the two runs cannot drift apart.

| Metric | Value |
|---|---|
| Valid JSON | **22 / 24 = 0.9167** (21 direct + **1** via the frozen structural extraction) |
| `JSON_PARSE_ERROR` | **2** (cases 3 and 8) |
| Failure shape | both: prose instead of a JSON object, natural EOS, 13-15 tokens |
| Empty queries | 0 |
| Missing `generatedQuestion` / `lastResponseSummary` | 2 / 2 (the same two cases) |
| Wrong field types | 0 |
| Reasoning-only outputs | 0 |
| Latency | mean **8.86 s** (median 8.14, min 2.98, max 13.33) |
| Generated tokens | mean 68 (min 13, max 114) |
| Termination | `stopping_criteria` 22, `eos` 2 |

One acceptance came from the frozen leniency path (the extraction chain every backend shares) rather than a direct parse — reported separately for exactly that reason. No repair, retry or fallback was applied to the two failures; they remain failures.

## 11. CausalLM vs native (Part 18)

`causallm_vs_native_comparison.json`.

**Architecture** — the selection criterion:

| | Auto-class | Loaded class | Matches the checkpoint's declaration |
|---|---|---|---|
| Stage 2 | `AutoModelForCausalLM` | `Qwen3_5ForCausalLM` | **no** — Stage 2's final record says so explicitly (`loaded_class_matches_checkpoint: false`, measured against the checkpoint's declaration); an earlier variant of that check compared against the *loaded* wrapper config's empty `architectures` field and was vacuous, and Stage 2 replaced it before writing the artifact |
| Stage 3 | `AutoModelForImageTextToText` | `Qwen3_5ForConditionalGeneration` | **yes** |

**Quantization**: 606 vs 496 `Linear4bit`, 82.95 % vs 82.71 % quantized, 16.688 vs 16.456 GiB, 98.9 s vs 100.4 s — the delta is the vision tower, as predicted in §6.

**Generation**:

| | CausalLM baseline | Native |
|---|---|---|
| Valid JSON | 23/24 = 0.9583 | 22/24 = 0.9167 |
| … direct / extracted | 20 / 3 | 21 / 1 |
| Parse errors | 1 | 2 |
| Latency mean (max) | 11.73 s (47.37 s) | **8.86 s (13.33 s)** |
| Generated tokens mean | 84.4 | 68 |
| Termination | 23 stopping-criteria, 1 EOS | 22, 2 |

The native path is slightly slower to fail and slightly faster overall, produced one fewer valid JSON on this sample, and — being the same weights — is not claimed to be *better* at the task. It is the correct architecture, which is the only thing this stage selects on. A higher or lower JSON rate would not have changed which class is legitimate.

## 12. Multi-turn qualification (Part 19)

Crescendo, Opposite Day and Acronym — the same three attacks, the same goals, the same two-turn budget and the same recording harness as Stage 2; only the attacker backend differs (`native_multiturn_qualification.json`).

| Metric | Crescendo | Opposite Day | Acronym |
|---|---|---|---|
| Attacker calls (valid / invalid) | **9 (8 / 1)** | **3 (2 / 1)** | **3 (2 / 1)** |
| Target calls | 8 | 2 | 2 |
| Target generated tokens | 30, **641**, 32, 22, 22, 25, 24, 30 | 28, 21 | 60, 27 |
| Target termination | `eos` ×8 | `eos` ×2 | `eos` ×2 |
| Target hit output budget | none | none | none |
| Evaluator calls (failures) | 9 (0) | 2 (0) | 3 (0) |
| Attacker prompt tokens | 1590 → **2796** | 819-945 | 1012-1202 |
| Turns produced | 0 | 0 | 0 |
| Peak VRAM | 17.93 GiB | 17.93 GiB | 17.93 GiB |
| Run outcome | `AttackGenerationError` on the 9th attacker call | on the 3rd | on the 3rd |

What actually happened: the attacker worked — it produced schema-valid escalation questions for **12 of 15 calls (80 %)**, driving Crescendo nine rounds deep with the prompt growing from 1590 to 2796 tokens, and the target answered at natural EOS every time. Each run then ended when one attacker reply arrived as **prose instead of a JSON object**, which the frozen contract records as a failed run (`AttackGenerationError`); the pipeline stopped there because no repair and no retry exist by design.

Two things this does *not* mean:

- **It is not a context or interface failure.** Context grew correctly (21 → 2796 tokens), the evaluator returned usable output on all 14 calls, and both failure-free roles behaved identically to Stage 2.
- **It is not specific to the native architecture.** The same failure shape appears in the isolated qualification on **both** paths — 2/24 native, 1/24 CausalLM — and the CausalLM path's own multi-turn sample was only 6 calls, too small to distinguish "0 %" from the 8-20 % measured here. The failure probability is a property of this checkpoint under the frozen no-repair contract, not of the loader.

The measured per-call rates (isolated 8.3 %, multi-turn 20 %, n = 24 and 15) are not statistically separable at these sample sizes, and no significance is claimed. What *is* established is the consequence: an 8-turn attack needs on the order of 16 attacker calls, so a non-trivial per-call failure rate compounds — every one of the three runs here terminated in `AttackGenerationError`, and that is a cost the reviewer must accept or address in a later stage, not something this stage may paper over.

**Harness note:** this artifact's `switching_validation` block is superseded by `native_residency_validation.json`. The Stage 2 switching helper calls `_get_tokenizer()`, which the multimodal backend does not expose (it has `_get_processor()`), so its attacker steps recorded an `AttributeError`. The Stage 3 residency validation performs the same ATTACKER→TARGET→JUDGE→ATTACKER sequence with the correct accessors and is the authoritative evidence (§13).

## 13. GPU / residency validation (Part 22)

Native attacker → target → judge → native attacker, three complete cycles, with the pre-load gate, post-load verification and ceiling check all active (`native_residency_validation.json`):

| Cycle | Role | Loaded class | Quantization | Resident models | Driver-free |
|---|---|---|---|---|---|
| 0 | attacker | **`Qwen3_5ForConditionalGeneration`** | `nf4` | 1 | 5.804 GiB |
| 0 | target | `LlamaForCausalLM` | none | 1 | 7.722 GiB |
| 0 | evaluator | `Qwen3ForCausalLM` | none | 1 | 7.423 GiB |
| 1 | attacker | `Qwen3_5ForConditionalGeneration` | `nf4` | 1 | 5.806 GiB |
| 1 | target | `LlamaForCausalLM` | none | 1 | 7.722 GiB |
| 1 | evaluator | `Qwen3ForCausalLM` | none | 1 | 7.423 GiB |
| 2 | attacker | `Qwen3_5ForConditionalGeneration` | `nf4` | 1 | 5.806 GiB |
| 2 | target | `LlamaForCausalLM` | none | 1 | 7.722 GiB |
| 2 | evaluator | `Qwen3ForCausalLM` | none | 1 | 7.423 GiB |

- **Errors: none.** No OOM, no CUDA error, no refusal.
- **max resident models = 1** in every one of the nine steps.
- The native attacker's processor is `Qwen3VLProcessor` on every load; the text roles use their tokenizers.
- Driver-free VRAM is **identical to the third decimal across cycles** (5.804/5.806, 7.722, 7.423) — no drift, which is the cheapest available proof of no leak.
- After `unload_all()`: **0.0 GiB allocated**, 22.679 GiB driver-free, cache empty.

Gates were not weakened and no threshold was changed. The only estimate that had to change was the *quantization-aware footprint* introduced in Stage 2 (18.4 GB estimate for a 16.7 GiB load), which is why the 4-bit native attacker is admitted at all.

## 14. Decision (Part 24) and the two required answers (Part 25)

### 14.1 Classification: NATIVE CONDITIONALLY QUALIFIED

Every criterion in Part 24's NATIVE QUALIFIED list is met — declared architecture loaded, native processor working, official weights correctly consumed, NF4 working, text generation working, attacker interface working, JSON contract working, multi-turn generation executing, model switching working, VRAM safe. One documented runtime deviation keeps it one tier below:

> The checkpoint occasionally answers the attacker prompt **without the required JSON envelope** — measured 2/24 (8.3 %) isolated and 3/15 (20 %) in the multi-turn runs. Under the frozen contract that is an `AttackGenerationError` with no repair path, and it terminated **3 of 3** multi-turn runs, after 2-9 valid attacker turns each.

The condition is stated, quantified and *not* caused by the native architecture: the same failure shape is present in the CausalLM baseline (1/24 isolated), and its 6-call multi-turn sample is too small to contradict the 8-20 % range. It is a property of this checkpoint under a no-repair contract, and it will affect any Phase 17 attack run more strongly the longer the run is.

Not NATIVE NOT QUALIFIED, because nothing was unreliable in the sense that part is about: the interface, the processor, the loader, the context handling, the switching and the evaluator all worked on every call, and the failures are the model's own, correctly detected and reported rather than hidden. Not NATIVE QUALIFIED, because a deviation that ends three of three runs has to be attached to the verdict rather than buried in a caveat.

Not NATIVE BLOCKED: transformers supports the architecture natively, the model fits, the processor handles text-only interaction, and the native load was demonstrated end to end.

### 14.2 The two questions (Part 25), answered separately

**Q1 — Is the Qwen3.8-27B `Qwen3_5ForConditionalGeneration` checkpoint legitimately usable as the Phase 17 attacker through a native architecture path on this RTX 4500?**

**Yes, with the stated condition.** It loads as the class it declares (`AutoModelForImageTextToText` → `Qwen3_5ForConditionalGeneration`), on the official weights, under the approved 4-bit NF4 runtime, in **16.69 GiB** of a 23.99 GiB card — 5.8 GiB left free with the target and judge rotated out. It generates text-only with no fabricated image input, satisfies the JSON contract on 22 of 24 isolated probes, drives real multi-turn attacks that grow context correctly, rotates with the target and judge with **zero** residency errors, and returns VRAM to baseline (0.0 GiB, 22.68 GiB free). The condition attached to "usable": the checkpoint's per-call tendency to omit the JSON envelope will terminate long runs as `AttackGenerationError`, and that rate is measured rather than eliminated.

**Q2 — Is the previous `Qwen3_5ForCausalLM` path methodologically acceptable, or must it be rejected as an architecture substitution?**

**It must be rejected as an architecture substitution** — on identity, explicitly not on observed breakage. Three measured facts decide it, and a fourth bounds the claim:

1. The class loaded was not the class the checkpoint declares (`Qwen3_5ForCausalLM` vs `Qwen3_5ForConditionalGeneration`) — a mismatch Stage 2 itself had already recorded as `loaded_class_matches_checkpoint: false`.
2. The two classes expect **different parameter names**: the native class wants `model.language_model.layers.*` and a vision tower; the CausalLM class wants `model.layers.*` and has no vision module at all — while the checkpoint publishes the former and 333 `model.visual.*` tensors.
3. The declared class is loadable and executes here under the *same* quantization, so no hardware or runtime constraint forced the substitution. It was avoidable, and it was avoided in this stage.
4. The substitution was **not observed to damage text quality** — transformers remaps the keys, the CausalLM path consumed the text weights and scored a comparable JSON rate. So the rejection is about architectural faithfulness, not about a defect anyone measured. That distinction is the honest version, and it matters: a reader should not conclude the CausalLM path produced wrong answers, only that it ran the checkpoint through a class the checkpoint does not declare.

Under a reproduction that must be able to say *which architecture executed*, that is not acceptable when the declared architecture is both available and proven. Stage 2 recorded it as "the single most consequential open question"; this stage closes it.

### 14.3 Recommended configuration (for the reviewer, not frozen here)

```text
ATTACKER (recommended) : Qwen/Qwen3.8-27B via backend `qwen38_native`
AUTO CLASS             : AutoModelForImageTextToText
LOADED CLASS           : Qwen3_5ForConditionalGeneration
REVISION               : 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
PROCESSOR              : Qwen3VLProcessor (checkpoint-declared)
REFERENCE PRECISION    : BF16
TESTED PRECISION       : 4-bit
QUANTIZATION           : NF4 PTQ at load time (bitsandbytes, bf16 compute, double quant)
RESIDENCY              : sequential (16.7 GiB cannot co-reside with a 15 GiB partner)
GENERATION             : temperature 0.7, top_p 1.0, max_new_tokens UNSET,
                         enable_thinking False (same template setting as the baseline)
STATUS                 : NATIVE CONDITIONALLY QUALIFIED
CONDITION              : per-call JSON-envelope omission (8-20 %) terminates runs under the
                         frozen no-repair contract — accepted knowingly, not discovered mid-study
```

The Stage 2 CausalLM implementation remains available and unmodified, as Part 23 requires; it is simply not the path this stage recommends.

### 14.4 Final gate (Part 30)

```text
Declared architecture:
Qwen3_5ForConditionalGeneration

Native class:
Qwen3_5ForConditionalGeneration          (AutoModelForImageTextToText)

Native processor:
Qwen3VLProcessor                          (checkpoint-declared, loaded)

Native NF4:
PASS    — 606 Linear4bit, 82.95 % of params quantized, 16.688 GiB, 98.9 s

Native generation:
PASS    — text-only, 'READY', 2 tokens, natural EOS, no images

Native JSON:
22/24 valid (0.9167) — 21 direct + 1 via the frozen extraction, 2 prose failures

Native multi-turn:
EXECUTES — 12/15 attacker replies valid; all 3 runs ended on a prose reply
           (AttackGenerationError), 0 accepted turns; target EOS 12/12;
           evaluator 14/14 usable

Native residency:
PASS    — 3 cycles × 3 roles, max resident 1, zero errors, 0.0 GiB after release,
          22.68 GiB driver-free, processor/class identity correct every cycle

CausalLM substitution:
NOT ACCEPTABLE — different class from the declared architecture, different expected
                 parameter names, no vision tower, and avoidable because the declared
                 class loads and runs on this hardware

FINAL NATIVE STATUS:
CONDITIONALLY QUALIFIED
```

## 15. Test state

| Gate | Result |
|---|---|
| Full suite after the Stage 3 changes | **804 passed, 2 failed** |
| Stage 3 backend tests (`tests/test_phase17_stage3_native_backend.py`) | **14 passed** |
| Backends adjacent to the change (GLM, Ornith, Stage 2 infrastructure) | **157 passed** |

The two failures are the same pre-existing ones Stage 2 reported and did not
repair: `test_default_config_valid` and `test_attacker_present` assert that
`Qwen/Qwen3.5-4B` is present in the local cache, and that model was evicted in
Phase 17 Stage 0 under a user-approved eviction. They are environment state, not
code, and this stage touched neither.

One pre-existing test was updated, and only because the change is intentional:
`test_ornith15_attacker_backend.py::TestOrnith15Profile::test_profile_is_registered`
asserts the complete list of registered profiles, which now includes
`qwen38_native`. The assertion keeps its exactness; no test was weakened and
nothing was changed to make a test pass.

## 16. Caveats

- **Everything measured here is 4-bit NF4**, never the BF16 reference. Any statement about `Qwen3.8-27B` in this document means "the official checkpoint, quantized at load time by bitsandbytes NF4".
- **The `mtp.*` head is unused** (15 tensors). It is an auxiliary multi-token-prediction head; neither the native class nor the CausalLM class instantiates it, and the model generates correctly without it. Recorded rather than glossed.
- **The JSON failure rate is a measurement, not a pass mark.** 2/24 native, 1/24 CausalLM on different samples; the project has no numeric JSON gate, and none was invented.
- **The comparison is not a seeded A/B.** Both runs were unseeded (temperature 0.7), so per-prompt output equality was never expected and is not claimed; the comparison is architectural and distributional.
- **Three harness defects occurred while building this validation** and are recorded rather than hidden: each was a reference the measurement itself held (the model, the pipeline root, the wrong accessor on the text backend), which kept weights alive so the *correctly* behaving pre-load gate refused the next load. The final validation drops every reference in a `finally` block and records driver-free VRAM before each activation so a leak is visible immediately.
- **No ASR, no pilot.** See §18.

## 17. Evidence index

| Artifact | Path (under `results/phase17_model_qualification/stage3_qwen38_native/`) |
|---|---|
| Architecture audit | `architecture_audit.json` |
| Native loader probe | `native_loader_probe.json` |
| NF4 + census | `native_quantization_validation.json` |
| Native load + release | `native_load_validation.json` |
| Minimal generation | `native_generation_probe.json` |
| JSON qualification | `native_json_qualification.json` |
| Multi-turn qualification | `native_multiturn_qualification.json` |
| Residency validation | `native_residency_validation.json` |
| CausalLM vs native | `causallm_vs_native_comparison.json` |
| Code changes | `stage3_code_changes.md` |
| Raw attacker outputs | `raw_native_attacker_outputs/` |
| Raw conversations | `raw_native_multiturn_outputs/` |
| Scripts | `scripts/phase17_stage3_{architecture_audit,native_qualification,native_multiturn}.py` |

## 18. What was not done

- No ASR, no attack-success comparison, no NBF ON/OFF, no 90-run pilot, no 1,200-run study.
- No other attacker was tested: Qwen3.6-27B, Ministral, Devstral and Bonsai are out of scope for this stage.
- No target, judge, rubric, prompt, parser, NBF or success-criterion change.
- The Stage 2 CausalLM implementation and its evidence were not modified, overwritten or deleted.
