# Phase 16.6 — Replace Phi-4-mini Target with Phi-4

**Status:** stopped at the feasibility gate.
**No ASR. No 90-run pilot. No 1,200-run study. No production file modified.**

Question this phase had to answer: *does replacing the current Phi-4-mini target with
the paper's local Phi-4 remove the target-generation incompatibility identified as F1?*

Answer: **it cannot be determined on this hardware — Phi-4 does not fit at the
reference precision.** The phase brief's own rule for this case is to stop, which is
what happened. Along the way the reference local target path was located, and it
**materially changes how F1 should be read**.

---

## 1. Reference model verification (§2)

| question | value | source |
| --- | --- | --- |
| Exact model name | **`microsoft/phi-4`** | author's code |
| HF identifier | `microsoft/phi-4` | `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/generate.py:12` — `model_id = "microsoft/phi-4"` |
| Revision specified | **NOT SPECIFIED** (not pinned in the reference) | `generate.py:10-22` |
| Precision | **bfloat16** | `generate.py:20` — `torch_dtype=torch.bfloat16` |
| FP16 / FP32 / quantization | no / no / **none** | `generate.py:18-22` |
| Runtime | HuggingFace `AutoModelForCausalLM` + `device_map="auto"` | `generate.py:18-22` |
| `max_new_tokens` | **256** | `generate.py:37` |
| Stop sequences | none beyond the EOS list | `generate.py:30-38` |
| EOS configuration | `eos_token_id=[tokenizer.eos_token_id, convert_tokens_to_ids("<|eot_id|>")]` — note `<|eot_id|>` is a **Llama** token, not a Phi token, so in practice this is EOS-only | `generate.py:30-33` |
| JSON / structured output | **not used** — the local Phi-4 path asserts it off: `assert not json_format` | `generate.py:100` |
| Instruction-tuned or base | **instruction-tuned** (`microsoft/phi-4`); the LoRA baseline points at a finetuned variant via a placeholder path | `generate.py:9-12` |
| Context length | `max_position_embeddings = 16384` | model `config.json` |
| Paper's API path, for contrast | GPT-4o with **no** `max_tokens` (unbounded, server-side maximum only) | `generate.py:137-147` |

Everything not stated above is recorded as **NOT SPECIFIED** in
`generation_config_validation.json`; nothing was invented.

**This is the decisive context for F1:** the reference's *local* target path — the
only path that corresponds to our local-model substitution — **does specify an output
ceiling of 256 tokens**, and uses `top_p=0.9`. Only the paper's *API* path is
unbounded.

---

## 2. Feasibility (§4) — MODEL_NOT_FEASIBLE_AT_REQUIRED_PRECISION

Measured, not estimated:

| quantity | value | how measured |
| --- | --- | --- |
| Parameters | **14,659,507,200** (14.660 B) | model built from the real `config.json` on the **meta device** (zero memory) and counted |
| bf16 weight bytes | **29.319 GB** | parameters × 2 |
| Cross-check | hub index `metadata.total_size = 29,319,014,400` bytes = **29.319 GB** | the hub's own `model.safetensors.index.json` |
| GPU free VRAM | **24.385 GB** | `torch.cuda.mem_get_info()` on the RTX 4500 Ada |
| GPU total | 25.763 GB | same |
| **Shortfall** | **+4.934 GB** | — |
| Local availability | **weights not present**: 6 `.incomplete` blobs totalling 6.867 GB; no completed shard; tokenizer fails to load | hub cache inspection |

Two independent methods agree on the footprint (meta-device count and the hub index),
so the shortfall is arithmetic, not an estimate.

**Project gates would refuse it independently of raw size:**
`pre_run_check` requires `free − incoming ≥ 3.0 GB` → needs ≥ 32.319 GB free
(have 24.385) → REFUSED. Post-load residual would be **−4.934 GB** against a 1.5 GB
floor → REFUSED.

**Stop.** No load was attempted, no quantization was introduced, and no
CPU-offload workaround was applied. `target_stopping_probe.json` records every
metric as **null** rather than a placeholder — the F1 stopping test could not be run.

### Alternatives that were rejected (and why)

| option | disposition |
| --- | --- |
| 8-bit / 4-bit quantization (~14.7 / ~7.3 GB) | **rejected** — the reference used bf16 with no quantization; the brief forbids quantizing to force a fit |
| `device_map="auto"` CPU offload (~5 GB to host RAM) | **rejected without operator approval** — technically it would run (the reference itself passes `device_map="auto"`), but it departs from the frozen pipeline's `device_map="cuda"` + VRAM-gate design and is the workaround class §4 prohibits |
| A larger GPU | not available on this host |
| Downloading the remaining 22.5 GB to observe the OOM | **not done** — the outcome is fixed by arithmetic; 29.3 GB of transfer would prove nothing new |

---

## 3. Baseline and integrity (§3, §12)

`results/phase16_6_phi4_target/baseline_manifest.json` (captured before anything else):

```text
git commit        : b46d6fd7ed37b7a6ee9e643476ae2b2263b85624  (branch main)
target model      : microsoft/Phi-4-mini-instruct   revision NOT SPECIFIED
target precision  : bfloat16        device_map: cuda      max_new_tokens: 256
frozen config     : 27B660A7052054521A94E3430013D9863964E825157056D3861228250F19A358
NBF checkpoint    : CEA1A75BCEF4FC515814B69C42541C95114F587ABCC4505C9B096BBFA2A136FE
dataset           : AC789DE8859E755C11EE6CD600DD6A2EB88128C7AB119056DCE3F7F125D014EB
Track A (180)     : FCB952F1598B697B033D605B0791CFBB64A010F8E43CD1ADF3D0EDC359D8A092
Track A (108)     : C12A562A9454D83B6CC0EE93B2653F4AA5AB058C5AC3E50CEED4EF7A8EC57FB3
results/phase14/batch00.jsonl : FCB952F1598B697B0… (unchanged)
```

`results/phase15_pilot_30/trackA_frozen/`, `results/phase14/` and
`configs/reproduction_phase14_frozen.yaml` were read only and verified unchanged.
All Phase 16.6 artifacts are new files in `results/phase16_6_phi4_target/`.

---

## 4. Two reference-fidelity discoveries that change F1's reading

### G1 — the reference *does* specify a local target ceiling: 256

`generate.py:37` — the reference's own `local_phi` calls
`generate(..., max_new_tokens=256, ...)`.

**This corrects Phase 16.5.** F1 was declared `BLOCKED` on the reasoning that the
256-token cap was an *artificial* project choice with no reference counterpart and
that no defensible stopping rule existed. The second half of that is now known to be
wrong: for the **local** target path — the one our substitution corresponds to — 256
*is* the reference value. The project's frozen configuration was already faithful on
this parameter.

What remains true from Phase 16.5: with the cap removed, Phi-4-mini does not
terminate on its own (1/8 natural stops; median 1,300 tokens still generating), and
95.0 % of frozen target responses sat exactly at the 256 boundary. So the open
question is no longer "should we remove the cap?" but **"does the paper's local
Phi-4 reach EOS naturally inside the reference's own 256-token allowance, where
Phi-4-mini does not?"** That is precisely the question this phase was meant to
answer, and it is unanswerable here for a hardware reason.

### G2 — the reference local target uses `top_p = 0.9`

`generate.py:41` — `top_p=0.9`, with `do_sample=True` and the caller's temperature.

Neither project configuration matches:

| parameter | reference local | frozen config | Phase 16.5 config |
| --- | --- | --- | --- |
| `max_new_tokens` | **256** | 256 ✓ | `null` (uncapped) ✗ |
| `top_p` | **0.9** | inherited ≈1.0 ✗ | 1.0 ✗ |
| `temperature` | 0.7 | 0.7 ✓ | 0.7 ✓ |
| `do_sample` | true | true ✓ | true ✓ |
| dtype / quantization | bfloat16 / none | bfloat16 / none ✓ | bfloat16 / none ✓ |

G2 is *not* the same issue as F3. F3 was about declared-vs-effective sampling (a
reporting defect, fixed in Phase 16.5). G2 is about **which value is correct**: the
reference's local path uses 0.9, so the honest options are to declare 0.9 for the
local roles or to document the deviation. Under the Phase 16.5 rule, any value the
reference does not specify is reported as `inherited_from_generation_config` rather
than asserted — that mechanism stays in place and was not changed here.

---

## 5. Sections that could not be executed

| brief section | status |
| --- | --- |
| §5 model loader integration | **NOT DONE** — no loadable model; the existing `HFLocalChatLLM` is architecturally compatible (`Phi3ForCausalLM`, standard `AutoModelForCausalLM`), but that was not exercised |
| §6 generation-parameter recording | **audited, not executed** — see G1/G2; no target generation occurred |
| §7 F1 stopping test (20 probes) | **NOT RUN** — all metrics `null`, never estimated |
| §9 Phi-4-mini vs Phi-4 comparison | **NOT COMPARABLE** — one column cannot be measured |
| §10 response-quality sanity check | **NOT RUN** — no responses to inspect |
| §11 GPU residency under Phi-4 | **NOT MEASURED** — recorded as null (no regression occurred: nothing changed) |
| §13 no ASR | **honoured** — no ASR, no pilot, no 1,200-run study |

---

## 6. Regression tests (§12)

```text
existing full test suite : see results/phase16_6_phi4_target/regression_after.json
```

No production code was modified in this phase, so the suite reflects the Phase 16.5
state (which was 728 passed / 0 failed). Every affected area named in §12 was
verified unchanged by file hash or by the fact that the corresponding module was
never touched: Qwen attacker, Llama evaluator, GLM backend, Ornith backend, attack
parity, NBF parity, F2 classification, F3 explicit sampling, model switching, GPU
residency invariants and Track A files.

---

## 7. Deliverables

```text
results/phase16_6_phi4_target/baseline_manifest.json
results/phase16_6_phi4_target/model_qualification.json
results/phase16_6_phi4_target/generation_config_validation.json
results/phase16_6_phi4_target/gpu_qualification.json
results/phase16_6_phi4_target/target_stopping_probe.json
results/phase16_6_phi4_target/regression_after.json
results/phase16_6_phi4_target/code_changes.md
results/phase16_6_phi4_target/phase16_6_report.md
```

Raw target outputs were not retained because no target output exists.

---

```text
PHASE 16.6 DECISION

PHASE 16.6 = BLOCKED

Reason:
MODEL_NOT_FEASIBLE_AT_REQUIRED_PRECISION
  microsoft/phi-4 requires 29.319 GB at bfloat16 (14,659,507,200 parameters; the
  figure is confirmed independently by the hub weight index). The GPU provides
  24.385 GB free and 25.763 GB total — a 4.934 GB shortfall. The project's own
  pre-load gate would refuse it (needs >= 32.319 GB free). No quantization, no
  CPU-offload workaround and no substitution ceiling was introduced to force a fit.

Reference target model:
VERIFIED — microsoft/phi-4 at bfloat16, max_new_tokens=256, top_p=0.9,
do_sample=True, no quantization, device_map="auto", context 16384.

F1 target stopping:
UNRESOLVED (not answered either way)
  The question "does Phi-4 terminate naturally inside the reference's own
  256-token allowance where Phi-4-mini does not?" could not be tested because the
  model cannot be loaded on this hardware.

F1 status correction:
  The reference's LOCAL target path specifies max_new_tokens=256, so the frozen
  configuration's cap was reference-faithful and Phase 16.5's premise that no
  reference ceiling exists was incorrect. Whether to revert Phase 16.5's uncapped
  change is an operator decision.

F2 failure classification:
UNCHANGED (fixed in Phase 16.5) — no proposal in this phase; the attacker-side JSON
question is unaffected by the target substitution.

F3 sampling:
UNCHANGED (fixed in Phase 16.5) — no proposal in this phase.
NEW: G2 — the reference local path uses top_p=0.9, which neither the frozen config
(inherited ~1.0) nor the Phase 16.5 config (explicit 1.0) matches. Declaring 0.9 for
the local roles would close a real fidelity gap, but it changes sampling for every
role and is therefore left to the operator.

GPU safety gates:
UNCHANGED — no regression, 0 OOM/CUDA errors, architecture untouched.

Track A integrity:
PASS — 180-record FCB952F1598B697B…, 108-record C12A562A9454D83B…, frozen config
27B660A705205452…, checkpoint CEA1A75B…, dataset AC789DE8… — all unchanged.

Tests:
see regression_after.json; unaffected areas verified unchanged.

ASR experiment:
NOT RUN

90-run pilot:
NOT RUN

NEXT STEP (§17):
STOP. Do not launch ASR. The decision required is not a code change but a hardware /
methodology one, and it is the operator's:
  (a) accept that the local target stays Phi-4-mini and that F1's residual question
      ("does the paper's Phi-4 terminate naturally?") remains permanently UNKNOWN on
      this hardware; or
  (b) approve a specific, declared deviation that makes Phi-4 runnable (quantization
      or CPU offload), knowing it departs from the reference's bfloat16
      full-residency setup; or
  (c) provide hardware that fits 29.3 GB at bfloat16.
Nothing else in the reproduction is blocked by this: F2 and F3 are fixed, and the
only other open item remains the attacker-side 256-token ceiling identified in
Phase 16.5.
```
