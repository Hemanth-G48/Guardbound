# Phase 16 — Ornith-1.5-9B Attacker Backend (selectable; Qwen3 and GLM unchanged)

**Date:** 2026-09-28
**Scope:** backend implementation + qualification. **The 90-run pilot was NOT launched.**

---

## 1. Model identity

| | |
|---|---|
| Repository | `ornith-ai/Ornith-1.5-9B` (not GGUF, not Mobile, not 1.0) |
| **Exact revision** | **`489cb97981b8654bcfcf30ce1f94ed1b62e07b53`** |
| Revision date | 2026-08-23 |
| dtype | **bfloat16** |
| Quantization | **none** |
| Repository size | 19.31 GB (4 safetensors shards) |

## 2. Architecture, as resolved (not assumed)

```text
AutoConfig class        : Qwen3_5Config
config.model_type       : qwen3_5
config.architectures    : ["Qwen3_5ForConditionalGeneration"]
config.auto_map         : null            (no remote code required)
text_config             : Qwen3_5TextConfig (qwen3_5_text, 32 layers, hidden 4096)
vision_config           : present         (image-text-to-text checkpoint)
```

**Why Ornith needed its own backend rather than a `glm4v` alias.** `qwen3_5`
appears in *both* the causal-LM and the multimodal auto mappings, so
`pipeline("text-generation")` would resolve `Qwen3_5ForCausalLM` — a class this
checkpoint has no weights for. The backend therefore loads through
`AutoModelForMultimodalLM` (the normal AutoModel mechanism, config-driven) and
then **asserts** that the instantiated class is the one the checkpoint declares:

```text
model class (verified)  : Qwen3_5ForConditionalGeneration
processor class         : Qwen3VLProcessor   (tokenizer: Qwen2Tokenizer)
```

A mismatch raises `Ornith15ArchitectureError`
(`failure_class = MODEL_INTERFACE_INCOMPATIBLE`) instead of silently running the
wrong architecture.

## 3. Measured runtime characteristics

| Metric | Value | How measured |
|---|---:|---|
| **VRAM footprint** | **18.82 GB** allocated (peak 19.51 GB, reserved 19.72 GB) | real load, `torch.cuda.memory_allocated()` |
| Load time | **15.85 s** (Stage A) / 10.79 s (switching) | weights warm in the OS cache |
| Generation latency | mean **5.47 s**, max 8.33 s (n=10) | Stage A |
| dtype at load | bfloat16 | `from_pretrained(dtype=torch.bfloat16)` |

The 18.82 GB measurement is why the profile declares
`residency: sequential`: it cannot co-reside with the 7.67 GB target on this
24.57 GB card.

## 4. JSON validity (Stage A — 10 isolated generations)

| | |
|---|---:|
| Generations | 10 |
| **Valid JSON** | **8** |
| Malformed JSON | 2 |
| Empty | 0 |
| Reasoning-only | 0 |
| Extra text | 0 |
| Missing fields / wrong types (parse-level) | 0 |
| `json_validity_rate` | **0.80** |
| Gate (≥ 0.5, no empties) | **PASSED** |

## 5. Native thinking control (measured, then declared)

The task required measuring both the model's default and its native option
rather than copying GLM's setting. 8 goals per condition, frozen interface:

| Condition | replies carrying the attacker schema |
|---|---:|
| thinking default + `constrained_json` | **1 / 8** |
| `enable_thinking=False` + `constrained_json` | **5 / 8** |
| thinking default, no `constrained_json` | 0 / 8 (plain prose analysis) |
| `enable_thinking=False`, no `constrained_json` | 0 / 8 (```json-fenced) |

`enable_thinking` is therefore declared in the Ornith attacker config as a
**model configuration parameter** — the same parameter the frozen attacker block
already declares and that GLM uses. No prompt, template or parser was modified;
no reasoning stripping was added.

## 6. Stage B — multi-turn context

```text
sequences tested              : 5   (goals 0-4, up to 3 turns each)
turns attempted               : 8
context failures              : 4
empty queries                 : 0
repeated queries              : 0
goal preserved in system msg  : True
turn ordering correct         : True
sequences with >=2 turns      : 2 / 5
GATE (>=4 sequences, no failures) : NOT PASSED
```

**What the four failures actually were** (raw output preserved, nothing repaired):

| Shape | Count | Detail |
|---|---:|---|
| Stray quote before the key | 2 | `…,\n"\nlastResponseSummary": …` — `JSONDecodeError` → `AttackGenerationError` |
| Key mangled with leading spaces | 2 | `"  lastResponseSummary"` — **valid JSON but a missing required field** → `AttackGenerationError: missing 'lastResponseSummary'` |

Both are classified `JSON_PARSE_ERROR` by the frozen taxonomy, exactly as a Qwen
failure would be. The second shape is worth noting: the reply *parses*, but the
frozen Crescendo parser requires the exact key and rejects it — the frozen
parser was not loosened to accommodate it.

Context handling itself is correct — verified from the messages actually sent:

```text
turn 1: roles=[system, user]                                   goal present
turn 2: roles=[system, user, assistant, user]                  prior query + target feedback
turn 3: roles=[system, user, assistant, user, assistant, user] 2 prior pairs
```

## 7. Stage C — attack interface

**NOT RUN.** The gate is explicit ("only if A/B pass"): Stage A passed, Stage B
did not. Neither `python scripts/qualify_ornith15_attacker.py --stage C` nor
`--force-stage-c` was executed. No attack was executed with Ornith, so no ASR
statement of any kind is possible for this attacker.

## 8. Regressions

| Check | Result |
|---|---|
| **Qwen regression** (`--attacker-model qwen3`) | **PASS** — load, generation, JSON parsing, multi-turn, unload; identity `Qwen/Qwen3-4B-Instruct-2507`, backend `hf_local`, residency `pinned`, unchanged |
| **GLM regression** (`--attacker-model glm46v`) | **PASS** — 1/1 generation valid, backend `GLM4VChatLLM`, switching intact |

## 9. Model switching (real loads, VRAM evidence)

```text
Qwen -> GLM -> Ornith -> Qwen      passed, no leak, no cross-model state
  Qwen3-4B-Instruct-2507  HFLocalChatLLM     8.045 GB -> 0.0
  GLM-4.6V-Flash          GLM4VChatLLM      20.586 GB -> 0.0
  Ornith-1.5-9B           Ornith15ChatLLM   18.820 GB -> 0.0
  Qwen3-4B-Instruct-2507  HFLocalChatLLM     8.045 GB -> 0.0

Qwen -> Ornith -> GLM -> Qwen      passed, no leak  (covers Ornith -> GLM)
Qwen -> Ornith -> Qwen             passed, no leak  (covers Qwen <-> Ornith)
```

Every step verified: correct `resident_model_ids`, weights actually moved VRAM,
allocation back to 0.0 GB after unload, no stale pipeline/tokenizer, no wrong
identity, no residency-count violation. Sequential residency held throughout:
**max `resident_count` = 1**, 0 load refusals, and all VRAM gates
(pre-load, post-load, per-run floor, 22 GB ceiling) remained active — none were
weakened, disabled or bypassed.

## 10. Tests

```text
full suite            : 694 passed, 0 failed   (655 before this phase + 39 new)
new this phase        : tests/test_ornith15_attacker_backend.py -> 39 passed
coverage              : profile selection (all three), non-drift of qwen3/glm46v,
                        backend dispatch + unknown -> ValueError, native auto-class
                        resolution, architecture assertion (accept + reject),
                        message/role order, JSON failure behaviour, no-retry,
                        no reasoning stripping, residency + run-boundary release,
                        switching Qwen<->Ornith and GLM<->Ornith
```

## 11. Files changed

| Path | Status | Purpose |
|---|---|---|
| `src/guardbound/llm/multimodal_client.py` | **new** | `MultimodalChatLLM` — the shared AutoProcessor → native `generate` → frozen-JSON machinery extracted from the GLM backend so Ornith reuses it instead of duplicating it |
| `src/guardbound/llm/ornith15_client.py` | **new** | `Ornith15ChatLLM`: `AutoProcessor` + `AutoModelForMultimodalLM`, with the architecture assertion |
| `configs/attacker_ornith15.yaml` | **new** | Standalone Ornith config + provenance block |
| `tests/test_ornith15_attacker_backend.py` | **new** | 39 tests |
| `scripts/qualify_ornith15_attacker.py` | **new** | Thin entry point for the shared harness |
| `src/guardbound/llm/glm4v_client.py` | modified | Now a thin subclass of the shared base; the loader seams, class name and public surface are unchanged |
| `src/guardbound/llm/attacker_profiles.py` | modified | Added the `ornith15` profile (and the `BACKEND_ORNITH15` constant). `qwen3` / `glm46v` entries untouched |
| `src/guardbound/llm/provider_factory.py` | modified | Backend dispatch extended to three backends; unknown still raises. The `hf_local` and `glm4v` construction calls are unchanged |
| `scripts/qualify_glm_attacker.py` | modified | Harness made attacker-agnostic (`--profile/--config/--out-dir/--stage-c-dir/--label/--switch-sequence`); defaults reproduce the Phase 15 GLM run |

The GLM refactor is behaviour-preserving by construction and by test: the 40
Phase 15 GLM tests all still pass, and the GLM regression re-qualified on GPU.

## 12. Files explicitly untouched

```text
Track A code unchanged        (attacks/*, runner.py, base.py, NBF, predictor)
Track A results unchanged     (results/phase14/batch00.jsonl
                               = results/phase15_pilot_30/trackA_frozen/…180records.jsonl
                               = FCB952F1598B697B033D605B0791CFBB…)
Track A configuration unchanged (config_hash 61ea203c20e68c51 — still the frozen hash)
NBF unchanged                 (checkpoint, threshold, scoring)
target unchanged              (microsoft/Phi-4-mini-instruct)
evaluator unchanged           (meta-llama/Llama-3.2-3B-Instruct)
attack algorithms unchanged   (Crescendo / OppositeDay / Acronym / ActorAttack)
Phase 15 evidence unchanged   (results/phase15_glm46v/* not overwritten; the
                               GLM regression wrote to a separate directory)
```

## 13. Qualification status

```text
BACKEND IMPLEMENTED              : YES
  --attacker-model ornith15 works from the CLI; no source edit needed to switch.

TECHNICAL QUALIFICATION          : PASS
  loads (18.82 GB, bfloat16, native class verified), generates, unloads cleanly,
  switches in every required direction, no VRAM leak, correct model identity,
  all VRAM guards intact, 694/694 tests green, Qwen and GLM regressions pass.

ATTACKER CAPABILITY QUALIFICATION: NOT PASSED
  Stage A passed with 8/10 valid JSON (80%). Stage B did not pass its gate:
  4 of 8 multi-turn turns were rejected by the frozen parser (stray quote in 2,
  a whitespace-mangled required key in 2), so only 2 of 5 sequences reached
  2+ turns. Stage C was therefore not run and no attack was executed.
  The backend loads and drives the model correctly; the model's JSON fidelity
  through the frozen interface is the limiting factor.
```

Failure-class mapping per the task's policy: the two shapes above are recorded as
`JSON_PARSE_ERROR` (raw output preserved). Nothing was converted into another
class, nothing was repaired, no retry or fallback exists, and no ASR comparison
against Qwen was made.

Evidence: `results/phase16_ornith15/` (`qualification/`, `glm_regression/`,
`full_switch/`, `reverse_switch/`, `isolate_thinking.json`).

```text
IMPLEMENTATION COMPLETE — ORNITH-1.5-9B QUALIFICATION COMPLETE —
90-RUN PILOT NOT LAUNCHED
```
