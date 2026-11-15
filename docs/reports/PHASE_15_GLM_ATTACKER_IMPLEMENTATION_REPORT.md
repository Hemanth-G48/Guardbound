# Phase 15 — GLM-4.6V-Flash Attacker Backend (selectable, Qwen3 unchanged)

**Date:** 2026-09-28
**Scope:** implementation + qualification. The 90-run pilot was **NOT** launched.

---

## 1. Files changed

| Path | Status | Purpose |
|---|---|---|
| `src/guardbound/llm/glm4v_client.py` | **new** | `GLM4VChatLLM` — native `AutoProcessor` + `Glm4vForConditionalGeneration` + `model.generate(...)` backend. Reuses the frozen `_extract_json_block` / `_JSONStoppingCriteria` and `build_json_logits_processor`; adds no repair, retry, fallback or reasoning handling. Module-level loader seams (`load_processor`, `load_model_class`) make it unit-testable without weights. |
| `src/guardbound/llm/attacker_profiles.py` | **new** | Named attacker profiles (`qwen3`, `glm46v`) + `apply_attacker_profile()` + `attacker_runtime_telemetry()`. The single place the attacker is switched. |
| `configs/attacker_glm46v.yaml` | **new** | Standalone GLM experiment config (existing schema, `models.attacker.backend/revision/dtype/quantization/residency`), with a provenance block listing exactly what differs from the frozen Track A config. |
| `tests/test_glm_attacker_backend.py` | **new** | 40 focused tests: selection, backend dispatch, GLM message conversion/role order, native path (and a parsed-AST assertion that `AutoModelForCausalLM`/`pipeline` are never referenced), JSON failure behaviour, residency, release-at-boundary, switching. |
| `scripts/qualify_glm_attacker.py` | **new** | Qualification harness: Stages A/B/C plus a Qwen regression stage and a real switching stage. Stage C is gated on A/B passing. |
| `src/guardbound/llm/provider_factory.py` | modified | Local **backend dispatch** (`backend: hf_local` \| `glm4v`, unknown → `ValueError`), `revision` on the pipeline backend rejected instead of silently ignored, residency applied only when declared. The `hf_local` construction call is byte-identical to before. |
| `src/guardbound/llm/model_manager.py` | modified | `set_residency()` / `residency_mode()` / `release_sequential()` + `register()` now releases a *re-pointed* role. No threshold, gate or check was changed. |
| `scripts/phase14_full_reproduction.py` | modified | `--attacker-model` selector, `--out` output override, `attacker_runtime` telemetry, run-boundary release of sequential roles. |
| `tests/test_phase14_8_telemetry.py` | modified | Schema pin 55 → 56 (the new `attacker_runtime` field), with the reason documented in the test. |

Total: **+204 / −18** lines across 4 tracked files, plus 5 new files.

## 2. Files not changed

```text
Track A attack code: unchanged        (crescendo_paper.py, opposite_day.py, acronym.py,
                                       actor_attack.py, runner.py, base.py)
NBF: unchanged                        (models/, defense/, thresholds, scoring)
target: unchanged                     (microsoft/Phi-4-mini-instruct config + backend)
evaluator: unchanged                  (meta-llama/Llama-3.2-3B-Instruct config + backend)
frozen results: unchanged             (see §8 hashes)
frozen configuration: unchanged       (config_hash still 61ea203c20e68c51)
```

The frozen JSON parser is **authoritative**: the GLM adapter calls the same
`json.loads → _extract_json_block → raw string` chain the Qwen backend uses, so a
malformed GLM reply fails exactly as a malformed Qwen reply does.

## 3. Backend architecture

```text
--attacker-model qwen3      (or no flag → config verbatim)
    ↓  apply_attacker_profile
models.attacker { backend: hf_local, residency: pinned }
    ↓  provider_factory.build_role_llm
HFLocalChatLLM  →  pipeline("text-generation")  →  AutoModelForCausalLM
    ↓  ManagedLocalChatLLM (ModelManager)
existing attack pipeline  (unchanged)

--attacker-model glm46v
    ↓  apply_attacker_profile
models.attacker { backend: glm4v, model: zai-org/GLM-4.6V-Flash,
                  revision: 411bb4d77144a3f03accbf4b780f5acb8b7cde4e,
                  dtype: bfloat16, quantization: none, residency: sequential }
    ↓  provider_factory.build_role_llm
GLM4VChatLLM  →  AutoProcessor.apply_chat_template(...)
              →  Glm4vForConditionalGeneration.generate(...)
              →  processor.batch_decode(...)
    ↓  ManagedLocalChatLLM (ModelManager)
existing attack pipeline  (unchanged — the attack layer never sees GLM)
```

Message conversion keeps the attack's canonical representation untouched: the
adapter translates the existing `[{role, content}]` list through
`processor.apply_chat_template(..., add_generation_prompt=True, tokenize=True,
return_dict=True, return_tensors="pt")`. Verified: the processor and the tokenizer
render the *identical* prompt (same text, 1538 tokens) — the adapter adds no
conversation format of its own.

## 4. GPU architecture (sequential residency)

```text
run boundary:  release_sequential()  -> nothing ~20.6 GB stays resident between runs

attacker turn: load GLM (20.586 GB) -> generate -> released when the target needs the GPU
target turn:   load Phi-4 (7.67 GB) -> generate -> released when the attacker needs the GPU
evaluator turn:load Llama-3.2-3B    -> generate -> released ...
```

`residency: sequential` marks the attacker releasable; the pre-existing eviction
path does the work. **No guard was weakened**: the pre-load gate, post-load
verification, per-run floor and 22 GB reserved ceiling all still run on every
load, and no quantization/offloading was introduced.

Measured (switching stage, real loads, `torch.cuda.memory_allocated()`):

| step | allocated after load | after unload |
|---|---:|---:|
| Qwen (pinned) | 8.045 GB | 0.0 GB |
| GLM (sequential) | **20.586 GB** | 0.0 GB |
| Qwen again | 8.045 GB | 0.0 GB |

Stage C confirmed it in the real pipeline: **max `resident_count` = 1**, 0 load
refusals, peak VRAM 21.987 GB.

## 5. Tests

```text
existing tests        : 615 passed before this work
full suite now        : 655 passed, 0 failed  (615 existing + 40 new)
new tests             : tests/test_glm_attacker_backend.py -> 40 passed
Qwen regression       : PASS (load 17.5 s, generation, JSON parse, multi-turn,
                        unload; factory log: backend=hf_local, revision=None,
                        residency=pinned — the legacy path, unchanged)
GLM qualification     : Stage A 5/10 valid JSON; Stage B 5/5 sequences, 0 context
                        failures; Stage C 9/9 runs, exit 0
model switching       : PASS both directions, no stale identity, no VRAM leak
VRAM / residency      : PASS (max 1 resident; no OOM; 0 refusals)
```

## 6. Qualification results (evidence in `results/phase15_glm46v/`)

| Stage | Result |
|---|---|
| **A** — 10 isolated generations | 5/10 valid JSON (50.0 %), 5 malformed, 0 empty, 0 reasoning-only, mean latency 13.4 s |
| **B** — 5 multi-turn sequences (15 turns) | 0 context failures, ordering correct, goal preserved, 0 empty queries, 2 repeated queries |
| **C** — 3 goals × 3 attacks, NBF OFF | 9/9 records, exit 0; 3 `JSON_PARSE_ERROR` + 6 `ATTACK_FAILURE`, 0 successes; `attacker_runtime` correct on all 9; NBF-OFF contamination 0 |
| **Qwen regression** | PASS |
| **Switching** | PASS |

## 7. Findings (all measured; nothing papered over)

1. **`enable_thinking` is load-bearing for GLM.** The model's chat template
   implements it (emits `/nothink`). With thinking left on, GLM answers the
   instruction block instead of the requested schema — measured **0/8**
   correct-schema replies versus **8/8** with it disabled. The frozen attacker
   block already declares `chat_template_kwargs: {enable_thinking: false}`, so the
   GLM profile carries the same declaration. This is a configuration value, not
   prompt/parser/template surgery.
2. **GLM's JSON fidelity is ~50 %.** Failures are a systematic malformed-JSON
   defect (an unquoted `lastResponseSummary` key / stray quote). Every failure is
   classified `JSON_PARSE_ERROR` with the raw output preserved. **No repair, no
   retry, no second generation, no fallback** was added — verified by test.
3. **No reasoning-wrapper handling.** With constrained JSON the model never emits
   `<think>`; unconstrained it does, and the frozen interface passes it through
   verbatim. A test pins that the adapter does not strip it.
4. **Stage C trajectories are very short** (turns 0–1 for 6 of 9 runs). This is a
   model-behaviour observation from the qualification, **not** an ASR result: the
   3-goal Stage C is an interface check, and no 90-run pilot was launched.
5. **Pre-existing runner bug, reported not fixed.** `run_batch` prints raw
   exception text; on a cp1252 console any non-encodable character (GLM's output
   contained `→`) raises `UnicodeEncodeError` *inside the failure-reporting path*
   and aborts the batch. Worked around at the environment level in the
   qualification harness (`PYTHONIOENCODING=utf-8` + UTF-8 subprocess decode)
   rather than by editing frozen runner logging.
6. **`--out` added for isolation.** The runner previously had a hardcoded output
   path, so any GLM run would have appended to `results/phase14/batch00.jsonl`
   (Track A's file). `--out` defaults to the historical path, so Track A behaviour
   is unchanged while a separate experiment stays separate.

## 8. Experimental status

```text
Track A results   : untouched  (results/phase14/batch00.jsonl
                                 = results/phase15_pilot_30/trackA_frozen/…180records.jsonl
                                 = FCB952F1598B697B033D605B0791CFBB…)
Track A config    : untouched  (config_hash 61ea203c20e68c51, unchanged)
GLM evidence      : results/phase15_glm46v/ (qualification/, stage_c/)
90-run pilot      : NOT LAUNCHED
```

Run the qualification yourself with:

```bash
python scripts/qualify_glm_attacker.py --stage qwen        # Qwen regression
python scripts/qualify_glm_attacker.py --stage switching   # Qwen -> GLM -> Qwen
python scripts/qualify_glm_attacker.py --stage A
python scripts/qualify_glm_attacker.py --stage B
python scripts/qualify_glm_attacker.py --stage C           # gated on A and B

# the switchable experiment itself (example; not run here)
python scripts/phase14_full_reproduction.py --batch 0 --limit 30 --attack all \
    --condition both --config configs/attacker_glm46v.yaml \
    --attacker-model glm46v --structured-output-mode constrained_json \
    --out results/phase15_glm46v/batch00.jsonl
```

```text
IMPLEMENTATION COMPLETE — GLM QUALIFICATION READY — 90-RUN PILOT NOT LAUNCHED
```
