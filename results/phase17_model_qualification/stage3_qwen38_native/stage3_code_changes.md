# PHASE 17 — STAGE 3: Code changes

Scope of this stage's code changes: **the model backend/loader only**. No attack
algorithm, attack prompt, NBF component, rubric, judge, target, success criterion
or parser semantic was touched. The changes are additive and opt-in; every
existing caller keeps its previous defaults, which is pinned by tests.

Blast radius: **6 files** — 4 production (`local_client` untouched; three changed,
one new) and 2 test files (one changed, one new) — plus the Stage 3 scripts. No
change to `results/phase17_model_qualification/stage2_attacker/**`: the Stage 2
evidence is untouched, verified by mtime (every file there predates this stage)
and by the fact that the Stage 3 JSON harness redirects its raw-output directory
instead of writing into Stage 2's.

---

## 1. `src/guardbound/llm/multimodal_client.py` (changed)

| Field | Value |
|---|---|
| Change | Added opt-in `quantization` (`None` \| `"nf4"`), declared `top_p` / `do_sample`, `max_new_tokens: int \| None`, a `_context_generation_budget()` helper, threading of `quantization_config` into the model load, and per-call `last_generation_stats` telemetry |
| Reason | The approved quantization policy (official weights → 4-bit NF4 PTQ → bitsandbytes) had no expression on the native path; the attacker contract needs `max_new_tokens` UNSET and a declared `top_p`, and the architecture comparison needs the same per-call telemetry the text backend records |
| Experimental impact | **None while unset.** `quantization=None`, `top_p=None`, `do_sample=None` and `max_new_tokens=256` reproduce the previous behaviour exactly, so GLM-4.6V, Ornith-1.5-9B and every existing config are unaffected |
| Tests | `TestStage3AdditionsAreOptIn` (GLM and Ornith defaults unchanged), `TestNativeBackend` |

The one behavioural subtlety, and why it is safe: when `max_new_tokens` is
`None`, the budget is the model's own context window minus the prompt (the same
rule the text backend uses), not a new number. Nothing here truncates output that
would otherwise have continued.

## 2. `src/guardbound/llm/qwen38_native_client.py` (new)

| Field | Value |
|---|---|
| Change | `Qwen38NativeChatLLM` — resolves `AutoProcessor` + `AutoModelForImageTextToText` for `Qwen/Qwen3.8-27B`, pinned to revision `1d4bf0f2ff60…`, and asserts the instantiated class is the one the checkpoint declares (`Qwen3_5ForConditionalGeneration`), raising `Qwen38NativeArchitectureError` (`failure_class = MODEL_INTERFACE_INCOMPATIBLE`) otherwise |
| Reason | The text-generation path resolves `Qwen3_5ForCausalLM`, whose expected parameter names (`model.layers.*`) do not match the checkpoint's own (`model.language_model.*`, `model.visual.*`). The declared class is loadable, so the substitution is avoidable, not necessary |
| Experimental impact | None unless selected. Only the loader is model-specific; the pipeline (processor chat template, generation parameters, frozen JSON stopping criterion, frozen JSON contract) is the shared `MultimodalChatLLM`, so this is not a second attack path |
| Tests | `TestNativeBackend` — defaults are the pinned checkpoint; the declared class passes verification; the `Qwen3_5ForCausalLM` substitution is rejected with the typed error |

Not a duplicate of `Ornith15ChatLLM`: that backend exists for a different model
id and resolves `AutoModelForMultimodalLM`; both resolve their own class and
share the whole pipeline. Reusing the Ornith module would have meant a Qwen
backend whose error type and label name another model.

## 3. `src/guardbound/llm/attacker_profiles.py` (changed)

| Field | Value |
|---|---|
| Change | Added `BACKEND_QWEN38_NATIVE` and the `qwen38_native` profile (model, revision, backend, sequential residency, `quantization: nf4`, `chat_template_kwargs: {enable_thinking: False}`) |
| Reason | Model selection must be a runtime choice, not a source edit. The profile registry is the project's existing mechanism; the profile is data-only and leaves generation parameters (including `max_new_tokens: null`) to the experiment, exactly as the other profiles do |
| Experimental impact | None unless selected — the registry is only read when an attacker profile is requested |
| Tests | `TestAttackerProfile` — registration, selection, and that the caller's config is not mutated |

## 4. `src/guardbound/llm/provider_factory.py` (changed)

| Field | Value |
|---|---|
| Change | Added the `qwen38_native` / `qwen3_8_native` / `qwen38` backend aliases and a construction branch that forwards `quantization`, `top_p`, `do_sample` and `revision` to `Qwen38NativeChatLLM` |
| Reason | The backend must be reachable from config like every other backend; the attack layer stays unaware of which backend is in use |
| Experimental impact | None unless `backend: qwen38_native` appears in a role config. Unknown backend names still fail loudly |
| Tests | `TestProviderFactoryWiring` — builds through the manager, registers the role, applies residency, forwards the declared quantization; unknown alias still raises |

## 5. `tests/test_phase17_stage3_native_backend.py` (new, 14 tests)

Covers the two properties that make the change safe to land: the Stage 3
additions are opt-in (existing backends unchanged), and the native backend
refuses a class the checkpoint does not declare. Offline: no weights, no GPU.

## 6. `tests/test_ornith15_attacker_backend.py` (changed — one assertion)

| Field | Value |
|---|---|
| Change | The exact expected profile list became `["glm46v", "ornith15", "qwen3", "qwen38_native"]`, with a comment saying why |
| Reason | The test asserts the **complete** set of registered profiles; registering a profile legitimately changes that set. This is an intentional runtime-API extension, not a failure being accommodated |
| Experimental impact | None — the assertion keeps its exactness, so a future profile still has to be recorded here deliberately |
| Tests | It is itself the test |

**This is the only pre-existing test touched in this stage.** No production
behaviour was changed to make any test pass, and no test was weakened.

---

## Non-goals, verified unchanged

- Attack algorithms, prompts, history construction, refusal/backtracking
  semantics, round budgets — untouched (`git diff` covers no file under
  `attacks/`).
- NBF, its checkpoint, threshold and scoring — untouched.
- Rubric, judge prompts, judge model/revision/thinking mode, success criterion —
  untouched.
- Parser semantics: the native backend reuses `_extract_json_block` and
  `_JSONStoppingCriteria` **by import** from `local_client`; no second parser and
  no repair path exists.
- Target and target generation policy — untouched.
- Stage 2 evidence — not modified, not overwritten, not deleted (Part 23).

## Reuse rather than reimplementation

The JSON qualification for the native path calls the **Stage 2 harness itself**
(`phase17_stage2_attacker_qualification.part_h`) with the native client, and its
raw-output directory is redirected at runtime so Stage 2's file is never
appended to. That keeps the prompt, the call signature, the classifier and the
no-repair policy identical between the two paths by construction — the
comparison in `causallm_vs_native_comparison.json` cannot drift from the
baseline because it is produced by the same code.
