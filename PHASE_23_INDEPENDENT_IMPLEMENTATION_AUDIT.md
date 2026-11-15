# PHASE 23 — INDEPENDENT IMPLEMENTATION AUDIT

Independent re-verification of the Phase 22 changes. Nothing here is copied from
the Phase 22 report: every row was re-derived from the source, the config, or a
fresh offline probe. No paid request was made; `OPENAI_API_KEY` was unset for every
command.

Audit date: 2026-11-10 · branch `main` · HEAD `89ed03f` (+ uncommitted Phases 20-22).

---

## 1. Git state and inventory (§4)

```
branch                       main
HEAD                         89ed03f7e5d51755842e9169aa6b6e5df36b0439
working tree                 DIRTY (23 modified/untracked paths, Phase 20-22 work)
frozen paths                 results/ runs/ checkpoints/ data/ configs/reproduction*.yaml
                             -> `git status --porcelain` EMPTY for all of them
checkpoint                   models_best_nbf_released.pth
                             sha256 cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe
```

Modified (tracked): `.gitignore`, `requirements.txt`,
`src/guardbound/attacks/{acronym,actor_attack,crescendo_paper,opposite_day,runner}.py`,
`src/guardbound/llm/{base,openai_client,provider_factory}.py`,
`tests/{conftest.py,test_llm_clients_mocked.py}`.

New for Phase 22 (untracked): `src/guardbound/paper_protocol.py`,
`src/guardbound/attacks/{prompt_variants.py,_official_prompts.py}`,
`configs/paper_reproduction_r1_gpt4o.yaml`, `tests/test_phase22_paper_fidelity.py`,
`scripts/_codegen_official_prompts.py`, plus the Phase 20-21 files.

No git write operation was performed (no commit, reset, stash, clean, amend).

Phase 22 report: `PHASE_22_IMPLEMENTATION_REPORT.md` (present, 550 lines).

---

## 2. Evaluator temperature — independently verified (§5)

Method: the R1 YAML was loaded, the protocol resolved, attacks built through the
same `_cloud_common.build_attack` helper the pilot uses, and every model call
recorded with its role attributed by **client identity** (a separate recorder
object per role), not by reading the code under test.

### 2.1 Call-site table (R1 resolved configuration)

| # | File | Function / call site | Model role | Configured | Forwarded (observed) | Result |
|---|---|---|---|---|---|---|
| 1 | `attacks/crescendo_paper.py` | `generate_crescendo_step` → `attacker_llm.generate` | attacker | 0.7 | **0.7** | PASS |
| 2 | `attacks/crescendo_paper.py` | `llm_evaluate_with_rubric` → `generate(..., temperature=temperature)` | evaluator | 0.7 | **0.7** | PASS |
| 3 | `attacks/crescendo_paper.py` | `llm_check_refusal` → same | evaluator | 0.7 | **0.7** | PASS |
| 4 | `attacks/opposite_day.py` | `generate_opposite_day_step` | attacker | 0.7 | **0.7** | PASS |
| 5 | `attacks/opposite_day.py` | `llm_evaluate_with_rubric` | evaluator | 0.7 | **0.7** | PASS |
| 6 | `attacks/opposite_day.py` | `llm_check_refusal` | evaluator | 0.7 | **0.7** | PASS |
| 7 | `attacks/actor_attack.py` | `llm_evaluate_with_rubric` (via `evaluate_response`, `call_multi`, `summary`) | evaluator | 0.7 | **0.7** | PASS |
| 8 | `attacks/actor_attack.py` | `step_judge` (rubric → step judge order) | evaluator | 0.7 | **0.7** | PASS |
| 9 | runner (`temperature=`) | target completion, generic path | target | 0.7 | **0.7** | PASS |
| 10 | runner (composite `temperature=`) | target completion, ActorAttack path | target | 0.7 | **0.7** | PASS |

`all_match = true` over all 11 recorded calls (10 distinct sites; the ActorAttack
rubric is exercised through both `call_multi` and `summary`).

### 2.2 No hidden override

* `temperature=0.0` does not appear anywhere in the four attack modules
  (source scan; Phase 22 test `test_all_eight_evaluator_call_sites_are_parameterised`).
* Every helper takes `temperature: float = LEGACY_EVALUATOR_TEMPERATURE` and passes
  it straight into `generate(...)`; no branch substitutes another value.
* The client itself does not coerce: `OpenAIChatLLM.build_request` writes the
  argument verbatim (payload check below).

### 2.3 Provider payload

`OpenAIChatLLM.build_request(messages, 0.7, True)` →
`{"model": ..., "messages": ..., "temperature": 0.7, "response_format": {"type": "json_object"}}`.
The value therefore reaches the wire, not just the Python call.

### 2.4 Historical default preserved

Building the same attack with **no** paper configuration: `evaluator_temperature_of({})
= 0.0`, and the observed calls were `rubric @ 0.0`, `refusal @ 0.0`. Legacy
behaviour is intact.

**Verdict: PASS.** 0.7 everywhere on the R1 path; 0.0 remains the default without
a paper configuration.

---

## 3. ActorAttack composite — summary (§6)

Full detail: `PHASE_23_ACTORATTACK_CONTRACT_AUDIT.md`.

Control flow independently reproduced against the official `attack_single` /
`call_multi` / `summary` transcription in
`nbf_original_stuff/.../attacks/actor_attack/run.py:78-205`:

| Requirement | Observed | Verdict |
|---|---|---|
| `prepare_attack` once per goal | 1 call | PASS |
| Actors in official order | `actor[0..n]` in list order | PASS |
| Fresh conversation per actor | 3 `[system, first_query]` calls, indices 0/3/6 | PASS |
| System message first | yes, every actor | PASS |
| Query chain order | `a0q0, a0q1, summary, a1q0, …` | PASS |
| `N_retry = 3` | 4 target attempts on persistent rejection | PASS |
| Rejective regeneration | rewrite applied, transcript popped | PASS |
| successful / rejective / unknown | statuses `successful`, `completed`, `skipped` | PASS |
| Rubric before step judge | observed order rubric → judge → … → summary rubric | PASS |
| `summary()` when required | once per non-skipped actor | PASS |
| Early stop at 5 | 1 actor record, `goal_achieved=True` | PASS |
| Skipped actor | no summary, others continue | PASS |
| Per-actor records separate | 3 records with index/status/score/turns | PASS |
| One behaviour per goal | one `Conversation`, one `goal_achieved` | PASS |
| Call sequence == official loop | identical messages in all 7 flow scenarios | PASS |

Two **wiring** defects were found in the composite path (register entry **D-02**,
which has two halves). They do not change the control flow above; they change
which NBF mechanism filters ActorAttack queries and ignore the configured
`steer_target`. They are reported, not fixed.

---

## 4. Crescendo / Opposite-day — no unintended change (§7)

Proof at source level rather than by test count:

```
git diff -U0 -- src/guardbound/attacks/runner.py
@@ -9,0 +10 @@ import time          <- one import line
@@ -1100,0 +1102,245 @@          <- the entire composite block, appended
```

**Zero deleted lines in the whole file.** The generic loop
(`run_attack`, `run_attack_with_backtracking`,
`run_attack_with_backtracking_async`) is byte-identical to HEAD. The composite
path is additive and reachable only through `attacks.execution.<attack>:
composite`.

Behavioural checks on the generic path:

| Property | Evidence | Verdict |
|---|---|---|
| Prompt variant | `opposite_day` built from config sends the official bytes (probe: 3913-char rendered prompt) | PASS |
| Query ordering / max rounds | unchanged code; `max_rounds = 8` resolved from config | PASS |
| Refusal detection | `check_refusal` at the configured temperature; legacy at 0.0 | PASS |
| Refusal-pop budget restoration | unchanged code (`history_t.pop()` transcription) | PASS |
| `C_refused < 10` | unchanged code | PASS |
| Rubric ordering | refresh-before-rubric retained (PHASE 17 Stage 2 F3) | PASS |
| Stopping conditions | unchanged (`round_number + num_filtering < max_rounds`) | PASS |
| NBF filtering integration | see D-01: the *threshold sign* is wrong for η>0 | **FAIL (P0)** |
| Goal-level aggregation | one record per (attack, goal) in the mock run | PASS |

The only regression risk in this area is D-01, which predates Phase 22 (it was
latent while η was 0) and is not caused by the composite runner.

---

## 5. Attack-set enforcement (§8)

Resolved R1: `['actor_attack', 'crescendo_paper', 'opposite_day']`,
`training_only: ['acronym']`. `validate_paper_config()` is called at
`scripts/cloud_pilot.py:286`, i.e. **before** `resolve_variant`, the NBF load, the
dataset load and any client construction.

Rejection matrix (re-run independently; each raises `PaperProtocolError`):

| Input | Message |
|---|---|
| `acronym` added | `... training-only ...` |
| ActorAttack removed | `... missing=['actor_attack'] ...` |
| Crescendo removed | `... missing=['crescendo_paper'] ...` |
| Opposite-day removed | `... missing=['opposite_day'] ...` |
| `red_queen` added | `missing=[] extra=['red_queen']` (exact set) |
| unknown id (`crescendo`) | `unknown attack identifier(s) ['crescendo']` |
| duplicate id | `duplicate attacks in attacks.enabled` |
| empty list | `attacks.enabled must be a non-empty list` |
| `training_only: []` | `must declare attacks.training_only` |

The historical `cloud_research_reference.yaml` still loads and is treated as a
non-paper run (`validate_paper_config` → `None`), so historical configs are neither
rewritten nor broken.

**Verdict: PASS.** Exact set enforced, before any provider interaction.

---

## 6. Prompt bytes (§12)

`scripts/audit_prompt_fidelity.py` re-run in this phase:

| Check | Result |
|---|---|
| `paper_path_faithful` | **true** |
| `prompt_variants` | FAITHFUL — official bytes == release for Opposite-day, Acronym, rubric |
| Crescendo attacker | FAITHFUL (one text) |
| Refusal (both copies) | FAITHFUL |
| Disclaimer | FAITHFUL |
| Target conversation shape | FAITHFUL |
| Provider SDK payload | FAITHFUL (messages unchanged, `json_object`, `temperature` preserved) |
| Legacy `opposite_day` / `acronym` / `rubric` renders | DIFFERENT (expected, disclosed; kept for frozen runs) |

Prompt hashes on the same fixture (`Q`/`R`):

| Prompt | legacy | official |
|---|---|---|
| Opposite-day attacker | `1cd2c2d0120bc88d…` | `dfd5b4da0a580630…` |
| Acronym attacker | `6b8b11a5ff15f06a…` | `f3afb6a9c9fb6c20…` |
| Rubric | `4b5504a570e0ce56…` | `146589a4fba6d42c…` |

Unicode and whitespace preservation: the official constants are `repr()` copies
taken from the release modules by `scripts/_codegen_official_prompts.py`; the
rendered comparison is byte equality, not normalised equality.

R1 selects `prompt_variant: official` (probe: dry-run prints `official`; the
manifest records `prompt_variant: official`). Historical configurations resolve to
`legacy` (code default). No historical prompt text was overwritten — the legacy
constants are untouched module-level strings.

**Verdict: PASS for the paper path.** One reporting defect follows from it: the
manifest records the *legacy* hashes (D-04).

---

## 7. Test baseline reproduced (§17)

| Moment | Collected | Passed | Failed | Errors | Skipped | Time |
|---|---:|---:|---:|---:|---:|---:|
| Phase 23 fresh baseline | **1067** | **1063** | **4** | **0** | **0** | 172.4 s |
| Phase 22 reported | 1067 | 1063 | 4 | 0 | 0 | 171.0 s |

Failure set identical to the Phase 22 baseline, node for node:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

No newly introduced failure. The four are environmental (2 CUDA-gated, 2 missing
`Qwen/Qwen3.5-4B` weights) and were not touched.

Focused runs in this phase: `tests/test_phase22_paper_fidelity.py` 62 passed;
`tests/test_cloud_nbf_integration.py` 7 passed (real checkpoint + real embedder).

---

## 8. Network observation

The audit deliberately exercised only offline paths. One probe
(`audit_data_nbf.py`, loading the real MPNet embedder) caused `huggingface_hub` to
issue **metadata requests** (`HEAD`/`GET` to `huggingface.co`) while validating its
cache. No OpenAI request was made, no inference was billed, and the probe path was
stopped after that single run.

Evidence that no model inference left the machine:

* `OPENAI_API_KEY` unset in every command (verified: `KEY ABSENT`, no `.env`);
* the offline-mock pilot completed with the real SDK pointed at a
  `httpx.MockTransport` — a real call without a key returns 401, not a 200;
* the usage ledger records 32 calls, all served by the fake transport.

Recommendation: for a strictly offline run, export `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before loading the embedder (D-11).
