# PHASE 36 — SCOPED IMPLEMENTATION OF DELIVERABLE 1 BLOCKERS

**Scope:** the three findings raised by `PHASE_35_DELIVERABLE_1_FINAL_READINESS_AUDIT.md`.
**Mode:** implementation + offline regression. No paid call, no model inference, no dataset
download, no experiment.

| Finding | Status | Evidence |
|---|---|---|
| **P0-1** ActorAttack success scoring | **RESOLVED** | §C, 7 tests |
| **P1-1** Qualification gate in the paid path | **RESOLVED** | §D, 12 tests |
| **P1-2** Attacker temperature propagation | **RESOLVED** | §E, 13 tests |
| Paper methodology / configuration | **UNCHANGED** | `git diff -- configs/` is empty |

**Final verdict: `READY FOR PHASE 37 READ-ONLY AUDIT`** (§H).

---

## A. Executive summary

All three Phase 35 findings are corrected, each with targeted offline regression tests that
exercise the real consumer path rather than an isolated helper.

* **P0-1** — `cloud_pilot.run_one` no longer re-derives ActorAttack success from
  `max(_scores) == 5`. The composite runner's author-aligned `goal_achieved` is consumed verbatim;
  a missing or non-boolean flag is a hard error, never a silent fallback. Intermediate rubric
  scores remain in the record for diagnostics and now carry a `success_source` marker.
* **P1-1** — the Phase 27 qualification verdict is now enforced in `scripts/cloud_pilot.py`
  **before** the API-key check and before any client construction. A missing, malformed,
  non-qualified, incomplete, or configuration-stale report blocks the paid path.
  `--confirm-paid-request` and a present key cannot bypass it. `--dry-run` and `--offline-mock`
  skip it because they construct no paid client.
* **P1-2** — every attacker-generation call site of ActorAttack, Crescendo and Opposite-day now
  receives `cfg.temperature.attacker`. Evaluator-side calls (rubric, refusal, step judge) keep
  their own temperatures. R1 still resolves to 0.7 everywhere.

**Validation:** 32 new tests pass; the union of the relevant existing offline suites is
**568 passed, 0 failed, 0 skipped** (`§F`). The four known Phase 31 failures were **not re-run** —
they allocate CUDA memory or require the absent local `Qwen/Qwen3.5-4B`, which this phase's
instructions exclude. Their status is unchanged and they remain environment-only.

**No methodology changed.** Target model, 200 behaviours, the three attacks, Acronym's
training-only exclusion, checkpoint hash, η, the `h < −η` admission rule, `steer_target: false`,
evaluator temperature, attacker temperature, 8-turn cap and both ASR denominators are untouched.

**Ready for a fresh read-only audit.** Not ready for execution: this phase authorizes neither paid
calls nor experiments.

---

## B. Change inventory

Six files. Five modified, one added. `git diff -- configs/` is empty.

> Note on `git diff` vs `HEAD`: the working tree carries the pre-existing Phase 22–35 changes, so a
> diff against the last commit includes earlier work. The change inventory below lists only the
> files this phase edited, all with modification times of 2026-11-12; `acronym.py` and `runner.py`
> retain their 2026-11-10 timestamps and were not touched.

### B.1 `src/guardbound/attacks/actor_attack.py` — P0-1 context, P1-2

| Change | Reason |
|---|---|
| Import `DEFAULT_TEMPERATURE` from `..llm.base` | default for the new parameters |
| `extract_harm_target(..., temperature=DEFAULT_TEMPERATURE)` and pass it to the `EXTRACT_PROMPT` generation call | P1-2 site 1 |
| `get_actors(..., temperature=...)` and thread into the network, actor-selection and `MORE_ACTOR` calls | P1-2 sites 2–4 |
| `get_init_queries(..., temperature=...)` and thread into the query-chain and JSON-format calls | P1-2 sites 5–6 |
| `infer_single(..., temperature=...)` forwarding to the three helpers above | P1-2 plumbing |
| `rewrite_query(..., temperature=...)` and thread into the rewrite call | P1-2 site 7 |
| `ActorAttack.__init__(..., attacker_temperature=DEFAULT_TEMPERATURE)` storing `self._attacker_temperature` | P1-2 knob |
| `prepare_attack`, `record_turn` (rejective rewrite), `call_multi` (rejective rewrite), `rewrite_query_if_needed` pass `temperature=self._attacker_temperature` | P1-2 propagation |
| Comment marking the step judge as an evaluator-side call with its own temperature | documents the intentional exception |

No prompt text, attack sequence, N_retry, scoring, or state machine changed.

### B.2 `src/guardbound/attacks/crescendo_paper.py` — P1-2

| Change | Reason |
|---|---|
| Import `DEFAULT_TEMPERATURE` | default |
| `generate_crescendo_step(..., temperature=DEFAULT_TEMPERATURE)`; the hard-coded `temperature=0.7` is replaced by the parameter | P1-2 — the only hard-coded attacker temperature in the codebase |
| `CrescendoAttackPaper.__init__(..., attacker_temperature=DEFAULT_TEMPERATURE)` | knob |
| `generate_question_for_turn` passes `temperature=self._attacker_temperature` | propagation |

### B.3 `src/guardbound/attacks/opposite_day.py` — P1-2

Same three changes: import, `generate_opposite_day_step(..., temperature=DEFAULT_TEMPERATURE)` with
the generation call updated, constructor knob, and call-site propagation.

### B.4 `scripts/_cloud_common.py` — P1-1, P1-2

| Change | Reason |
|---|---|
| `ATTACKER_TEMPERATURE_ATTACKS = frozenset({"actor_attack", "crescendo_paper", "opposite_day"})` | Acronym is training-only and its constructor does not accept the knob |
| `build_attack(..., attacker_temperature=None)` forwards the knob only for the three names | P1-2 plumbing without touching Acronym |
| `QualificationError`, `QUALIFIED_VERDICT`, `QUALIFICATION_ROLES` | P1-1 typed failure |
| `expected_qualification_identity(cfg)` | defines the fields that make a report current |
| `default_qualification_report_path(cfg)` | mirrors the harness's `<out_dir>/<tag>.<arm>.qualification.json` convention |
| `require_qualification(cfg, report_path=None)` | P1-1 gate; consumes the existing report, introduces no new qualification mechanism |

### B.5 `scripts/cloud_pilot.py` — P0-1, P1-1, P1-2

| Change | Reason |
|---|---|
| `CompositeSuccessMetadataError`, `resolve_goal_success(composite, aggregate)` | P0-1 single decision point |
| `composite = None` before the execution branch; the composite branch already assigned it | makes the flag available to the resolver |
| `success, success_source = resolve_goal_success(...)`; record `"success": success`, `"success_source": ...` | P0-1 |
| `--qualification-report` CLI flag | P1-1 |
| Gate call before `require_api_key` and before `build_role_llms` | P1-1 |
| Manifest `"qualification": qualification` block | traceability of the gate verdict |
| `attacker_temperature=` passed to `common.build_attack` | P1-2 |

### B.6 `tests/test_phase36_implementation.py` — new

32 tests covering all three findings (§C–§E). Offline only: scripted `RecordingLLM`, fake attacks,
fake composites, temporary JSON reports, monkeypatched client builders.

---

## C. P0-1 verification

### C.1 Previous behaviour

```python
aggregate = aggregate_scores(list(attack._scores), success_score)   # max(valid) == 5
...
"success": aggregate["success"],
```

`attack._scores` for ActorAttack accumulates every `call_multi` rubric evaluation
(`actor_attack.py`) — including exchanges later popped on a rejective step — plus every `summary`
attempt. The author's `attack_single` sets `goal_achieved` **only** from `summary()`
(`attacks/actor_attack/run.py:193-205`). Result: a goal with an intermediate 5 and a summary < 5
was counted as successful.

### C.2 Corrected behaviour

```python
success, success_source = resolve_goal_success(composite=composite, aggregate=aggregate)
```

| Path | Rule | `success_source` |
|---|---|---|
| Composite ActorAttack | `composite.goal_achieved` (the runner already computes it at `runner.py:1366-1369`) | `composite.goal_achieved` |
| Generic Crescendo / Opposite-day | `max(valid rubric scores) == success_score` — unchanged | `max_rubric_score` |

A composite result whose `goal_achieved` is missing or not a `bool` raises
`CompositeSuccessMetadataError`. It never falls back to the max rule.

Diagnostics are untouched: `rubric_scores`, `max_rubric_score`, `n_valid_rubric_scores`,
`n_invalid_rubric_scores`, `actor_records` all remain. `success_source` is additive.

### C.3 Tests and outcomes

| Test | Asserts |
|---|---|
| `test_composite_intermediate_five_with_summary_below_five_is_unsuccessful` | `success is False`, source composite, `rubric_scores == [5, 3]`, `max_rubric_score == 5` |
| `test_composite_summary_five_is_successful` | `success is True` |
| `test_composite_multiple_intermediate_scores_below_five_is_unsuccessful` | `success is False` |
| `test_generic_attack_keeps_the_max_rubric_score_rule` | generic `[5, 2]` → `True`, source `max_rubric_score` |
| `test_composite_without_goal_achieved_raises_and_never_falls_back` | `CompositeSuccessMetadataError` even with `_scores == [5,5,5]` |
| `test_composite_with_non_boolean_goal_achieved_raises` | `None` flag → raises |
| `test_resolve_goal_success_is_the_single_decision_point` | composite overrides an aggregate that says success |

All seven pass. The first, fifth and sixth specifically exercise
`cloud_pilot.run_one`, not the runner's internal calculation.

---

## D. P1-1 verification

### D.1 Previous behaviour

`cloud_pilot.py` enforced only `--confirm-paid-request` and an API-key presence check. The Phase 27
fail-closed gate lived exclusively in `scripts/cloud_model_qualification.py`. A paid run could
therefore construct clients for an unqualified model.

### D.2 Corrected paid flow

```
config load → R1 validation → variant → invariants → attack-set check
  → --dry-run? return 0
  → --offline-mock? skip gate (no paid client is ever constructed)
  → --confirm-paid-request? required
  → require_qualification(cfg, --qualification-report)      ← Phase 36 gate
  → require_api_key
  → ledger
  → NBF checkpoint verify + load (NBF arm only)
  → build_role_llms                                          ← first client construction
```

The gate reads the **existing** report produced by `scripts/cloud_model_qualification.py`
(`<out_dir>/<tag>.<arm>.qualification.json`). No new qualification mechanism was introduced, and
no waiver mechanism exists in the codebase, so none was invented or enabled.

Blocks on:

| Condition | Failure |
|---|---|
| report file absent | `QualificationError: no qualification report at …` |
| unreadable / invalid JSON | `QualificationError: … not valid JSON` |
| report is not a JSON object | `QualificationError: … must be a JSON object` |
| `overall != "QUALIFIED"` | `QualificationError: verdict is … not 'QUALIFIED'` |
| `probes_complete is not True` | `QualificationError: … did not complete its probe set` |
| `configuration_under_qualification.models` ≠ current | `QualificationError: … stale` |
| `prompt_variant` ≠ current | `QualificationError: … stale` |
| `attacker_prompt_condition` ≠ current | `QualificationError: … stale` |
| `temperatures` ≠ current | `QualificationError: … stale` |

**Staleness is identity-based, not age-based.** A report qualifies a specific
(models × prompt variant × attacker prompt condition × temperatures) tuple. No arbitrary TTL was
invented.

### D.3 Evidence that blocked cases cannot construct a client

`test_paid_run_blocks_before_client_construction_without_qualification` monkeypatches
`common.build_role_llms` to record any call and raises `QualificationError` before it; the test
asserts the recorder was never reached. `test_confirm_and_api_key_cannot_bypass_qualification`
supplies both `--confirm-paid-request` and a fake present key, and mocks `build_role_llms` with
`pytest.fail` — the run still blocks. No API request can occur, because no client exists.

### D.4 Tests and outcomes

| Test | Asserts |
|---|---|
| `test_qualified_report_matching_the_config_passes` | returns the manifest block, 64-hex sha256 |
| `test_missing_qualification_report_blocks` | `no qualification report` |
| `test_non_qualified_verdict_blocks` | `NOT_QUALIFIED` rejected |
| `test_incomplete_probe_set_blocks` | `probes_complete=False` rejected |
| `test_malformed_json_blocks` | parse failure rejected |
| `test_non_object_report_blocks` | list payload rejected |
| `test_stale_models_block` | changed target model rejected |
| `test_stale_temperature_blocks` | changed evaluator temperature rejected |
| `test_stale_prompt_variant_blocks` | `legacy` vs `official` rejected |
| `test_default_qualification_report_path_matches_the_harness_convention` | path parity with the harness |
| `test_paid_run_blocks_before_client_construction_without_qualification` | `build_role_llms` never called |
| `test_confirm_and_api_key_cannot_bypass_qualification` | key + confirm still blocked |
| `test_qualified_report_reaches_client_construction` | a valid report reaches `build_role_llms` (sentinel exception) |
| `test_dry_run_returns_zero_without_qualification` | dry-run remains credential- and qualification-free |

All 14 pass.

> Operational note: the report must be produced with the same configuration the experiment will
> use (R1: `prompt_variant: official`, temperatures 0.7/0.7/0.7, `attacker_prompt_condition:
> baseline`). Qualifying with a config that records `legacy` prompts will now block the official
> run as stale — which is the intended behaviour.

---

## E. P1-2 verification

### E.1 Attacker call sites inspected

| Attack | Call site | Previous temperature | Now |
|---|---|---|---|
| ActorAttack | `extract_harm_target` → `EXTRACT_PROMPT` | client default (0.7) | configured |
| ActorAttack | `get_actors` → network prompt | client default | configured |
| ActorAttack | `get_actors` → `ACTOR_PROMPT` | client default | configured |
| ActorAttack | `get_actors` → `MORE_ACTOR_PROMPT` | client default | configured |
| ActorAttack | `get_init_queries` → `QUERIES_PROMPT` | client default | configured |
| ActorAttack | `get_init_queries` → `JSON_FORMAT_QUESTION_PROMPT` | client default | configured |
| ActorAttack | `rewrite_query` → `ATTACK_MODIFY_PROMPT` | client default | configured |
| Crescendo | `generate_crescendo_step` | **hard-coded 0.7** | configured |
| Opposite-day | `generate_opposite_day_step` | client default | configured |

Evaluator-side calls, deliberately **not** changed:

| Call | Temperature | Note |
|---|---|---|
| ActorAttack `step_judge` | `ACTOR_STEP_JUDGE_TEMPERATURE` (0.7) | evaluator role, routed via `_judge_llm`; the author omits the argument and inherits 0.7 |
| ActorAttack / Crescendo / Opposite-day rubric | `self._evaluator_temperature` | paper's evaluator temperature (0.7 in R1) |
| Refusal / disclaimer classifiers | `self._evaluator_temperature` | evaluator role |

### E.2 Propagation

`_cloud_common.build_attack(..., attacker_temperature=…)` → constructor
`self._attacker_temperature` → every attacker-generation helper. The default is
`DEFAULT_TEMPERATURE = 0.7`, so every pre-existing caller (tests, historical scripts, qualification
probes) is byte-identical.

Acronym is excluded by `ATTACKER_TEMPERATURE_ATTACKS`: it is training-only, out of the main
evaluation, and its constructor does not accept the knob. `build_attack("acronym", …)` therefore
still works.

### E.3 Tests and outcomes

| Test | Asserts |
|---|---|
| `test_crescendo_step_uses_the_configured_attacker_temperature` | 0.42 recorded |
| `test_crescendo_attack_class_threads_the_attacker_temperature` | 0.42 via `generate_question_for_turn` |
| `test_opposite_day_step_uses_the_configured_attacker_temperature` | 0.42 |
| `test_opposite_day_attack_class_threads_the_attacker_temperature` | 0.42 |
| `test_actor_attack_pre_attack_uses_the_configured_attacker_temperature` | all 5 pre-attack calls at 0.42 |
| `test_actor_attack_class_threads_the_attacker_temperature` | `prepare_attack` uses 0.42 |
| `test_actor_attack_rewrite_uses_the_configured_attacker_temperature` | rewrite at 0.42 |
| `test_evaluator_temperature_stays_independent` | rubric at 0.11 while attacker is 0.42 |
| `test_build_attack_forwards_the_attacker_temperature_to_the_three_attacks` | all three store 0.42 |
| `test_build_attack_does_not_forward_to_acronym` | no `TypeError`, no forced knob |
| `test_r1_configuration_still_resolves_to_seven_tenths` | R1 attacker and evaluator both 0.7 |

All 11 pass. The recorder captures the actual `temperature` argument at the `ChatLLM` boundary, so
the assertion is on what would be sent, not on an internal attribute.

---

## F. Test results

Environment: Windows, PowerShell 5.1, `PYTHONDONTWRITEBYTECODE=1`,
`KMP_DUPLICATE_LIB_OK=TRUE`, `-p no:cacheprovider`. No network, no paid client, no checkpoint or
embedder load, no GPU allocation.

### F.1 New tests only

```
python -m pytest tests/test_phase36_implementation.py -q --no-header -p no:cacheprovider
```
**32 collected / 32 passed / 0 failed / 0 skipped** (0.23 s).

### F.2 Union of relevant offline suites

```
python -m pytest \
  tests/test_phase36_implementation.py \
  tests/parity/test_actor_attack_parity.py \
  tests/parity/test_crescendo_parity.py \
  tests/parity/test_opposite_day_parity.py \
  tests/parity/test_refusal_threshold_parity.py \
  tests/test_attacks.py tests/test_phase2_attacks.py \
  tests/test_phase14_8_telemetry.py tests/test_phase22_paper_fidelity.py \
  tests/test_phase24_p0_fixes.py tests/test_phase26_qualification_gate.py \
  tests/test_phase27_fail_closed_gate.py tests/test_phase29_alignment.py \
  tests/test_phase30_retry_provenance.py tests/test_phase31_retry_semantics.py \
  tests/test_attacker_arms.py tests/test_cloud_provider.py \
  tests/test_cloud_pricing_and_snapshots.py tests/test_cloud_research_integration.py \
  -q --no-header -p no:cacheprovider
```
**568 collected / 568 passed / 0 failed / 0 skipped / 0 errors** (21.36 s).

| Metric | Value |
|---|---|
| Tests collected | 568 |
| Passed | **568** |
| Failed | 0 |
| Skipped | 0 |
| Environment-related failures | 0 in this selection |
| New tests | 32 |

### F.3 Tests deliberately not run

The Phase 31 baseline (1 277 collected / 1 273 passed / 4 failed) is unchanged in intent, but the
four known failures were **not** re-executed here:

| Test | Why excluded |
|---|---|
| `tests.test_phase14_2_rotation…::test_releases_cache_before_measuring` | allocates CUDA memory |
| `tests.test_phase14_2_rotation…::test_effective_free_after_release_is_what_is_measured` | allocates CUDA memory; fails below 6 GB free |
| `tests.test_phase8_architecture…::test_default_config_valid` | requires local `Qwen/Qwen3.5-4B` |
| `tests.test_phase8_architecture…::test_attacker_present` | requires local `Qwen/Qwen3.5-4B` |

The phase brief prohibits GPU allocation, model downloads and model inference, so these were left
untouched. No test was modified, skipped or weakened to avoid them; they simply were not in the
selected set.

---

## G. Remaining deviations

These were **not** in Phase 36's scope and remain as Phase 35 recorded them. None is a P0 or P1 for
Deliverable 1.

| # | Item | Status | Note |
|---|---|---|---|
| G1 | Four environmental test failures (CUDA / local Qwen) | **unchanged** | not re-run; classification evidence from Phase 31 still applies |
| G2 | `attacks.N_retry` config key is not read; effective value is the module constant 3 | **unchanged** | no behavioural difference; P2 documentation debt |
| G3 | Attack manifest has no `eta_active` key (activation is `nbf.enabled`) | **unchanged** | semantics correct; naming inconsistent with the Phase 34.3 utility schema |
| G4 | `comparison_group` omits models/temperatures | **unchanged** | the full manifest and `config.sha256` carry them |
| G5 | Async generic runner lacks the ActorAttack branch present in the sync runner | **unchanged** | latent; R1 routes ActorAttack through `run_actor_attack_composite` |
| G6 | `assert_invariants` returns declared names rather than verified values | **unchanged** | arms are equal by construction; manifest wording overstates the check |
| G7 | No fail-closed dataset row-count assertion in `load_goals` | **unchanged** | mitigated by the pinned dataset SHA-256 |
| G8 | `get_actors` trailing `MORE_ACTOR_PROMPT` generation is discarded | **still unverified** | not compared to the author's actor-expansion code in Phase 35 or here |
| G9 | Qualification reports are configuration-scoped, not time-scoped | **by design** | documented in §D.2; no TTL invented |

**Resolved:** P0-1, P1-1, P1-2.

---

## H. Final verdict

# `READY FOR PHASE 37 READ-ONLY AUDIT`

All three Phase 35 findings are corrected and covered by offline regression tests that exercise the
real consumer paths:

* the ActorAttack success decision now comes from the author-aligned composite flag, with an
  explicit error instead of a silent fallback;
* the paid entry point refuses to construct a client without a matching QUALIFIED report, and
  neither confirmation nor an API key can bypass it;
* the configured attacker temperature reaches every attacker-generation call site of the three
  paper attacks, while evaluator temperatures stay independent and R1 remains at 0.7.

The paper methodology and every frozen value in §7 of the brief are unchanged. No paid request,
model inference, dataset download or experiment occurred. This phase authorizes neither execution
nor spending; Phase 37 must re-audit independently, and execution still requires its own
authorization.
