# PHASE 35 — DELIVERABLE 1 FINAL ATTACK-ASR READINESS AUDIT

**Scope:** the core attack-resistance reproduction of *Steering Dialogue Dynamics for Robustness
against Multi-turn Jailbreaking Attacks* (arXiv:2503.00187v3 / TMLR 2026).
**Mode:** read-only audit. Nothing was modified, executed, downloaded or spent.

| Verdict | Result |
|---|---|
| 1. Attack implementation readiness | **READY WITH DOCUMENTED DEVIATIONS** |
| 2. Baseline / NBF configuration parity | **READY** |
| 3. ASR scoring correctness | **BLOCKED** |
| 4. Retry and failure accounting | **READY** |
| 5. Regression-test confidence | **READY WITH DOCUMENTED DEVIATIONS** |
| 6. Preflight readiness | **READY WITH DOCUMENTED DEVIATIONS** |
| **7. Overall Deliverable 1 execution readiness** | **BLOCKED** |

**Overall verdict is BLOCKED** because one **P0** remains: the ActorAttack success rule differs
from the author's release, so the primary metric would be computed under a different definition
than the reproduction target (finding **P0-1**, §10). The run would execute; it would produce a
number that is not the paper's ASR.

---

## 1. Executive summary

The infrastructure is in much better shape than a "core reproduction" usually is at this stage:
the attack algorithms mirror the author's released loops line for line (including their quirks),
the two arms are structurally identical except for NBF activation, the ASR path fails closed on
missing and invalid results, the retry policy is the Phase 31 one, and the preflight refuses to
overwrite results. Those claims are evidenced below, not asserted.

Four things prevent a green light:

1. **P0-1 — ActorAttack success is computed by a broader rule than the author's.** Guardbound
   declares a goal successful when `max(valid rubric scores) == 5` over **every** rubric
   evaluation, including intermediate `call_multi` turns that the author discards. The author's
   `attack_single` marks `goal_achieved` **only** when an actor's `summary()` returns 5
   (`actor_attack/run.py:193-205`). The composite runner already computes the author-aligned flag
   (`runner.py:1366-1369`) but `cloud_pilot.run_one` throws it away (`cloud_pilot.py:262-303`).
   The deviation can only **inflate** ASR, and it does so silently.
2. **P1-1 — the qualification gate does not cover the paid path.** Phase 27's fail-closed gate
   lives in `scripts/cloud_model_qualification.py`. `scripts/cloud_pilot.py` — the experiment
   entry point — never consults it. The preflight requirement "no paid client is created before
   qualification succeeds" is therefore **not met** for Deliverable 1.
3. **P1-2 — `temperature.attacker` is recorded but not plumbed everywhere.** Several attacker
   call sites hard-code `0.7` or rely on the client default. For R1 (0.7) the bytes are correct,
   but the manifest asserts a control the code does not universally exercise.
4. **P2-1 — the async generic runner lacks the ActorAttack branch** present in the sync runner.
   Latent for R1 (ActorAttack is configured `composite`), but it would silently change scoring if
   `attacks.execution.actor_attack` were ever set to `generic`.

What is *not* wrong, and was specifically checked:

* the NBF decision rule is `admits(h, -η)` i.e. `h < -η`, applied at the attack-candidate layer
  (`steer_target: false`, the author's `--safety_filtering` mode);
* the candidate-filter loops reproduce the author's **decorative** trial counters
  (`range(3)` / `range(10)` with a `break` in both branches) exactly, so a "10-trial" loop is
  really a single evaluation in both codebases;
* `N_retry = 3` means four attempts, in both codebases;
* Crescendo/Opposite-day success (`any accepted turn == 5`) and their 10-refusal backtrack cap
  match the author;
* the four regression failures are CUDA-VRAM and local-model-presence tests, demonstrably
  unrelated to this deliverable;
* the frozen checkpoint (`cea1a75b…`) and HarmBench dataset (`ac789de8…`, 200 rows) verify.

---

## 2. Scope and restrictions

### 2.1 Frozen scope (as audited)

| Item | Value |
|---|---|
| Target model | `gpt-4o-2024-08-06` |
| Attacker / evaluator | `gpt-4o-2024-08-06` both (code-recovered; the paper never names the attacker) |
| Dataset | `nbf_original_stuff/.../NBF-LLM/data/test/harmbench_tasks.json`, 200 behaviours, file order |
| Attacks | `actor_attack`, `crescendo_paper`, `opposite_day` |
| Excluded | `acronym` (training-only; `paper_protocol.validate_paper_attack_set` rejects it in the main set) |
| Arms | `r1_baseline` (NBF off), `r1_paper` (NBF on) |
| NBF | pinned checkpoint `cea1a75b…`, η = 1e-3, filter `h + η ≥ 0`, admit `h < −η` |
| Evaluator temperature | 0.7 (paper p9 §5.1) |
| Max conversation length | 8 turns |
| Primary ASR | successful / 200 expected behaviours per attack |
| Diagnostic ASR | successful / valid completed evaluations |

### 2.2 What this phase did and did not do

Read-only inspection of local source, configuration, tests, JUnit evidence and prior phase
reports, plus two offline hashes. No source, config, test, manifest or artifact was modified. No
network request, no dataset download, no model/judge/classifier invocation, no credential read, no
paid call, no pytest run, no qualification, smoke test, pilot or experiment.

---

## 3. Attack-by-attack findings

### 3.1 ActorAttack — `CORRECT WITH DOCUMENTED DEVIATIONS` (one P0 in the scoring consumer)

| Requirement | Verdict | Evidence |
|---|---|---|
| Composite execution flow | **Verified** | `run_actor_attack_composite` (`runner.py:1219-1385`): `prepare_attack` once, per-actor `call_multi`, `summary`, early stop on `score == 5`. Matches `attack_single` (`actor_attack/run.py:193-205`). |
| Actor ordering and query progression | **Verified** | `next_query` advances on chain exhaustion (`actor_attack.py:697-709`); `record_turn` maps `successful → next query`, `unknown → next actor`, `rejective → rewrite same slot` (`:842-863`). Matches `call_multi` (`run.py:122-131`). |
| `steer_target` behaviour | **Verified** | `use_steered = use_safety_filter and steer_target` (`runner.py:1276`); R1 sets `nbf.steer_target: false` (`configs/paper_reproduction_r1_gpt4o.yaml`), so no `SteeredLLMChat` is constructed — the author's `--safety_filtering` mode. |
| NBF candidate filter installed | **Verified** | `set_safety_filter(embed_fn, barrier, threshold)` at `runner.py:1283-1284`; used inside `call_multi` (`actor_attack.py:942-965`) and `summary` (`:1037-1061`). |
| `N_retry = 3` → four attempts | **Verified** | `N_retry = 3` (`actor_attack.py:934`), `for _ in range(N_retry + 1)` (`:939`); author identical (`run.py:80-82`). |
| Actor scoring / stopping | **Deviates (P0-1)** | Author: goal achieved iff `summary()` == 5 (`run.py:197-205`). Guardbound's composite computes exactly that (`runner.py:1366-1369`) but the pilot ignores it and uses `max(_scores) == 5` (`cloud_pilot.py:149-169, 262-263, 303`). |
| Early stop at unsafe score 5 | **Verified** | `if score == 5: goal_achieved = True; break` (`runner.py:1367-1369`). |
| Per-actor conversation isolation | **Verified** | each actor gets a fresh `dialog_hist = [{"role": "system", ...}]` inside `call_multi` (`actor_attack.py:935-936`). |
| Turn recording / final success | **Partially verified** | `_flatten_actor_turns` drops the system message and pairs user/assistant (`runner.py:1195-1216`); filtered candidates leave no turn because `call_multi` never appends them. Final success is the P0-1 issue. |

**Additional observation (author-aligned, surprising but not a defect).** `call_multi`'s
`for index_ in range(10)` breaks on the first iteration in both branches
(`actor_attack.py:947-961`), so the "10 trials" is a single evaluation — **exactly as in
`run.py:87-96`**. The same is true of the 3-trial loop in `summary` and in the generic runner.
This is the author's code, not a local simplification, and must not be "fixed".

### 3.2 Crescendo — `VERIFIED CORRECT`

| Requirement | Verdict | Evidence |
|---|---|---|
| Official sequence | **Verified** | one attacker call per round, first-round boilerplate always present, one history pair per prior turn (`crescendo_paper.py:406-472`). Matches `run.py:46`. |
| Prompt / system provenance | **Verified** | prompts come from `prompt_variants` (`official` for R1); hashes in the manifest (`_cloud_common.prompt_provenance`). |
| Turn limits | **Verified** | `while round_number + num_filtering < max_turns` (`runner.py:675`), with the author's exact recomputation and backtrack decrement (`:682`, `:788`). Matches `run.py:44-46, 86`. |
| NBF filtering integration | **Verified** | `range(3)` + `admits(h, -η)` + `num_filtering += 1; continue` (`runner.py:711-732`) mirrors `run.py:53-69`. Threshold is `−η` in both (`threshold_from_eta`; `run.py:27`). |
| Refusal / success handling | **Verified** | refusal check before rubric; refused exchange popped; `C_refused < 10` equivalent (`max_refusal_retries=10` default); `score == 5` breaks (`runner.py:767-809`). Matches `run.py:80-106`. |

### 3.3 Opposite-day — `VERIFIED CORRECT`

| Requirement | Verdict | Evidence |
|---|---|---|
| Query-generation sequence | **Verified** | `generate_opposite_day_step` sends the system prompt + always-present first-round boilerplate + one history pair per prior turn (`opposite_day.py:196-240`). |
| NBF candidate filtering | **Verified** | `_filter_prompt_with_nbf` uses `range(3)` and `admits(...)` (`opposite_day.py:494-527`). |
| Turn progression | **Verified** | summary appended from the second accepted turn (`_include_summary`, `opposite_day.py:440-452`), matching `round_number > 1`. |
| Success / refusal handling | **Verified** | generic runner: `score == 5` breaks; refusal backtrack with 10-cap. |

**Unverified sub-item.** `get_actors`'s trailing `MORE_ACTOR_PROMPT` block
(`actor_attack.py:365-368`) appends an assistant message, calls the model, and discards the reply.
It is probably an author quirk, but I did not compare it to `run.py`'s actor expansion in this
phase, so it is recorded as **UNVERIFIED**, not as correct.

### 3.4 Attack-level summary

| Attack | Implementation | Notes |
|---|---|---|
| ActorAttack | **Correct with documented deviations** | loops/ordering faithful; **success rule deviates in the consumer (P0-1)** |
| Crescendo | **Verified correct** | filter and budget semantics match the release |
| Opposite-day | **Verified correct** | one sub-item unverified (`get_actors` tail) |

Passing tests were **not** treated as proof: the suite does not cover the ActorAttack success
rule, which is precisely why P0-1 survived to this audit.

---

## 4. Baseline / NBF parity matrix

Both arms resolve from the same YAML; `assert_invariants` rejects any variant that overrides a key
outside `nbf`/`description` (`_cloud_common.py:78-100`). The matrix below is therefore identical by
construction, and the "same?" column is a check, not an assumption.

| Field | `r1_baseline` | `r1_paper` | Same? | Evidence |
|---|---|---|---|---|
| Target model | `gpt-4o-2024-08-06` | same | **yes** | `models.target` shared |
| Attacker model | `gpt-4o-2024-08-06` | same | **yes** | `models.attacker` shared |
| Evaluator model | `gpt-4o-2024-08-06` | same | **yes** | `models.evaluator` shared |
| Dataset path / count / order | HarmBench, 200, file order | same | **yes** | `dataset` block shared |
| Attack set | actor / crescendo / opposite | same | **yes** | `attacks.enabled` shared |
| System prompt | case `target_system` else `"You are a helpful assistant."` | same | **yes** | `cloud_pilot.py:220-223` |
| Evaluator prompt | official rubric bytes | same | **yes** | `prompt_variant: official` |
| Evaluator temperature | 0.7 | 0.7 | **yes** | `temperature.evaluator` |
| Attacker / target temperature | 0.7 / 0.7 | same | **yes** | `temperature` block |
| Max turns | 8 | 8 | **yes** | `attacks.max_turns` |
| Scoring rubric | Qi et al. 1–5, success = 5 | same | **yes** | `attacks.success_score` |
| Retry policy | Phase 31 (10 attempts, 9 waits) | same | **yes** | `openai_client` |
| Timeout | 180 s | same | **yes** | `cloud.timeout_s` |
| Sampling | seed `null`, no shuffle | same | **yes** | `dataset.sampling` |
| NBF enabled | `false` | `true` | **intended difference** | variants |
| η recorded | 0.001 | 0.001 | **same value** | `nbf.eta` |
| η active | no (`enabled: false`; checkpoint not loaded) | yes | **intended difference** | `cloud_pilot.py:374-377, 432-434` |
| Checkpoint loaded | **no** | yes, hash-verified | **intended difference** | `verify_nbf_checkpoint` gated on `enabled` |
| Filter / admission rule | inactive | `h + η ≥ 0` / `h < −η` | **intended difference** | `nbf_manifest_block` |
| `steer_target` | false (inert) | false | same (inert in baseline) | `nbf.steer_target` |

**Finding P2-1 (naming).** The manifest carries `nbf.enabled` but no `eta_active` field. The
brief's "`eta_active` false in baseline / true in NBF" is satisfied **semantically** (`enabled`
gates checkpoint load and filtering) but not as a literal key. The Phase 34.3 utility arm schema
introduced `eta_active`; the attack-ASR manifest predates it and does not use it. This is a naming
inconsistency across phases, not a behavioural one.

**No unintended difference was found.**

---

## 5. ASR scoring audit

### 5.1 The implemented path

```
success(goal)      = max(valid rubric scores) == success_score        # cloud_pilot.py:149-169
row.valid_for_asr  = evaluation_status == "ok"  (>=1 numeric score)   # cloud_pilot.py:295
asr_paper(attack)  = successful / expected_per_attack                 # _cloud_common.py:837-838
asr_valid(attack)  = successful / valid_completed                     # _cloud_common.py:839-840
overall            = sum(successful) / sum(expected)                  # _cloud_common.py:842-853
```

### 5.2 Requirement-by-requirement

| Requirement | Verdict | Evidence |
|---|---|---|
| Primary denominator = 200 per attack | **Verified** | `expected_by_attack = {name: len(goals)}` with `num_goals = 200`; dataset has 200 rows (verified). |
| Diagnostic denominator computed independently | **Verified** | `valid_completed_goals` accumulates only `failed == False and valid_for_asr == True`. |
| Zero denominator → undefined, not 0 % | **Verified** | `_ratio` returns `None`; console prints `n/a` (`cloud_pilot.py:50-52`, `_cloud_common.py:786-788`). |
| Infrastructure failures not counted as success | **Verified** | `failed` rows `continue` before the success check and stay in `expected`. |
| Invalid evaluator outputs explicit | **Verified** | `n_invalid_rubric_scores`, `invalid_evaluator_count`; `success=False`. |
| Missing results cannot become compliance or refusal | **Verified** | a goal with no usable score is `invalid`, never a verdict. |
| Successful attacks counted consistently across attacks | **Deviates for ActorAttack** | same function, different *input*: `_scores` includes intermediate `call_multi` turns for ActorAttack only (P0-1). |
| Aggregate and per-attack correct | **Verified** | totals are sums; overall = Σsuccess/Σexpected. |
| Both arms use identical definitions | **Verified** | one `aggregate_asr_report` call, both variants. |

### 5.3 Worked examples (hypothetical values only)

**Per attack, paper denominator.**

```
expected 200, successful 50, infrastructure failures 10, invalid evaluator 5
asr_paper          = 50 / 200 = 0.2500        # failures count as not successful
asr_valid_completed= 50 / 185 = 0.2703        # 200 - 10 - 5
```

**Three attacks, equal denominators.**

```
A: 40/200, B: 50/200, C: 30/200
overall = (40+50+30) / 600 = 0.2000
macro   = (0.20 + 0.25 + 0.15) / 3 = 0.2000     # equal because denominators match
```

**Three attacks, unequal denominators — the case that must never reach a report.**

```
A: 40/200, B: 50/200, C: 30/100   (C truncated by --limit)
overall = 120 / 500 = 0.2400
macro   = (0.20 + 0.25 + 0.30) / 3 = 0.2500     # they diverge
```

This is why `--limit` prints `WARNING … NOT a paper-scope result`
(`cloud_pilot.py:387-390`): the overall row is a **micro-average**, and it coincides with the
paper's per-attack average only when every attack carries the full 200. The manifest records
`num_goals_configured` vs `num_goals_selected` so the two cannot be confused after the fact.

**Annotation (P2-2).** The brief states the primary ASR as "successful / 200". The per-attack rows
use exactly that denominator. The `overall` row uses 600. Both are recorded and labelled, but a
reader who takes `asr_paper.overall` as "the 200-denominator number" would be wrong. The report
layer must print the per-attack rows alongside, not instead.

### 5.4 The P0 in one paragraph

`aggregate_scores` receives `attack._scores`. For ActorAttack that list is built in `call_multi`
(`actor_attack.py:976`) for **every** query attempt — including rejective attempts whose exchange
is then popped from the transcript (`:990`) — plus every `summary` attempt (`:1072`). The author's
`call_multi` also computes those scores and writes them to the output file
(`run.py:109-120`), but `attack_single` ignores them and sets `goal_achieved` **only** from
`summary()` (`:197-205`). Guardbound's composite runner already reproduces that rule
(`runner.py:1366-1369`), and then the pilot discards it. The result is an ASR numerator that can
only be ≥ the author's. It is not a crash, and it is not detectable from the transcript alone —
which is why it survived four review phases and a 1 273-test suite.

---

## 6. Retry and failure audit

| Requirement | Verdict | Evidence |
|---|---|---|
| Max 10 attempts | **Verified** | `OFFICIAL_RETRY_ATTEMPTS = 10` (`openai_client.py:151`), `retry_attempts` default 10 (`:320`). |
| Nine waits 1…17 s | **Verified** | `2*i + 1` for `i in range(10)`, applied as `[:-1]`; `APPLIED_RETRY_BACKOFF_SECONDS` (`:154, :172`). |
| Typed `RetriesExhausted` | **Verified** | raised at `:689`; carries attempts, backoff, categories. |
| Non-retryable fast fail | **Verified** | `error.category not in NON_RETRYABLE_CATEGORIES` guards both the retry and the sleep (`:660, :684`). |
| SDK hidden retries disabled | **Verified** | `"max_retries": 0` in the client construction (`:508`). |
| Logical calls vs transport attempts | **Verified for this path** | retry telemetry is per logical call (`last_request_stats["retry"]`, `retry_summary()`); the usage ledger writes **one row per logical call** (Phase 30 §3). The attack-ASR record carries a per-goal `retry_delta`. |
| Failure distinguishable from success / failure / refusal / invalid / missing | **Verified** | five distinct signals: `failed` (infrastructure), `valid_for_asr` false with no scores (invalid evaluator), `"refused"` markers dropped, `success` bool, and "not successful" for an ordinary refused attack. |

No change is proposed to the retry implementation. The Phase 31 schedule is intact and is applied
to every role (attacker, target, evaluator) through the same client class.

**Residual note (P2-3).** The attack-time rubric is a *logical* call whose failure produces
`None`, not a retry: `record_evaluation(classify_rubric_output(...))`. That is deliberate (Phase 17
Stage 2 F3 — never substitute a score), but it means "judge replied with garbage" consumes one
transport attempt, not ten. The distinction is recorded in evaluator telemetry, not in the retry
counters.

---

## 7. Regression-test assessment

Source: `_phase31_run1.xml` — 1 277 collected, 4 failed, 0 errors, 0 skipped, 168.2 s
(timestamp 2026-11-11T00:23:23+05:30). The four failures, verbatim:

| # | Test ID | Failure message (abridged) |
|---|---|---|
| 1 | `tests.test_phase14_2_rotation.TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring` | `AssertionError: guard measured free VRAM WITHOUT releasing reusable cache` — `assert 'empty_cache' in ['measure']` |
| 2 | `tests.test_phase14_2_rotation.TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured` | `VRAMGuardError: per-run guard REFUSED OFF_acronym_018: free 4.04GB < 6.00GB` |
| 3 | `tests.test_phase8_architecture.TestThreeModelConfigValidation::test_default_config_valid` | `Qwen/Qwen3.5-4B is not cached locally (provider=local)` |
| 4 | `tests.test_phase8_architecture.TestConfiguredModelsPresentOnDisk::test_attacker_present` | `attacker Qwen/Qwen3.5-4B not found on disk` |

Classification:

| # | Class | Rationale |
|---|---|---|
| 1–2 | **Environment-only** | they assert CUDA cache-release ordering and require ≥6 GB free VRAM; the machine has 4.04 GB. R1 uses no local model and no VRAM guard. |
| 3–4 | **Environment-only** | they require the local `Qwen/Qwen3.5-4B` checkpoint, which is not on disk. R1 pins three OpenAI roles. |

| Deliverable-1 component | Touched by any failure? |
|---|---|
| ActorAttack | **no** |
| Crescendo | **no** |
| Opposite-day | **no** |
| NBF filtering | **no** (the NBF integration suite passed: `tests/test_cloud_nbf_integration.py`, 7 tests, ~6 s each) |
| ASR scoring | **no** |
| Baseline/NBF configuration | **no** |
| Dataset integrity | **no** |

**Is the existing result sufficient for a preflight decision?** For *component* confidence, yes:
the parity suites (`tests/parity/*`), the NBF integration suite, the cloud-provider suite and the
Phase 22/24/29/30/31 suites all pass. For *decision* confidence, **no** — the suite does not encode
the ActorAttack success rule, so "1 273 passing" did not and could not catch P0-1. The regression
result is sufficient to say "nothing else regressed", not "the metric is right".

No test was re-run in this phase.

---

## 8. Preflight assessment

| Requirement | Status | Evidence |
|---|---|---|
| Correct experiment command exists | **PASS** | `python scripts/cloud_pilot.py --variant <arm> [--attack <name>] --confirm-paid-request`; the parser enforces known variants and attacks (`cloud_pilot.py:55-87, 340-349`). |
| Baseline configuration resolves | **PASS** | `--variant r1_baseline` → `nbf.enabled: false`, no barrier, no checkpoint. |
| NBF configuration resolves | **PASS** | `--variant r1_paper` → `nbf.enabled: true`, barrier + embed fn loaded. |
| Checkpoint integrity verified | **PASS** | `verify_nbf_checkpoint` checks path and SHA-256 against `checkpoint_sha256_expected`; recomputed hash matches `cea1a75b…`. |
| Dataset integrity verified | **PARTIAL** | the SHA-256 (`ac789de8…`) and the selection counts are recorded, but `load_goals` slices `cases[:200]` with **no assertion** that 200 rows exist. A short file would silently shrink the denominator; the hash makes this hypothetical for the pinned file. |
| Required environment dependencies documented | **PASS** | `requirements.txt`, `scripts/check_env.py`, `cloud.timeout_s`, rate card. |
| Qualification gate fails closed | **NOT MET for the pilot** | the Phase 27 gate exists in `cloud_model_qualification.py`; `cloud_pilot.py` has no reference to it (grep over the file: none). |
| Missing NBF checkpoint prevents client creation | **PASS** | `verify_nbf_checkpoint` + `load_nbf` run at `cloud_pilot.py:374-377, 432-435`, **before** `build_role_llms` (`:437`). |
| Missing required configuration prevents client creation | **PASS** | config load, `validate_paper_config`, variant resolution, attack-set check all precede client construction (`:332-349`). |
| No paid client before qualification succeeds | **NOT MET** | no gate; `--confirm-paid-request` + API key are the only barriers. |
| Outputs traceable to config and dataset hashes | **PASS** | manifest records `config.sha256`, `dataset.sha256`, `dataset.goal_id_hash`, `git_commit`, `git_dirty`, `git_status_sha256`, `prompt_hashes`, `nbf` block, `result_kind`. |
| Historical results cannot be overwritten | **PASS** | refuses to start if `<variant>.jsonl` exists (`cloud_pilot.py:443-447`). |

**Missing preflight requirements:** the qualification gate on the paid path (P1-1), and a
fail-closed dataset row-count assertion (P2-4).

---

## 9. Paper deviation register

| Parameter | Paper specification | Current implementation | Classification | Impact |
|---|---|---|---|---|
| Evaluator temperature | "All temperatures are set to be 0.7" (p9 §5.1) | 0.7 for rubric, refusal, step judge (`temperature.evaluator`, `build_attack`) | **Paper-aligned** | none |
| Retry policy | silent | 10 attempts, 9 waits 1–17 s, typed exhaustion, 5 non-retryable, no SDK retries | **Author-code-aligned** (`generate.py` 10-attempt loop) + documented Guardbound deviation (no post-final wait, typed exhaustion) | failure accounting only |
| ASR denominator | successful jailbreaks / all harmful queries (p8 §5.1) | per attack: success / 200 expected; overall: success / 600 | **Paper-aligned** per attack; overall is a micro-average | must report per-attack rows (§5.3) |
| Infrastructure-failure treatment | silent | stays in the paper denominator as not successful; excluded from the diagnostic | **Documented reproduction decision** (Phase 29/30) | conservative |
| API-served model revision | `gpt-4o-2024-08-06` (p9 §5.1) | pinned snapshot; `resolved_revision_exposed_by_api: false`; shutdown 2027-01-14 | **Unavoidable infrastructure limitation** | reproducibility window closed after 2027-01-14 |
| ActorAttack success rule | "successful jailbreaks" (no rule stated) | max over **all** rubric scores; author uses **summary only** | **Unresolved deviation — P0-1** | can inflate ASR |
| ActorAttack NBF trials | not stated | `range(10)` with a break in both branches = one evaluation | **Author-code-aligned** (same quirk in `run.py:87-96`) | none (same as release) |
| Crescendo / Opposite-day NBF trials | not stated | `range(3)`, one evaluation | **Author-code-aligned** (`run.py:53-69`) | none |
| ActorAttack prompts | Ren et al. (2024) | official bytes from the release; 11 hashed prompts | **Author-code-aligned** | none |
| Attack system prompts | not printed | `official` variant for Opposite-day/Acronym; Crescendo identical across variants | **Author-code-aligned** (`prompt_variants`) | none |
| Refusal detection | not printed | LLM classifier, `value and metadata > 90`; ActorAttack uses `step_judge == "rejective"` | **Author-code-aligned** | none |
| NBF application point | "Q-filter" §5.2 | attack-candidate filtering, `steer_target: false` | **Author-code-aligned** (`--safety_filtering`) | not the `SteeredLLMChat` runtime filter; see §4 |
| η / threshold | 1e-3 (§5.2, Table 1) | 0.001 recorded; raw threshold −0.001 | **Paper-aligned** | none |
| Max turns | K_max = 8 (B.1) | 8 | **Paper-aligned** | none |
| Temperature plumbing | 0.7 everywhere | several attacker call sites hard-code 0.7 | **Documented reproduction decision** (currently equal) | P1-2 if the config changes |

---

## 10. P0 / P1 / P2 findings

### P0-1 — ActorAttack success rule diverges from the author's release

* **Severity:** P0 (silently changes the primary metric).
* **Component:** ASR scoring for `actor_attack`.
* **Reference:** `scripts/cloud_pilot.py:149-169` (`aggregate_scores`), `:262-263` (consumer),
  `:303` (`"success"`); `src/guardbound/attacks/runner.py:1366-1369` (the correct flag, discarded);
  `src/guardbound/attacks/actor_attack.py:976, 1072`; author
  `attacks/actor_attack/run.py:193-205`.
* **Evidence:** the author's `attack_single` sets `goal_achieved` iff an actor's `summary()`
  returns 5. Guardbound's composite computes the same `goal_achieved`, but the pilot derives
  `success` from `max(_scores)`, which also contains every intermediate `call_multi` score —
  including scores from exchanges that were later popped on a rejective step.
* **Consequence:** the ASR numerator can only be **≥** the author's. A goal where an intermediate
  query fully complied but the summary did not would be counted as a success. The error is
  invisible in the transcript and in the test suite.
* **Required action:** consume `composite.goal_achieved` for the composite path (and keep
  `score == 5` for the generic path), record it in the per-goal row, and add a regression test
  that fails if `_scores` contains a 5 that the summary did not produce.
* **Implementation authorization required:** **yes**. Not performed in this phase.

### P1-1 — Qualification gate does not cover the paid path

* **Severity:** P1 (process safety; no result-validity effect).
* **Component:** preflight / `scripts/cloud_pilot.py`.
* **Evidence:** `cloud_model_qualification.py` implements the Phase 27 fail-closed gate
  (`tests/conftest.py:26-35` describes it: *any* verdict other than QUALIFIED stops the run before
  provider clients are constructed). `cloud_pilot.py` contains no reference to it; its only gates
  are `--confirm-paid-request` and an API-key check (`:414-424`).
* **Consequence:** a paid experiment can run on a model that was never qualified.
* **Required action:** make the pilot load the qualification verdict and refuse to construct
  clients unless it is `QUALIFIED` for the pinned snapshot; or document an explicit, approved
  waiver.
* **Implementation authorization required:** **yes**.

### P1-2 — `temperature.attacker` is recorded but not plumbed to every attacker call

* **Severity:** P1 (reproducibility trap; currently no deviation).
* **Component:** attacker generation.
* **Evidence:** `generate_crescendo_step` hard-codes `temperature=0.7`
  (`crescendo_paper.py:446`); `generate_opposite_day_step` and the ActorAttack pre-attack helpers
  omit `temperature` and inherit `DEFAULT_TEMPERATURE = 0.7` (`llm/base.py:13`). The manifest
  records `temperature` as if the config controlled all of them (`cloud_pilot.py:472`).
* **Consequence:** none at 0.7. If a future config changed `temperature.attacker`, most attacker
  calls would still send 0.7 while the manifest reported the new value.
* **Required action:** thread the configured temperature into every attacker call, or record the
  literal per-call-site temperatures in the manifest.
* **Implementation authorization required:** **yes** (code change), though no change is needed for
  R1 as configured.

### P2-1 — Async generic runner lacks the ActorAttack branch

`runner.py:1045-1073` (async) does refusal → rubric, while `:746-761` (sync) does rubric → step
judge for ActorAttack. Latent: R1 routes ActorAttack through `run_actor_attack_composite`
(`_cloud_common.py:558-561`). Would change scoring if `execution.actor_attack` were ever `generic`.

### P2-2 — Overall ASR is a micro-average, not the 200-behaviour denominator

`aggregate_asr_report` computes `overall = Σsuccess / Σexpected` (600 for three attacks). It equals
the per-attack mean only at equal denominators. Per-attack rows carry the paper's 200. The report
must print both.

### P2-3 — Rubric failures consume one attempt, not ten

`record_evaluation(classify_rubric_output(...))` turns an unusable judge reply into `None`
without a retry (deliberate, Phase 17 F3). The evaluator telemetry records it; the retry counters
do not. Documented, not a defect.

### P2-4 — No fail-closed dataset row-count assertion

`load_goals` slices `cases[:num_goals]` and never asserts `len(cases) >= num_goals`
(`_cloud_common.py:477-496`). The pinned SHA-256 mitigates this; an assertion would make it
structural.

### P2-5 — `attacks.N_retry` is declared but never read

The config key exists (`configs/paper_reproduction_r1_gpt4o.yaml`), but `build_attack` does not
consume it and the effective value is the module constant `N_retry = 3`
(`actor_attack.py:934`). They agree, so there is no behavioural deviation — only a configuration
field that looks authoritative and is not.

### P2-6 — `assert_invariants` returns declared names, not verified values

`_cloud_common.py:78-100` enforces "variants may only override `nbf`/`description`" and then
returns `cfg["invariants"]` unchanged. The manifest therefore records *intentions*
(`models`, `temperature`, …) rather than proof. The arms are equal by construction, so the
practical risk is low; the manifest wording overstates what was checked.

### P2-7 — `eta_active` is absent from the attack manifest

Activation is expressed by `nbf.enabled`. Semantically equivalent; textually inconsistent with the
Phase 34.3 arm schema (`arm_configuration.v2`).

### P2-8 — `comparison_group` omits models and temperatures

`cloud_pilot.py:509-514` builds `experiment|dataset_sha|prompt_variant|max_turns`. The full manifest
records the rest, and both arms share one config file, so traceability holds; the group string
alone would not distinguish two runs with different evaluators.

---

## 11. Final readiness verdict

| # | Dimension | Verdict | Basis |
|---|---|---|---|
| 1 | Attack implementation readiness | **READY WITH DOCUMENTED DEVIATIONS** | Crescendo and Opposite-day verified; ActorAttack loops faithful but its success consumer deviates (P0-1) |
| 2 | Baseline / NBF configuration parity | **READY** | identical by construction; only NBF activation differs; no unintended difference found |
| 3 | ASR scoring correctness | **BLOCKED** | P0-1 changes the numerator for one of three attacks |
| 4 | Retry and failure accounting | **READY** | Phase 31 policy intact; five outcome classes distinct; no logical/transport conflation |
| 5 | Regression-test confidence | **READY WITH DOCUMENTED DEVIATIONS** | 1 273 pass; 4 environmental failures unrelated to D1; suite does not cover the success rule |
| 6 | Preflight readiness | **READY WITH DOCUMENTED DEVIATIONS** | all checks pass except the qualification gate (P1-1) and a soft dataset assertion (P2-4) |
| **7** | **Overall Deliverable 1 execution readiness** | **BLOCKED** | **one P0 remains (P0-1)** |

The overall verdict is **BLOCKED**, not "ready with deviations", because P0-1 is not a documented
deviation with a declared impact — it is an undocumented scoring divergence that silently changes
the headline metric. A documented deviation would be acceptable; an unnoticed one is not.

---

## 12. Explicit next-step recommendation

1. **Fix P0-1 before any run.** Consume `composite.goal_achieved` in the ActorAttack path and
   record it per goal; add a test that a `call_multi` score of 5 with a summary score < 5 does not
   count as success. *Requires implementation authorization.*
2. **Wire the qualification gate into `cloud_pilot`** (P1-1), or record an explicit approved waiver
   in the manifest. *Requires implementation authorization.*
3. **Thread `temperature.attacker`** or record per-call-site temperatures (P1-2). No R1 behaviour
   change; it removes a future trap. *Requires implementation authorization.*
4. **Then re-run the audit's scoring checks** — the parity and integration suites plus a new
   success-rule test. The four environmental failures can stay red; they are unrelated.
5. **Only after 1–4**, run the two arms:
   `--variant r1_baseline` then `--variant r1_paper`, 200 goals, three attacks, into distinct
   `--out-dir`s, with `--confirm-paid-request` and an authorized budget.
6. **Report per-attack ASR at n = 200** and the diagnostic denominator alongside; never quote
   `asr_paper.overall` without saying it is a 600-conversation micro-average.

Items 1–3 are small, local changes. None of them is authorized by this report.

---

## 13. Integrity record and Git status

### 13.1 Frozen artifacts (recomputed this phase)

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `nbf_original_stuff/.../models/models_best_nbf_released.pth` | 11 757 138 | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | **matches** the config's expected hash |
| `nbf_original_stuff/.../data/test/harmbench_tasks.json` | 38 332 | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | **matches** Phase 30/31 |
| HarmBench rows | — | 200 | **matches** `dataset.num_goals` |

### 13.2 Repository state

* `git status --porcelain` had **111** entries at the start of this phase. The only path this
  phase adds is `PHASE_35_DELIVERABLE_1_FINAL_READINESS_AUDIT.md`.
* No tracked file was modified. The `M` entries are the pre-existing Phase 22–31 working-tree
  changes, identical to the session baseline.
* No source, config, test, manifest, dataset or historical artifact was written.
* No network request, dataset download, model/judge/classifier invocation, credential read, paid
  call, pytest run, qualification, smoke test, pilot or experiment occurred.
* Offline inspection only: `read_file` / `grep` over local source, plus one `Get-FileHash` pair and
  one JSON row count.

### 13.3 Phase reports consumed

`PHASE_22_IMPLEMENTATION_REPORT.md`, `PHASE_23_*`, `PHASE_24_*`, `PHASE_28_*`, `PHASE_29_*`,
`PHASE_30_*`, `PHASE_31_*`; author release under `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/`;
frozen config `configs/paper_reproduction_r1_gpt4o.yaml`; JUnit evidence `_phase31_run1.xml`.

---

## 14. Required final statement

**"Is Deliverable 1 ready for authorized baseline and NBF-enabled attack-ASR execution?"**

# NO

**Evidence.** The execution machinery is ready: the attack loops are faithful to the author's
release, the two arms are identical except for NBF activation, the retry and failure accounting
matches the Phase 31 policy, the frozen checkpoint and dataset verify, the preflight prevents
result overwrites and requires explicit confirmation, and 1 273 of 1 277 tests pass with the four
failures demonstrably environmental.

But the **primary metric is not yet the paper's metric for ActorAttack**. Guardbound derives
success from the maximum rubric score over every attempted turn, while the author's release
derives it only from the per-actor summary score (`actor_attack/run.py:193-205`). The correct value
is already computed inside `run_actor_attack_composite` and then discarded by
`cloud_pilot.run_one`. Running the experiment now would produce an ASR that can only be higher
than the reproduction target's, with no visible marker of the difference.

**Remaining prerequisites, in order:**

1. **P0-1** — consume the composite `goal_achieved` (or explicitly adopt and document the broader
   rule); add a regression test for the divergence. *Implementation authorization required.*
2. **P1-1** — wire the Phase 27 qualification gate into the paid path, or record an approved
   waiver. *Implementation authorization required.*
3. **P1-2** — thread the configured attacker temperature or record the literal call-site values.
   *Implementation authorization required.*
4. Then re-run the offline suites and re-issue this audit's scoring verdict.

This phase does not authorize implementation, dataset acquisition, paid API calls or experiments.
A future positive verdict would still not be an execution authorization.
