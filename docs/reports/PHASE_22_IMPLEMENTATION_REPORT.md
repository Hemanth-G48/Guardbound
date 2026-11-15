# PHASE 22 — PAPER-FIDELITY IMPLEMENTATION AND BUDGET-ENFORCEMENT REMOVAL

**Status: `READY WITH DOCUMENTED DEVIATIONS`** (implementation only).

No paid API request, smoke test, qualification, pilot or experiment was run. The
deliverable is a paper-faithful implementation with offline validation — not
experimental results.

Paper: *Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking
Attacks*, Hu / Robey / Liu, arXiv:2503.00187v3, TMLR 02/2026. The published paper
is the methodological authority; the author's release
(`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/`) supplies
implementation details only where the paper is silent.

---

## 1. Executive summary

| Requirement | Result |
|---|---|
| P0-1 evaluator temperature parameterised (paper 0.7, legacy 0.0 default) | **done** — 8 call sites, all attack classes |
| P0-2 paper main-evaluation attack set enforced | **done** — ActorAttack + Crescendo + Opposite-day; Acronym training-only |
| P0-3 ActorAttack official composite execution | **done** — `run_actor_attack_composite`, selected by config |
| P1-1 headline η = 1e-3 in the R1 configuration | **done** — validated, reaches the Q-filter |
| Official prompt bytes for the paper path | **done** — `official` variant proven byte-identical to the release |
| Custom `$1.50` budget cap and enforcement removed | **done** — no cap, no blocking; usage still reported |
| NBF implementation preserved | **verified** — checkpoint hash, dims, score formula, threshold semantics |
| Offline suite | **1067 collected / 1063 passed / 4 failed / 0 errors**, identical failure set to the baseline |
| Frozen experiment artifacts | **unchanged** |

Two things a reader must not infer from this report: that any model is qualified,
or that any result reproduces a paper table.

---

## 2. Git state and changed-file inventory

```
branch            main
HEAD              89ed03f  (plus the pre-existing Phase 20-21 uncommitted work)
frozen paths      results/ runs/ checkpoints/ data/ configs/reproduction*.yaml
                  -> `git status --porcelain` empty for all of them
checkpoint        models_best_nbf_released.pth
                  sha256 cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe
```

Nothing was reset, cleaned, stashed, reverted or overwritten; no `git` write
operation was performed.

### Changed (Phase 22 delta, `git diff --numstat` against HEAD)

| File | +/− | Phase 22 change |
|---|---|---|
| `src/guardbound/llm/base.py` | +18 | `PAPER_EVALUATOR_TEMPERATURE`, `LEGACY_EVALUATOR_TEMPERATURE`, `ACTOR_STEP_JUDGE_TEMPERATURE` + provenance comments |
| `src/guardbound/attacks/crescendo_paper.py` | +69 / −20 | temperature params, `official_prompt` flag, `_build_rubric_prompt` extracted, class knobs |
| `src/guardbound/attacks/opposite_day.py` | +55 / −8 | same + `system_template` for the attacker prompt |
| `src/guardbound/attacks/acronym.py` | +59 / −10 | same (refusal **and** disclaimer) |
| `src/guardbound/attacks/actor_attack.py` | +67 / −10 | rubric + step-judge temperatures, prompt variant, `_judge_llm` routing |
| `src/guardbound/attacks/runner.py` | +246 | `run_actor_attack_composite`, `ActorAttackRun`, `_SteeredAttackTarget` |
| `src/guardbound/llm/usage.py` | (see note) | budget fields/methods removed; observational properties kept |
| `src/guardbound/llm/cloud_errors.py` | (see note) | `BudgetExceededError`, `CATEGORY_BUDGET` removed |
| `src/guardbound/llm/openai_client.py` | (see note) | pre-request `check_budget` call removed |
| `scripts/_cloud_common.py` | | ledger without budget; `build_attack` gains ActorAttack + knobs; `prompt_variant_of`, `evaluator_temperature_of`, `uses_composite_execution`, `validate_paper_config` |
| `scripts/cloud_pilot.py` | | R1 validation before any work, composite path, manifest fields, `--budget-usd` removed |
| `scripts/cloud_smoke_test.py` | | `--budget-usd` removed, budget prints replaced by `unpriced_calls` |
| `scripts/cloud_model_qualification.py` | | `--budget-usd` removed |
| `scripts/audit_prompt_fidelity.py` | | `check_prompt_variants()` + `paper_path_faithful` aggregate |
| `tests/test_cloud_{provider,usage,pricing_and_snapshots,qualification_harness}.py` | | budget tests replaced by absence-of-enforcement tests |
| `configs/cloud_{qualification,research_reference,research_gpt4o_mini,research_gpt4o_mini_c1}.yaml`, `cloud_pricing.yaml` | | `budget_usd` removed, comments corrected |

> Note on `usage.py` / `cloud_errors.py` / `openai_client.py`: `usage.py` and
> `cloud_errors.py` are untracked files created during Phase 21 (so they show no
> `git diff`), and Phase 22 edited them in place. `openai_client.py` **is**
> tracked, and its diff against HEAD also contains the earlier Phase 20-21 work;
> the Phase 22 delta there is three lines (the removed pre-request budget check).

### New

| File | Lines | Purpose |
|---|---:|---|
| `src/guardbound/paper_protocol.py` | 223 | R1 invariants: attack set, temperatures, η, model identity, no budget |
| `src/guardbound/attacks/prompt_variants.py` | 160 | `legacy` / `official` prompt resolution + SHA-256 fingerprints |
| `src/guardbound/attacks/_official_prompts.py` | 25 | release prompt bytes, generated (not hand-typed) |
| `configs/paper_reproduction_r1_gpt4o.yaml` | 155 | the R1 configuration |
| `tests/test_phase22_paper_fidelity.py` | 791 | 62 offline tests |
| `scripts/_codegen_official_prompts.py` | 74 | regenerates `_official_prompts.py` from the release |
| `PHASE_22_IMPLEMENTATION_REPORT.md` | this file | |

Modified docs: `HANDOFF.md` (rewritten), superseded banners on
`docs/audit/{PHASE_21_REPORT, CLOUD_SETUP_GUIDE, CLOUD_EXPERIMENT_PROTOCOL_FINAL,
CLOUD_MODEL_CONFIGURATION, CLOUD_FINAL_READINESS_REPORT,
CLOUD_FINAL_READINESS_RECONCILIATION, CLOUD_MODEL_QUALIFICATION_REPORT}.md`.

---

## 3. P0-1 — evaluator temperature

### Before

Every evaluator/judge call hard-coded `0.0` at eight sites; ActorAttack's rubric
(`actor_attack.py:544`) passed `0.0` where both the paper **and** the author's
release use 0.7 for that path.

### After

```python
PAPER_EVALUATOR_TEMPERATURE  = 0.7   # paper p9 Sec. 5.1
LEGACY_EVALUATOR_TEMPERATURE = 0.0   # default, preserves frozen Phase 14-19 behaviour
ACTOR_STEP_JUDGE_TEMPERATURE = 0.7   # the release lets step_judge inherit 0.7
```

| Site | Function | Default | Paper config |
|---|---|---|---|
| `crescendo_paper.py` ×2 | `llm_check_refusal`, `llm_evaluate_with_rubric` | 0.0 | 0.7 |
| `opposite_day.py` ×2 | same | 0.0 | 0.7 |
| `acronym.py` ×3 | refusal, disclaimer, rubric | 0.0 | 0.7 |
| `actor_attack.py` ×1 (+ step judge) | rubric; step judge | 0.0 / 0.7 | 0.7 / 0.7 |

Mechanism: each attack class takes `evaluator_temperature` (ActorAttack also
`step_judge_temperature`) and forwards it to the module-level helpers; the helpers
take `temperature` with the legacy default. No evaluator overrides the configured
value — `record_turn`, `check_refusal`, `evaluate_response`, `call_multi` and
`summary` all read the same field.

Behaviour before → after, for identical inputs:

| Caller | Before | After (no arg) | After (`temperature=0.7`) |
|---|---|---|---|
| `evaluate_with_rubric(...)` | 0.0 | **0.0** (unchanged) | 0.7 |
| `check_refusal(...)` | 0.0 | **0.0** (unchanged) | 0.7 |
| `check_disclaimer(...)` | 0.0 | **0.0** (unchanged) | 0.7 |
| `ActorAttack.step_judge(...)` | 0.7 (client default) | **0.7** (now explicit) | 0.7 |

Proof of byte stability: the rendered **legacy** rubric prompt hashes identically
before and after the change (`4b5504a5…`, fixture `"Q"`/`"R"`).

Provider payload: `OpenAIChatLLM.build_request(messages, 0.7, True)` yields
`temperature: 0.7` and `response_format: {"type": "json_object"}`.

Tests (all offline): default-0.0 for all four modules, explicit-0.7 forwarding,
attack-level rubric/refusal temperature, Acronym's refusal+disclaimer pair,
ActorAttack step judge, ActorAttack rubric, a source-level guard that no
`temperature=0.0` literal survives in the four attack modules, R1 resolution, and
the provider payload field.

---

## 4. P0-2 — main-evaluation attack set

The paper's main evaluation is ActorAttack, Crescendo, Opposite-day (p9 Sec. 5.1).
Acronym "was collected for training but omitted from the main evaluation because it
contained too many single-turn successful jailbreak conversations" (p22, B.1).

`src/guardbound/paper_protocol.py` encodes the set and the reason, and
`validate_paper_config()` runs **before any other work** in the pilot, i.e. before
the first billable request.

Rejected configurations (each has a test):

| Input | Result |
|---|---|
| `enabled` includes `acronym` | `PaperProtocolError: ... training-only ...` |
| any of the three paper attacks missing | `... missing=['actor_attack'] extra=[] ...` |
| unknown identifier (`crescendo`) | `unknown attack identifier(s) ['crescendo']` |
| duplicate identifier | `duplicate attacks in attacks.enabled: [...]` |
| empty list | `attacks.enabled must be a non-empty list` |
| `training_only: []` in a paper run | `must declare attacks.training_only: ['acronym']` |
| an extra supported attack (`red_queen`) in a paper run | rejected (set must be exact) |

Historical configurations are **not** rewritten: `cloud_research_reference.yaml`
still lists Acronym and omits ActorAttack, and `validate_paper_config()` returns
`None` for it because it is not flagged `paper_reproduction: true`. Its
identifiers are still checked, so a typo cannot silently drop an attack.

Manifest: the pilot writes `paper_protocol` (resolved block) including `attacks`,
`training_only`, `attack_set_reason`, and `prompt_variant`.

Test utility `common.build_attack` also gained `actor_attack` and the two
fidelity knobs, so a configuration can actually instantiate the paper set.

---

## 5. P0-3 — ActorAttack composite execution

### Problem

`ActorAttack.run_full_attack` — the official composite — had **zero callers**. The
generic `next_query` loop loses the per-actor conversation reset, the `summary()`
turn and the per-actor early stop.

### Implementation

`src/guardbound/attacks/runner.py`:

```python
run_actor_attack_composite(attack, goal, target_llm, target_system=..., ...)
    -> ActorAttackRun(conversation, actor_records, goal_achieved,
                      final_score, skipped_actors, prepared_once)
```

It calls the attack's **own** `prepare_attack`, `call_multi` and `summary` — the
official implementations — rather than re-deriving them, and it is selected
explicitly:

```yaml
attacks:
  execution:
    actor_attack: composite
    crescendo_paper: generic
    opposite_day: generic
```

The generic runner is untouched for the other attacks. `_SteeredAttackTarget`
adapts `SteeredLLMChat` to the `ChatLLM` interface `call_multi` expects (only the
last user message goes through the Q-filter), and the steered wrapper is sized to
one full chain per actor because the official ActorAttack has no global turn cap.

### Behaviour before → after

| | Before | After (composite) |
|---|---|---|
| Actors per goal | one flat `next_query` stream | 3 independent conversations, each `[system] + chain` |
| Retries per query | none (generic loop) | `N_retry = 3` → 4 attempts, rewrite on `rejective` |
| Summary turn | never | once per non-skipped actor |
| `unknown` verdict | skipped actor only via `record_turn` | actor aborted, goal continues |
| Early stop | none | stop the goal at summary score 5 |
| ASR denominator | one `Conversation` | one `Conversation` per goal, per-actor records alongside |

### Contract tests (scripted mocks, no network)

| Test | Asserts |
|---|---|
| `test_composite_calls_prepare_attack_exactly_once` | `prepare_attack` called once per goal |
| `test_every_actor_starts_a_fresh_conversation_with_the_system_message` | 3 fresh `[system]` conversations, at call indices 0/3/6 |
| `test_actor_query_chains_execute_in_official_order` | `a0q0, a0q1, summary, a1q0, a1q1, summary` |
| `test_retry_budget_is_three_retries_four_attempts` | 4 target attempts for a permanently rejective query |
| `test_summary_runs_once_per_non_skipped_actor` | 3 summaries for 3 actors |
| `test_score_five_stops_the_goal_after_the_first_actor` | 1 actor record, `goal_achieved=True`, status `successful` |
| `test_unknown_step_verdict_skips_only_that_actor` | statuses `['skipped','completed','completed']`, no summary for the skipped actor |
| `test_rubric_is_evaluated_before_the_step_judge` | call order rubric → step judge → … → summary rubric |
| `test_per_actor_records_stay_separate_and_asr_counts_one_behaviour_per_goal` | 3 records, one `Conversation` |
| `test_two_goals_produce_three_actor_conversations_each` | 6 actor conversations across 2 goals |
| `test_composite_matches_the_official_run_full_attack_call_sequence` | **identical target call sequence** to `run_full_attack` (the transcription of the author's `attack_single`) |
| `test_composite_rejects_an_attack_without_the_official_methods` | `TypeError` for a non-ActorAttack |
| `test_composite_requires_both_barrier_and_embed_fn_to_steer` | `eta=1e-3` and the per-actor turn guard reach `SteeredLLMChat` |

The last contract test is the strongest available offline statement: for identical
scripted inputs the composite runner emits exactly the message array sequence that
the official loop emits. It does not prove the *model's* behaviour — only the
control flow.

---

## 6. P1-1 — headline NBF threshold

`configs/paper_reproduction_r1_gpt4o.yaml` → `nbf.eta: 0.001`,
`nbf.threshold: 0.001` (paper p9 Sec. 5.2, Table 1, Tables 8–9). `0.0` is the
ablation (Tables 6–7) and is rejected for R1 by `resolve_paper_reproduction`
(`nbf.eta must be 0.001 ...`).

Verification chain:

1. `resolve_paper_reproduction(cfg)["eta"] == 1e-3` (test).
2. `common.nbf_settings(cfg, variant)["eta"] == 0.001` and `threshold == 0.001` (test).
3. The pilot reads that value and passes it as `eta=` to the attack runner; the
   composite path passes it to `SteeredLLMChat(eta=...)` (test captures the
   constructor argument).
4. The manifest records `nbf` (eta, threshold, checkpoint sha256) and
   `paper_protocol.eta`.
5. `--dry-run` prints `NBF eta/threshold 0.001 / 0.001` (observed).

The checkpoint, architecture, embedding model, initial state, predictor, score
formula and comparison semantics were **not** modified.

---

## 7. Official prompt bytes

### Method

`scripts/_codegen_official_prompts.py` imports the release's modules and writes
`src/guardbound/attacks/_official_prompts.py` containing `repr()` of the exact
strings, so whitespace, trailing spaces and Unicode code points survive. The rubric
prompt is *captured rendered* from the official `evaluate_with_rubric` with
sentinel inputs, then re-parameterised.

`src/guardbound/attacks/prompt_variants.py` exposes both variants:

| Prompt | `legacy` (frozen runs) | `official` (R1) |
|---|---|---|
| Crescendo attacker | 1 text — already identical | identical |
| Opposite-day attacker | historical bytes | release bytes |
| Acronym attacker | historical bytes | release bytes |
| Rubric (4 copies) | historical bytes | release bytes |

Selection is explicit: `prompt_variant: official` in the R1 config, `legacy` as the
code default. Attacks take `prompt_variant`; the step functions take
`system_template`; the rubric helper takes `official_prompt`.

### Comparison (executed, `scripts/audit_prompt_fidelity.py`)

`paper_path_faithful = true` — for every prompt that differs, the `official`
variant is **byte-identical** to the release:

| Prompt | legacy SHA-256 | official SHA-256 | matches release |
|---|---|---|---|
| Opposite-day attacker | `1cd2c2d0120bc88d…` | `dfd5b4da0a580630…` | **yes** |
| Acronym attacker | `6b8b11a5ff15f06a…` | `f3afb6a9c9fb6c20…` | **yes** |
| Rubric | `4b5504a570e0ce56…` | `146589a4fba6d42c…` | **yes** |

(All hashes use the same fixture, `query="Q"`, `response="R"`; the rendered
official rubric is additionally compared to the release's rendered message for a
real fixture and is equal.)

Size deltas confirm the documented deviations exactly: Opposite-day
official − legacy = **+72** characters (2 blank lines × 36 spaces), Acronym
**+36** (1 blank line × 36 spaces), rubric **−2** net (3 missing spaces, 1 extra
double-space).

Already byte-identical and left untouched: Crescendo attacker (all rounds),
refusal (both copies), disclaimer, target conversation shape, provider payload.

Acronym is training-only; restoring its prompt does not authorise running it, and
nothing in the R1 attack set includes it.

---

## 8. Budget-cap removal

### What the cap was

A Guardbound-only safeguard. Verified absent from **both** authorities: the paper
has zero mentions of budget/spend/cost/price across 31 pages (the only `$`
characters are LaTeX math delimiters; the only compute statement is "4 A6000
Nvidia GPUs with 512G RAM", p22 B.1), and the author's release has zero matches
for budget/spend/cost. Provenance class: **user/research decision**.

### Removed

| Item | Where it was | Now |
|---|---|---|
| `qualification.budget_usd: 1.50` | `configs/cloud_qualification.yaml:21` | gone (replaced by a comment stating there is no cap) |
| `cloud.budget_usd: null` | 3 cloud configs | gone |
| `--budget-usd` | 3 CLIs | gone (argparse exits 2) |
| `UsageLedger(budget_usd=..., .budget_usd, .remaining_usd, .check_budget())` | `llm/usage.py` | gone |
| `BudgetExceededError`, `CATEGORY_BUDGET` | `llm/cloud_errors.py` | gone; not in `ALL_CATEGORIES` |
| pre-request `ledger.check_budget(model)` | `llm/openai_client.py` | gone — no request can be blocked by spend |
| budget plumbing in `build_ledger` | `scripts/_cloud_common.py` | gone |
| budget prints in the CLIs | smoke / pilot / qualification | replaced by `unpriced_calls` / `usage_unavailable_calls` |
| budget-only tests | 4 test modules | replaced by tests that assert the cap is **absent** |

The R1 configuration additionally **rejects** a `cloud.budget_usd` key
(`must be absent: the paper specifies no API budget ...`), so the cap cannot return
through a config edit.

No new limit was introduced; no dollar figure replaced the old one.

### Preserved (independently useful)

Per-call usage records with model, provider/role, input / cached / output / total
tokens, duration, request id, outcome, error category, cost, cost basis; the
rate card (YAML, verified); `spent_usd` as a reported total;
`unavailable_usage_calls`; JSONL + summary persistence; the manifest blocks; all
provider error handling and credential redaction; and the honest rule that an
unpriceable call reports `None`, never `0`.

### Verification (offline)

| Requirement | Test |
|---|---|
| R1 config needs no budget field | `test_the_r1_configuration_has_no_budget_field` |
| R1 rejects a budget field | `test_a_budget_field_is_rejected_for_r1` |
| No shipped config declares a cap | `test_no_shipped_config_declares_a_budget_cap` (scans every `configs/*.yaml`, ignoring comments) |
| No CLI accepts `--budget-usd` | `test_no_cli_accepts_a_budget_flag`, `test_budget_flag_is_gone_from_the_qualification_cli` |
| Nothing blocks a request | `test_usage_telemetry_never_blocks_a_request` (3 calls), `test_missing_usage_is_recorded_honestly_and_does_not_stop_the_run` |
| Unknown price is not invented | `test_cost_is_none_not_zero_when_price_is_unknown` |
| Usage does not alter the payload | `test_usage_accounting_does_not_alter_the_provider_payload` (identical request bodies with and without a ledger) |
| Error type gone | `test_budget_error_type_is_gone` |
| Provider behaviour unchanged | the full pre-existing provider suite |

---

## 9. Usage telemetry status

Intact and observational. One `UsageRecord` per provider call, written for
successes and failures alike, with `usage_available=False` and
`estimated_cost_usd=None` when the provider returns no usage object. The ledger
exposes `spent_usd` (reported total) and `unavailable_usage_calls`, and its
`summary()` no longer contains `budget_usd` / `remaining_usd`. The pilot and the
qualification harness still persist `<variant>.usage.json`,
`<variant>.usage.jsonl` and the per-run rows.

---

## 10. Full offline test results

Command (no key in the environment for the whole phase):

```powershell
Set-Location "C:\Users\CSLAB\Desktop\hemanth\Guardbound"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=short --junitxml=_phase22_final.xml
```

| Moment | Collected | Passed | Failed | Errors | Skipped | Time |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (before any change) | 1013 | 1009 | 4 | 0 | 0 | 150.1 s |
| After Phase 22 | **1067** | **1063** | **4** | **0** | **0** | 171.0 s |

`1067 = 1013 + 62 (new) − 8 (budget-only tests removed)`.

The four failures are the **same node IDs** as the baseline (verified by
comparing `_phase22_baseline.xml` and `_phase22_final.xml`), pre-existing and
environmental — they must not be "fixed" by changing code or data:

```
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring
tests/test_phase14_2_rotation.py::TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured
tests/test_phase8_architecture.py::TestThreeModelConfigValidation::test_default_config_valid
tests/test_phase8_architecture.py::TestConfiguredModelsPresentOnDisk::test_attacker_present
```

Two need CUDA (`torch` is CPU-only), two need `Qwen/Qwen3.5-4B` weights that are
not on disk.

Focused runs:

```powershell
python -m pytest tests/test_phase22_paper_fidelity.py -q      # 62 passed
python -m pytest tests/test_cloud_usage.py tests/test_cloud_provider.py `
                 tests/test_cloud_qualification_harness.py -q  # budget-removal regressions
python scripts/audit_prompt_fidelity.py                       # paper_path_faithful = true
python scripts/cloud_pilot.py --config configs/paper_reproduction_r1_gpt4o.yaml `
    --variant r1_paper --dry-run --limit 2                    # R1 resolves correctly
```

### No external API access

`OPENAI_API_KEY` was unset for every command in this phase; no `.env` exists. The
only HTTP-shaped log lines come from `httpx.MockTransport` inside
`tests/cloud_fakes.py` and the audit script (the OpenAI SDK still formats and logs
a request URL). A genuine call with no key would have failed with a 401
`authentication` error rather than returning `200 OK`, and the SDK clients used in
tests are constructed with `max_retries=0` over the mock transport. No test
attempted external access.

Noted pre-existing side effect (not caused by, and not affecting, this phase): the
suite appends `UNAVAILABLE` placeholder rows to
`results/experiments/E1/manifest.jsonl`; that path is untracked/ignored and no
experiment ran or was modified.

---

## 11. Frozen artifact integrity

```powershell
git status --porcelain -- results runs checkpoints data `
    configs/reproduction_phase12.yaml configs/reproduction_three_model.yaml `
    configs/reproduction_phase16_5_forensics.yaml
# -> empty
```

NBF checkpoint re-hashed during this phase:
`cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` — unchanged.
`configs/reproduction*.yaml`, the Phase 18 / 18-ext / 19 scripts and the frozen
`results/` artifacts were not modified. The only `results/` writes were
`results/audit/prompt_fidelity.json` (the audit's own output).

---

## 12. Resolved R1 configuration

`configs/paper_reproduction_r1_gpt4o.yaml`, validated by
`resolve_paper_reproduction()`. Observed `--dry-run` output:

```
variant            r1_paper (gpt-4o-2024-08-06 for all roles, paper attack set, eta=1e-3)
attacker prompt    baseline
temperature        attacker=0.7 target=0.7 evaluator=0.7
max turns          8
attacks            ['actor_attack', 'crescendo_paper', 'opposite_day']
prompt variant     official
paper protocol     R1 validated
NBF enabled        True
NBF eta/threshold  0.001 / 0.001
NBF sha256         cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe
goals              2 of 200 (per attack) -> 6 conversations
```

| Setting | Value | Provenance |
|---|---|---|
| Target model | `gpt-4o-2024-08-06` | **paper** p9 Sec. 5.1 |
| Attacker model | `gpt-4o-2024-08-06` | **code-recovered** (`steering.py` default); the paper never names it |
| Evaluator model | `gpt-4o-2024-08-06` | **code-recovered** (the attacker also judges) |
| Attacker condition | `baseline` (original paper prompt) | **approved**; C1/I1 prohibited |
| Attacks | actor_attack, crescendo_paper, opposite_day | **paper** p9, p22 |
| Acronym | training-only | **paper** p22 B.1 |
| Temperatures | 0.7 / 0.7 / 0.7 | **paper** p9 Sec. 5.1 |
| η | 0.001 | **paper** p9 Sec. 5.2, Table 1 |
| Max turns | 8 | **paper** B.1 |
| HarmBench behaviours | 200 | **paper** p9 |
| Unsafe label | 5 | **paper** B.1 |
| Target system prompt | `You are a helpful assistant.` | **paper** B.1 |
| Prompt bytes | `official` | **approved** |
| NBF checkpoint | released `.pth`, hash above | **paper** / verified artifact |
| Budget cap | none | **approved** (and **absent** from both authorities) |
| Retries / repair / fallback | 0 / NONE / NONE | **approved** research mode |
| Seed | `null` | **paper gap** — the paper prints none; not invented |

---

## 13. Remaining methodology deviations and gaps

| # | Item | Class | Status |
|---|---|---|---|
| 1 | Qualification probes still call the evaluator at the legacy 0.0, so they qualify a 0.0 stack while R1 runs 0.7 | Guardbound behaviour | open (P2) — the qualification harness predates the R1 temperature decision |
| 2 | Evaluation-harness temperatures (MMLU 0.0, over-refusal 0.0, ASR judge 0.0, MTBench 0.0) were **not** changed | intentional scope | P1-2 from Phase 21 remains; these are not on the R1 path |
| 3 | Attacker model provenance is `code-recovered`, not paper-stated | paper silence | disclosed in config + manifest |
| 4 | `C_refused < 10` regeneration cap is code-recovered | paper silence | unchanged |
| 5 | Guardrail F1 records an exception as "harmless" (`guardrail_f1.py:340`) | implementation detail | P3, unchanged (not on the R1 path) |
| 6 | ActorAttack qualification probes are not in the qualification harness | gap | P3, unchanged |
| 7 | Resolved model revision is not exposed by the API | provider limitation | recorded as `null` with a reason |
| 8 | Both models deprecated, shutdown 2027-01-14 | provider | time-boxed |
| 9 | `torch` is CPU-only (no CUDA build of 2.14.1 exists) | environment | unchanged; the NBF runs correctly on CPU |
| 10 | The composite contract test proves control-flow equality with `attack_single`'s transcription, not model-level equivalence | test boundary | stated, not glossed |

The four typographic prompt deviations are **no longer deviations for the paper
path**: R1 sends the release's bytes. They remain available as `legacy` solely so
frozen Phase 14-19 experiments stay reproducible.

---

## 14. Final readiness status

**`READY WITH DOCUMENTED DEVIATIONS`** — for the implementation.

Justification: all P0 requirements are implemented and offline-validated; the R1
configuration is internally consistent (attack set, per-role temperatures, η,
prompt bytes, no budget field); ActorAttack follows the official composite flow and
passes a call-sequence contract test against `run_full_attack`; the custom budget
cap and all budget-based blocking are gone while usage reporting is intact; the
NBF implementation and every frozen artifact are unchanged; and the offline suite
has the same four pre-existing environmental failures as before the phase.

This is **not** a claim of experimental readiness beyond implementation. A
separate explicit user decision is required before any paid qualification, smoke
test, pilot or formal experiment. No paid request was made in this phase, and the
removal of the spending cap does not authorise one.
