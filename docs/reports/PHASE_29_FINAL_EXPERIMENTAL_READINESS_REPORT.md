# PHASE 29 — FINAL EXPERIMENTAL READINESS REPORT

**Classification (attack-ASR scope): `READY WITH DOCUMENTED DEVIATIONS`**
**Full GPT-4o Table 1: `NOT READY` — MTBench/XSTest `BLOCKED — DATA REQUIRED`**

Implementation and offline verification only. No API call, no credential access, no
qualification, smoke test, pilot or experiment was run. Frozen artifacts are
untouched (`git status --porcelain -- results runs checkpoints data … nbf_original_stuff`
is empty). Nothing here authorises a paid run.

---

## 1. Phase 28 findings addressed

| Phase 28 finding | Sev | Status | Evidence |
|---|---|---|---|
| **A** ASR denominator was valid-completed only (paper: "over all harmful queries") | P1 | **FIXED** | `_cloud_common.aggregate_asr_report` emits `asr_paper` *and* `asr_valid_completed`; the pilot records both plus `inspect` counts (§4) |
| **B** Evaluator temperature 0.7 vs the code's 0.0 | P1 | **RETAINED — DOCUMENTED** | Explicitly out of scope to change (§7 forbids altering 0.7); the manifest states the temperature and the ASR provenance (§7) |
| **C** No baseline variant; invariant guard inert | P1 | **FIXED** | `r1_baseline` (NBF off) + `r1_paper` (NBF on); `assert_invariants` now executes and is recorded (§5) |
| **D** MTBench/XSTest unrunnable, wrong default η | P1 | **PARTIALLY FIXED / BLOCKED** | η default corrected to 1e-3; missing data now a loud `BLOCKED — DATA REQUIRED` with a schema inventory (§9) |
| **E** Zero retries vs the code's 10-attempt loop | P1/P2 | **RESOLVED — RETAINED, DOCUMENTED** | Official semantics inspected and recorded; retained policy proven by tests (§6) |
| **F** Two ASR paths, the second non-official | P2 | **FIXED** | `evaluation.asr_source` is validated for paper configs; `judge_asr.py` stamps `secondary_judge` / `paper_comparable: false` (§7) |

Phase 28's `INCOMPLETE SCOPE` verdict was driven by D and by A/C. A and C are closed;
D remains, so the scope gap persists and is reported as such (§10, §12).

---

## 2. Files changed

| File | Change | Nature |
|---|---|---|
| `scripts/_cloud_common.py` | `aggregate_asr_report()` (dual denominators, per-attack, `None` when undefined); `PAPER_ASR_DENOMINATOR_DEFINITION`, `VALID_ASR_DENOMINATOR_DEFINITION`; `PAPER_CODE_RETRY_POLICY`; `prompt_provenance()` | new pure helpers |
| `scripts/cloud_pilot.py` | per-goal ASR rows; dual-ASR run block; `expected/completed/successful`, failure and invalid counts; `asr_by_attack`; paper-scope flags; paper-scope warning; `prompt_hashes`, `asr`, `linkage` manifest blocks; NBF device/precision; dual-ASR console lines; `_fmt`; corrected `--variant` help | additive |
| `src/guardbound/paper_protocol.py` | `PAPER_ASR_SOURCE` / `SECONDARY_ASR_SOURCE`; paper configs reject any other `evaluation.asr_source`; resolved protocol records the source and its provenance | validation |
| `src/guardbound/evaluation/readiness.py` | **new** — utility-dataset presence/schema inventory, `BLOCKED — DATA REQUIRED` status | new module |
| `scripts/judge_asr.py` | banner; `*.asr_source.json` marker (`secondary_judge`, `paper_comparable: false`) | provenance stamp |
| `scripts/eval_mtbench.py` | default η 5e-4 → **1e-3**; blocks with `BLOCKED — DATA REQUIRED` (exit 2) instead of a `FileNotFoundError` | fidelity + guard |
| `scripts/eval_over_refusal.py` | default η 5e-4 → **1e-3**; same dataset guard for XSTest | fidelity + guard |
| `configs/paper_reproduction_r1_gpt4o.yaml` | `evaluation.asr_source: attack_time_qi_rubric`; `variants.r1_baseline` (NBF off) alongside `r1_paper` | config only |
| `tests/test_phase29_alignment.py` | **new** — 30 tests | tests |

Not touched: attack implementations, NBF architecture/checkpoint/formula/η, prompt
bytes, dataset contents, evaluator temperature, model selection, frozen artifacts.

---

## 3. Exact methodology decisions

| Decision | Value | Authority |
|---|---|---|
| Paper ASR | `successful_goals / expected_goals`; failures stay in the denominator and count as not successful | paper p8 §5.1 ("over all harmful queries") |
| Diagnostic ASR | `successful_goals / valid_completed_goals`; failures and invalid evaluator outcomes excluded | user-approved Phase 24 D-03, kept as a *separate* metric |
| Undefined ASR | `None` when a denominator is zero — never `0.0` | project rule (no fabricated 0 %) |
| Invalid evaluator outcome | counted as not successful in both denominators | no usable verdict exists |
| Paper-scope labelling | `covers_paper_scope = (selected goals == dataset.num_goals)`; `comparable_to_paper = complete ∧ paper_scope ∧ no failures ∧ no invalid` | prevents a `--limit 5` run reading as a 200-goal result |
| Arms | `r1_baseline` = NBF off, `r1_paper` = NBF on (η = 1e-3); identical everything else | Table 1's before/after columns |
| ASR source | `attack_time_qi_rubric` only; `secondary_judge` rejected | paper §5.1 + Phase 28 E5 |
| Retry policy | 1 attempt, SDK `max_retries=0` (official loop inspected, not adopted) | §6 |
| Evaluator temperature | 0.7, unchanged | paper §5.1; §7 forbids changing it |
| Utility margin | 1e-3 | paper §5.2 / Table 1 |

---

## 4. Dual ASR validation

### 4.1 Unit tests (synthetic records, zero failures, one, five, ten, twenty)

All ten `test_dual_asr_*` cases pass. Independent probe output (52 successes of
Crescendo's 200 — the paper's 0.260 NBF figure):

| *f* | `asr_paper` | `asr_valid_completed` | inflation | paper denom | valid denom |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.2600 | 0.2600 | 0.00 % | 200 | 200 |
| 1 | 0.2600 | 0.2613 | +0.50 % | 200 | 199 |
| 5 | 0.2600 | 0.2667 | +2.56 % | 200 | 195 |
| 10 | 0.2600 | 0.2737 | +5.26 % | 200 | 190 |
| 20 | 0.2600 | 0.2889 | +11.11 % | 200 | 180 |

`asr_paper` is invariant to failures — exactly the property the paper's definition
requires. Additional cases proven: a zero denominator yields `None` (never 0 %), an
all-failed run yields `asr_paper = 0.0` with `asr_valid_completed = None`, an invalid
evaluator row can never contribute to the numerator even if it is flagged
`success: true`, per-attack counts sum to the overall block, and a row for an
unplanned attack raises `ValueError`.

### 4.2 End-to-end evidence (offline mock, `r1_baseline`, `--limit 2`)

```
run counts    expected=6 completed=4 valid=4 infrastructure_failures=2 invalid=0 successful=0
run asr       asr_paper=0.0 (denom 6)  asr_valid_completed=0.0 (denom 4)
run flags     complete=True covers_paper_scope=False comparable_to_paper=False paper_expected_goals=600
by_attack     actor_attack=(2 expected, 0 success, asr_paper 0.0, asr_valid None)
              crescendo_paper=(2, 0, 0.0, 0.0)   opposite_day=(2, 0, 0.0, 0.0)
```

`actor_attack` has **no valid goal** in this smoke run, and the report says
`asr_valid_completed: None` rather than 0.0 — the undefined-denominator rule working
live. (The two failures are the offline mock's ActorAttack limitation, §11.)

---

## 5. Baseline-versus-defence configuration comparison

Both arms were resolved through the **real CLI** (`--dry-run`, no client, no spend):

| | `r1_baseline` | `r1_paper` |
|---|---|---|
| variant resolved | ✓ | ✓ |
| invariants checked | `['models','temperature','dataset','attacks','prompt_variant','paper_reproduction']` | same |
| models | gpt-4o-2024-08-06 × 3 | identical |
| temperatures | 0.7 / 0.7 / 0.7 | identical |
| attacks | `[actor_attack, crescendo_paper, opposite_day]` | identical |
| prompt variant | `official` | identical |
| turn limit | 8 | identical |
| goals planned | 200 × 3 = 600 | identical |
| **NBF** | **disabled** | **enabled, η = 0.001 / threshold 0.001** |
| protocol | R1 validated | R1 validated |

* `assert_invariants` now returns the six declared invariants (with one variant it
  returned `[]`, so the guard was inert — Phase 28 C). A variant that overrides
  `temperature` raises `ValueError: Variants must differ only in NBF enablement`
  (tested).
* The two variant blocks contain only `description` and `nbf`; `nbf_settings`
  differs in exactly `{"enabled"}` (tested).
* **Same dataset bytes**: `ac789de8859e…d014eb` for both arms (one `dataset:` block,
  outside `variants`).
* **Same prompt bytes**: `prompt_hashes()` is configuration-level —
  Crescendo `7df65dbd…`, Opposite-day `dfd5b4da…`, Acronym `f3afb6a9…`, rubric
  `146589a4…`; the rubric differs between legacy and official variants and the R1
  config pins `official`.
* **Linkage**: both manifests carry
  `comparison_group = "paper_reproduction_r1_gpt4o|ac789de8…|official|8"` and
  `arms = [r1_baseline, r1_paper]`, so the pair is identifiable after the fact.

---

## 6. Retry-policy fidelity

### 6.1 What the author's code does (`attacks/utils/generate.py:129-151`)

| Property | Value |
|---|---|
| Attempts | **10** (numbered `ind_ = 0…9`) |
| Retried errors | **any** exception (`except Exception as e`), including auth/parse/network |
| Backoff | `time.sleep(2*ind_ + 1)` → 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 s |
| After the final attempt | returns `{}` |
| Consequence | callers index the result (`data['generatedQuestion']`, `data['score']`) → `KeyError` → the **whole run aborts**; the failed goal never reaches aggregation |
| Paper | says nothing about retries |

### 6.2 What Guardbound does, and why it is retained

`OpenAIChatLLM` is built with `max_retries=0` (`openai_client.py:220-224`), the config
declares `experiment.protocol.retries: 0`, and a provider error is raised as a typed
exception that the pilot records as `infrastructure_failure` (goal counted in
`asr_paper`, excluded from `asr_valid_completed`) and the run continues.

**Decision: retain, document** (§3's "otherwise" branch). Reproducing the official
loop is not separable from its failure semantics:

1. Adopting `{}`-on-exhaustion means adopting a `KeyError` that **destroys every
   remaining goal** of a paid 600-conversation run — a regression against the
   approved Phase 13/17 resilience work, and it would make `asr_paper` depend on
   *where* in the loop the failure occurred.
2. Retaining retries but keeping typed failures is a hybrid neither the paper nor the
   code specifies — a new experimental condition.
3. The paper is silent, so no fidelity requirement forces either choice; the official
   loop is code-recovered behaviour, not methodology.

The policy is recorded **in every manifest** under `asr.retry_policy` (both the
official semantics and the applied one) and in `run.retry_policy`, so it can never be
silently assumed.

### 6.3 Tests

| Test | Proves |
|---|---|
| `test_documented_official_retry_semantics` | the record matches the inspection (10 attempts, the exact backoff list, any-exception, `KeyError` outcome) |
| `test_sdk_client_is_constructed_with_zero_hidden_retries` | the SDK receives `max_retries=0` (no invisible retry layer) |
| `test_transient_failure_makes_exactly_one_attempt` | one failing call ⇒ exactly **1** attempt |
| `test_successful_call_is_one_attempt_with_no_retry_path` | a successful call is also exactly 1 attempt |
| `test_exhausted_attempts_never_become_a_verdict` | "exhausted" (the single attempt) becomes an infrastructure failure, not a score; `asr_paper = 0.0`, `asr_valid_completed = None` |

The brief's "successful retry" state has no analogue under a one-attempt policy; the
equivalent property — the call that *would* have been a retry is the first and only
attempt — is what the second test asserts. This is stated rather than papered over.

---

## 7. Evaluator-path verification

* **Declaration**: `configs/paper_reproduction_r1_gpt4o.yaml` →
  `evaluation.asr_source: attack_time_qi_rubric`.
* **Enforcement**: `paper_protocol.resolve_paper_reproduction` raises
  `PaperProtocolError` for any other value (tested with `secondary_judge`); the
  resolved protocol records `asr_source` and its provenance
  ("Qi et al. (2023) rubric via Ren et al. (2024), applied at attack time").
* **Manifest**: `asr.source`, `asr.source_provenance`, `asr.evaluator_temperature`,
  `asr.prompt_variant`, `asr.denominators` (both definitions), `asr.retry_policy`.
* **Secondary judge cannot impersonate R1**: `judge_asr.py` prints
  `ASR source: SECONDARY_JUDGE (diagnostic) — NOT the R1 paper path` and writes
  `<out>.asr_source.json` with `asr_source: secondary_judge`,
  `paper_comparable: false`, `paper_asr_source: attack_time_qi_rubric` (tested).
* **Preserved**: official rubric bytes (`146589a4…`, unchanged from Phase 22) and
  evaluator temperature 0.7 — neither was touched.

---

## 8. Manifest verification

Fields added (all present in the end-to-end manifest; existing fields retained):

| Requirement | Manifest path | Observed |
|---|---|---|
| Configuration SHA-256 | `config.sha256` (existing) | `33a0fa5705b0d995…` = file bytes |
| Explicit variant name | `variant`, `variant_description`, `linkage.variant` | `r1_baseline` |
| Dataset SHA-256 | `dataset.sha256` | `ac789de8…d014eb` |
| Prompt SHA-256 | `prompt_hashes.<prompt>.variants.{legacy,official}` | 4 prompts × 2 variants |
| NBF checkpoint SHA-256 | `nbf.sha256` (existing) | pinned value; `nbf.enabled: false` for the baseline arm |
| Model identifiers | `model_ids`, `models`, `model_identity` (existing) | gpt-4o-2024-08-06 × 3 |
| Configured temperatures | `temperature` (existing) | 0.7 / 0.7 / 0.7 |
| NBF device + precision | `nbf.device`, `nbf.precision` | `cuda`/`cpu` + `float32` when enabled |
| Expected / completed goals | `run.expected_goals`, `run.completed_goals` (+ existing `goals_expected`, `goals_completed`) | 6 / 4 in the smoke run |
| Paper ASR denominator | `run.asr_paper_denominator` | 6 (6 = 2 goals × 3 attacks) |
| Valid-completed denominator | `run.asr_valid_completed_denominator` (+ preserved `run.asr_denominator`) | 4 |
| Infrastructure failures | `run.infrastructure_failure_count` (+ `failed_goals`) | 2 |
| Invalid evaluator responses | `run.invalid_evaluator_count` (+ `invalid_evaluator_goals`) | 0 |
| Baseline/defence linkage | `linkage.comparison_group`, `linkage.arms`, `linkage.nbf_enabled` | shared string, both arms |
| ASR provenance | `asr.source`, `asr.retry_policy`, `asr.denominators` | as above |
| Paper-scope framing | `run.covers_paper_scope`, `run.paper_expected_goals`, `run.comparable_to_paper` | `false` / `600` / `false` |

Preserved unchanged: `paper`, `protocol`, `attacker_prompt`, `paper_protocol`,
`dataset.goal_id_hash`, `config.path`, `rate_card`, `environment`, `git_commit`,
`git_dirty`, `git_status_sha256`, `usage_summary`, `asr_denominator` +
`asr_denominator_definition`, and the whole usage telemetry (21 fields per call).

---

## 9. MTBench / XSTest readiness

| Component | Requirement | Implementation | Data | Status |
|---|---|---|---|---|
| **MTBench** | 1–10 score; on a positive barrier score replace the answer with the exact refusal string, then judge (B.1 p22) | `evaluation/mtbench.py`: substitution via `MTBENCH_REFUSAL_STRING`, judge at `temperature=0.0`, target at 0.7; schema `{question, reference?, turn2_question?, category?}` | absent (`data/raw/mtbench.jsonl`) | **`BLOCKED — DATA REQUIRED`** |
| **XSTest** | refused queries / all (§5.1) | `evaluation/over_refusal.py`: shared regex refusal detector; schema `{prompt|text, type?}` | absent (`data/raw/xstest.jsonl`) | **`BLOCKED — DATA REQUIRED`** |

Changes made: both CLIs now default to **η = 1e-3** (was 5e-4) and both refuse to run
with a printed `BLOCKED — DATA REQUIRED` and exit code 2 when the dataset is missing or
schema-invalid — no download, no fabrication, no substitution. The readiness inventory
(`guardbound.evaluation.readiness`) reports per-component path, existence, row count,
missing fields and status; on this checkout it returns
`status: BLOCKED — DATA REQUIRED`, `blocked: [mtbench, xstest]`, `table1_complete: false`
(tested for absence, validity and a schema gap).

Explicitly unspecified by the paper (reported, not guessed): the verbatim MTBench judge
prompt and judge model; how the turn-2 question is sourced; the exact XSTest refusal
detection algorithm. The author's release ships no utility data either, so these
metrics cannot be regenerated from the release alone.

**No claim of Table 1 readiness is made.**

---

## 10. Offline test comparison

```powershell
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase29_run1.xml
```

| | Phase 27 baseline | Phase 29 |
|---|---:|---:|
| Collected | 1189 | **1219** (+30) |
| Passed | 1185 | **1215** |
| Failed | 4 | **4** |
| Errors | 0 | **0** |
| Skipped | 0 | **0** |
| Time | 128.2 s | 127.9 s |

The four failing node IDs are identical to Phase 27 (two CUDA-gated, two requiring
`Qwen/Qwen3.5-4B` weights). No test was modified, skipped or deleted; the 30 new
tests live in `tests/test_phase29_alignment.py` (all pass).

---

## 11. Remaining discrepancies

| # | Item | Sev | Status / note |
|---|---|---|---|
| 1 | Evaluator temperature 0.7 (paper) vs 0.0 (author's code) | P1 | **Deliberately unchanged** — §7 forbids changing it; the deviation is recorded in every manifest (`asr.evaluator_temperature`) and discussed in Phase 28 §10 |
| 2 | Retry policy 1 attempt vs the code's 10 | P1/P2 | **Retained and documented**; official semantics recorded in `run.retry_policy`. A sensitivity choice must be made before the paid run, not after |
| 3 | MTBench / XSTest data missing | P1 (table) | `BLOCKED — DATA REQUIRED`; attack-ASR results must not be presented as Table 1 |
| 4 | `--offline-mock` cannot drive ActorAttack's pre-attack JSON contract | P3 | **Pre-existing harness gap**, first exposed because this was the first smoke run covering all three attacks (the earlier mock artifact contained only `crescendo_paper`). It surfaces as two accounted `infrastructure_failure` rows, which is the correct behaviour; the ActorAttack path itself is covered by the Phase 24 scenario tests. Not fixed — out of Phase 29 scope |
| 5 | No seed; single run | P2 | Unchanged (`seed: null`); both codebases sample at 0.7. No confidence intervals in the paper either |
| 6 | Served model revision not exposed; `gpt-4o-2024-08-06` deprecation announced for 2027-01-14 | P2/P3 | Manifest records the execution date; the CLI warns |
| 7 | `--limit` defaults to 5 | P2 | A paper run must pass `--limit 200`; a smaller run now prints a warning and is labelled `covers_paper_scope: false` / `comparable_to_paper: false` |
| 8 | Per-call wire settings (temperature/top_p/response_format) are not in the usage rows | P3 | Unchanged from Phase 28 (recommendation C3) |
| 9 | Crescendo/Opposite-day execute 7 target calls where the paper says "at most 8" | P3 | Unchanged — faithful to the author's loop |
| 10 | `judge_asr.py` still exists as a diagnostic | P3 | Now stamped and rejected by paper-config validation; it cannot be the R1 path silently |

---

## 12. Final classification

### Attack-ASR reproduction: `READY WITH DOCUMENTED DEVIATIONS`

| Acceptance dimension | State |
|---|---|
| Dual ASR implemented, paper denominator primary | ✓ (§4) |
| Failures never substituted; never silently excluded from `asr_paper` | ✓ tests + live run |
| Incomplete / limited runs labelled, not presented as 200-goal results | ✓ `covers_paper_scope`, `comparable_to_paper` |
| Baseline and NBF arms declared, invariant-checked, identical except NBF | ✓ (§5) |
| One official ASR path, enforced | ✓ (§7) |
| Manifest complete enough to reproduce inputs | ✓ (§8) |
| Methodology (attacks, NBF, prompts, dataset, temperatures, models) unchanged | ✓ |
| Deviations that remain | evaluator temperature (paper-driven), retry policy (retained), no seed |

### Full GPT-4o Table 1: `NOT READY`

MTBench and XSTest are `BLOCKED — DATA REQUIRED`. Three of the five table columns
(ActorAttack, Crescendo, Opposite-day ASR) are implemented and now correctly
reported; the two utility columns cannot be produced, and the paper's utility judge
details are unspecified. A successful attack-ASR run would be a **partial** row, and
must be reported as such.

### What is still required before a paid run

1. Decide the retry stance **before** execution (keep 1 attempt and report failures,
   or adopt a bounded policy and record it) — it changes `asr_paper`'s failure count,
   not the method.
2. Run both arms at `--limit 200` (600 conversations each) and report both
   denominators, with `comparable_to_paper` printed next to each number.
3. Obtain MTBench/XSTest data before any claim about Table 1's utility columns.
4. Explicit authorisation — none of the above authorises a paid request, and
   `READY WITH DOCUMENTED DEVIATIONS` is a readiness label, not a permission.
