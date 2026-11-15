# PHASE 31 — RETRY SEMANTICS CORRECTION & FINAL REPRODUCIBILITY AUDIT

**Attack-ASR: `READY WITH DOCUMENTED DEVIATIONS`**
**Retry alignment: `ALIGNED WITH DOCUMENTED DEVIATIONS`**
**MTBench: `BLOCKED — DATA REQUIRED`**
**XSTest: `BLOCKED — DATA REQUIRED`**
**Full GPT-4o Table 1: `NOT READY`**

Strictly offline implementation, verification and read-only research. **No dataset was
downloaded, cloned, materialised, copied or modified; no paid API call, credential read,
qualification, smoke test, pilot or experiment occurred.** Every LLM call in this phase went
to scripted fakes or the offline mock transport (`--offline-mock`), with injected sleep and
an injected clock.

Repository state: `main` @ `89ed03f` + uncommitted work (the Phase 22–30 tree).
Full suite: **1277 collected / 1273 passed / 4 failed / 0 errors / 0 skipped** — the four
failing node IDs are byte-identical to Phase 30's.

---

## 1. Executive summary

Phase 30's retry record contained one material error and one omission.

| # | Item | Phase 30 said | Source-verified truth (Phase 31) |
|---|---|---|---|
| 1 | Waits on **exhaustion** | nine (no wait after the tenth attempt) | **ten**: the 19 s wait happens *after* the tenth failure, then `return {}` → **100 s** |
| 2 | First-attempt API failure | "retries any exception" | **not retried**: the handler dereferences `response` before sleeping, so a first-attempt `create()` failure raises `UnboundLocalError` out of `generate()` |

The success path was correct in Phase 30 (10 attempts, nine waits, 81 s). The schedule is not
symmetric: **81 s is the maximum a *successful* call can schedule; 100 s is what the reference
schedules before giving up.**

Guardbound's applied policy is unchanged in behaviour (10 attempts, nine waits, 1–17 s, 81 s,
no post-final wait, typed `RetriesExhausted`, non-retryable categories fail fast). What changed
is that the *record* is now correct, the two schedules are reported separately, and every
remaining difference is enumerated instead of being described as exact equivalence.

Also delivered: immutable commit + git-blob pins for both utility datasets
(`lm-sys/FastChat` @ `b494d0c6…`, `paul-rottger/xstest` @ `475f10bf…`), recorded in every
manifest, with `PAPER VERSION UNVERIFIED` preserved for both.

| Item | Outcome |
|---|---|
| Reference sleeps inside `except` | **confirmed** (line 151, indent 16; `return {}` at line 152, indent 8) |
| Reference waits: success / exhaustion | **9 (81 s) / 10 (100 s)** |
| Guardbound waits: success / exhaustion | **9 (81 s) / 9 (81 s)** — omission documented |
| New retry tests | **33**, all passing; 12 of them execute the reference's own bytes |
| ASR worked example | **0.2500 / 0.2703** (200 / 50 / 10 / 5 / 185), unchanged |
| MTBench upstream | repo + tag `v0.2.36` + commit + 2 dataset blobs (`464e2c22…`, `8b66387a…`) pinned |
| XSTest upstream | repo (no tags) + commit + blob (`ad38204d…`) pinned |
| Dataset versions | `PAPER VERSION UNVERIFIED` (both) |
| Frozen artifacts | byte-identical (HarmBench, NBF checkpoint, reference source, official prompts) |

---

## 2. Exact original retry source and line references

File:
`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/generate.py`
(SHA-256 `8b2dcd8520def27f62a5f4a96766d9ef07b3c462d6898adb3b0ce067e07dcb29`, 5 455 bytes).

The OpenAI branch, verbatim (line numbers from the file):

```python
127     else:
128         # openai gpt 
129         for ind_ in range(10):
130             try:
131                 if 'o1' in model or 'o3' in model:
132                     response = client.chat.completions.create(
133                         model=model,
134                         messages=messages
135                     )
136                 else:
137                     response = client.chat.completions.create(
138                         model=model,
139                         messages=messages,
140                         temperature=temperature,
141                         response_format={"type": "text"} if not json_format else {"type": "json_object"}
142                     )
143
144                 if json_format:
145                     return json.loads(response.choices[0].message.content)
146
147                 return response.choices[0].message.content
148             except Exception as e:
149                 print("Error in generate: ", e)
150                 print("generate: ", response.choices[0].message.content)
151                 time.sleep(2*ind_+1)
152         return {}
```

Verified control flow (AST, not eyeballed):

| Element | Line | Indent | Structural fact |
|---|---:|---:|---|
| `for ind_ in range(10):` | 129 | 8 | loop over attempts |
| `try:` | 130 | 12 | body of the loop |
| `return response…content` | 147 | 16 | inside `try` |
| `except Exception as e:` | 148 | 12 | **sibling** of `try` |
| `print("generate: ", response…content)` | 150 | 16 | inside handler |
| `time.sleep(2 * ind_ + 1)` | 151 | 16 | **inside handler** (last statement) |
| `return {}` | 152 | 8 | **after the loop**, not inside it |

The handler contains exactly three statements (two prints and the sleep); `try` has no `else`
and no `finally`. The sleep is therefore executed after **every** failed attempt, including the
tenth, and never after a successful one (the `try` body returns).

`nbf_original_stuff/` is git-ignored (`.gitignore:56`), so integrity is tracked by SHA-256, not
by commit.

---

## 3. Original retry semantics (successful and exhausted calls)

### 3.1 Successful calls

| Outcome | Attempts | Waits | Scheduled backoff |
|---|---:|---:|---:|
| First attempt succeeds | 1 | 0 | 0 s |
| Second attempt succeeds | 2 | 1 | 1 s |
| Fifth attempt succeeds | 5 | 4 | 16 s |
| Tenth attempt succeeds | 10 | 9 | **81 s** |

### 3.2 Exhaustion

| Outcome | Attempts | Waits | Scheduled backoff | Return |
|---|---:|---:|---:|---|
| All ten attempts fail | 10 | **10** | **100 s** | `{}` |

The tenth wait (19 s) occurs *after* the tenth failure. The table in the Phase 31 brief is
therefore correct, and Phase 30's "none after the tenth attempt" was wrong.

### 3.3 What is retried, and the first-attempt asymmetry

* **JSON parsing failures are retried.** `json.loads` is inside the `try`; a malformed reply
  is caught and the loop continues.
* **API and response-processing exceptions share one handler**, so they look identical in
  category terms (bare `except Exception`).
* **But a first-attempt API failure is not retried.** Line 150 dereferences `response` inside
  the handler. On attempt 1 `response` is unbound, so the handler itself raises
  `UnboundLocalError`, which escapes `generate()`. Consequence: the loop only ever retries
  *after an attempt that produced a `response`*. Post-response failures (JSON, content access)
  retry normally from attempt 1; transport failures retry only from attempt 2 onwards.

Both facts were proven by executing the reference's own function body (see §6).

### 3.4 Downstream consequence of `{}`

`crescendomation/run.py:21` reads `data['generatedQuestion']` and
`evaluate_with_rubric.py:32` reads `data["score"]`, so `{}` becomes a `KeyError` and aborts the
run — which is why Guardbound raises a typed error instead (§4).

---

## 4. Guardbound applied retry semantics

`src/guardbound/llm/openai_client.py`:

| Property | Value |
|---|---|
| Policy name | `official_retry_schedule + Guardbound_typed_failure_handling` |
| Policy version | `phase31` |
| Attempts | 10 (`retry_attempts=10`) |
| Applied waits | `1, 3, 5, 7, 9, 11, 13, 15, 17` s — **9 waits, 81 s** |
| Wait after the tenth (final) failure | **none** (`APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE = False`) |
| Exhaustion | typed `RetriesExhausted` (`attempts`, `retries`, `backoff_seconds`, `total_scheduled_backoff_seconds`, `error_categories`, `exhausted=True`) |
| Non-retryable categories | `authentication`, `permission`, `invalid_request`, `context_length`, `content_filter` → one attempt |
| JSON parse failure | not retried; surfaced as `outcome="json_parse_error"`, one usage row |
| First-attempt transport failure | retried (no unbound-variable dereference) |
| SDK hidden retries | `max_retries=0` |
| Sleep injection | `sleep_fn=` (tests never wait) |
| Usage schema | unchanged — one row per logical call, attempts aggregated in retry telemetry |

Constants are now derived from the source expression rather than typed by hand:

```python
OFFICIAL_RETRY_ATTEMPTS = 10
OFFICIAL_RETRY_BACKOFF_SECONDS = tuple(2*i + 1 for i in range(10))   # 1 … 19
OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS = OFFICIAL_RETRY_BACKOFF_SECONDS[:-1]
OFFICIAL_RETRY_EXHAUSTION_BACKOFF_SECONDS = OFFICIAL_RETRY_BACKOFF_SECONDS
OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS = 81            # preserved meaning: success path
OFFICIAL_RETRY_EXHAUSTION_TOTAL_BACKOFF_SECONDS = 100
APPLIED_RETRY_BACKOFF_SECONDS = OFFICIAL_RETRY_SUCCESS_BACKOFF_SECONDS
APPLIED_RETRY_SLEEPS_AFTER_FINAL_FAILURE = False
```

`retry_policy_record()` (single source of truth, imported by `scripts/_cloud_common.py` so the
manifest cannot drift from the client) publishes both halves.

---

## 5. All differences between the reference and Guardbound

| # | Difference | Rationale | Classification |
|---|---|---|---|
| 1 | Non-retryable categories fail after 1 attempt (reference retries them 10×) | retrying a 401/403/400/context/filter error cannot change the outcome; it burns 81 s and real spend | documented deviation |
| 2 | Exhaustion raises typed `RetriesExhausted` (reference returns `{}`) | `{}` is indistinguishable from a reply at the call site and aborts the run with `KeyError`; Guardbound records an infrastructure failure and continues | documented deviation |
| 3 | No wait after the final failed attempt (reference waits 19 s, then `return {}`) | the wait changes no observable outcome, no cost and no transcript; it only delays the typed failure by 19 s. **The 19 s is still reported** as the official exhaustion total (100 s) next to the applied 81 s | documented deviation |
| 4 | JSON parse failures are not regenerated (reference re-samples) | one logical call = one set of sampling draws at 0.7; silent regeneration would add unlogged draws. A malformed reply is surfaced as a typed `json_parse_error` outcome | documented deviation |
| 5 | A first-attempt transport failure is retried (reference aborts with `UnboundLocalError`) | the reference's behaviour is a bug, not a policy; reproducing it would lose the attempt budget whenever the first call fails | reliability improvement |
| 6 | `RetriesExhausted.category` = last attempt's category; with `retry_attempts=1` the classified error is raised, not `RetriesExhausted` | keeps the failure taxonomy stable and truthful (no retry was exhausted when no retry was configured) | error-shape choice |

The applied policy is **not** described as exact retry equivalence. `retry_policy_record()`
records `"status": "ALIGNED WITH DOCUMENTED DEVIATIONS"` and lists the differences in
`differences` (5 entries).

Preserved Phase 30 behaviour: typed infrastructure failures, goal-level failure accounting,
continuation of subsequent goals, explicit retry telemetry, no fabricated evaluator verdicts,
no duplicate goal-level ASR records.

---

## 6. Retry test results and numerical backoff validation

`tests/test_phase31_retry_semantics.py` → **33 passed** (0.43 s).

Eight of them work directly on the **reference itself** — seven execute the compiled function
body, one asserts its AST structure. The executing tests extract the `generate` `FunctionDef`
from the frozen file with `ast.get_source_segment`, compile it with
`{"json": json, "time": <recording stub>}`, and drive it with a scripted fake client. `torch`
and `transformers` are never imported, so no GPU or model weights are involved.

| Scenario | Reference: calls | Reference: waits | Reference: Σ | Guardbound: calls | Guardbound: waits | Guardbound: Σ |
|---|---:|---|---:|---:|---|---:|
| Success on attempt 1 | 1 | `[]` | 0 s | 1 | `[]` | 0 s |
| Success on attempt 2 | 2 | `[1]` | 1 s | 2 | `[1]` | 1 s |
| Success on attempt 5 | 5 | `[1,3,5,7]` | 16 s | 5 | `[1,3,5,7]` | 16 s |
| Success on attempt 10 | 10 | `[1..17]` | **81 s** | 10 | `[1..17]` | **81 s** |
| All ten fail | 10 | `[1..19]` | **100 s** → `{}` | 10 | `[1..17]` | **81 s** → `RetriesExhausted` |
| Eleventh call scripted | never made | — | — | never made | — | — |
| First-attempt transport error | 1 | `[]` | 0 s → `UnboundLocalError` | retries (≤ 10) | — | — |

The reference rows for "attempt 2/5/10" were produced with `json_format=True` and invalid JSON
first, because a *transport* failure on attempt 1 aborts the reference function (§3.3); this is
asserted explicitly.

Requirement coverage (all offline, injected sleep):

| # | Requirement | Test |
|---:|---|---|
| 1–4 | success on attempts 1, 2, 5, 10 | `test_reference_*` + `test_guardbound_success_on_a_given_attempt[0,1,4,9]` |
| 5 | all ten fail | `test_reference_exhaustion_waits_after_the_tenth_failure`, `test_guardbound_exhaustion_schedules_no_final_wait` |
| 6 | successful tenth ⇒ exactly nine waits | `test_reference_tenth_attempt_succeeds_with_nine_waits`, `test_guardbound_success_on_a_given_attempt[9]` |
| 7 | failed tenth ⇒ verified exhaustion policy | `test_guardbound_exhaustion_schedules_no_final_wait` |
| 8 | no eleventh API request | `test_reference_never_makes_an_eleventh_call`, exhaustion test (11-item script) |
| 9 | no sleep after a success | `test_guardbound_never_sleeps_after_a_successful_attempt` |
| 10 | cumulative backoff | `sum(clock.waits)` assertions in every row above |
| 11 | interrupted sleeps | `test_guardbound_interrupted_sleep_is_recorded_not_hidden` |
| 12 | retry telemetry | `test_guardbound_success_on_a_given_attempt`, `..._exhaustion_...` |
| 13 | manifest records | `test_manifest_policy_record_separates_official_from_applied`, `test_failed_goal_...` (manifest block) |
| 14 | per-goal failure accounting | `test_failed_goal_does_not_abort_the_run_or_duplicate_rows` |
| 15 | no duplicate ASR rows | same (3 planned goals → 3 unique rows) |
| 16 | non-retryable fail fast | `test_guardbound_non_retryable_failures_fail_fast[5]` |
| 17 | SDK hidden retries disabled | `test_guardbound_sdk_hidden_retries_remain_disabled` |
| 18 | failed goal does not abort the run | `test_failed_goal_does_not_abort_the_run_or_duplicate_rows` |

**Scheduled vs elapsed are measured separately, never conflated:**

* with a fake clock advanced by each sleep: scheduled = **81.0 s**, elapsed ≈ **81.0 s**;
* with a no-op sleep: scheduled = **9.0 s** for 3 retries, elapsed ≈ **0 s**;
* interrupted on the third wait: scheduled = **9 s** (1+3+5), elapsed = **4 s** (1+3), 3 API
  calls, the interrupted attempt is not repeated.

---

## 7. Retry telemetry and manifest verification

Per-call record (`last_request_stats["retry"]`, and therefore every goal-level failure row):

```
attempts, retries, scheduled_backoff_seconds, error_categories,          # Phase 30 keys
policy, policy_version, max_attempts, actual_attempts, backoff_schedule,
elapsed_backoff_seconds, succeeded, exhausted, terminal_category,
official_exhaustion_total_backoff_seconds, sleeps_after_final_failed_attempt
```

Per-role cumulative block (`retry_summary()`, written to `retry_telemetry.<role>`):

```
policy, policy_version, model, role, max_attempts, actual_attempts,
backoff_schedule, max_scheduled_backoff_seconds,
official_success_total_backoff_seconds, official_exhaustion_total_backoff_seconds,
applied_exhaustion_total_backoff_seconds, sleeps_after_final_failed_attempt,
calls, attempts, retries, scheduled_backoff_seconds, elapsed_backoff_seconds,
successful_calls, exhausted_calls, non_retryable_calls, by_category, error_categories
```

Per-goal delta (`retry_delta`) carries the cumulative counters plus `actual_attempts` and
`by_category`; a goal that made no calls contributes nothing (no fabricated row).

Observed in a real offline-mock manifest produced in this phase:

| Requirement | Manifest path | Observed |
|---|---|---|
| success on attempt 10 vs failure after 10 | `retry_policy.official_implementation.success_path_total_backoff_seconds` / `…exhaustion_path_total_backoff_seconds` | `81.0` / `100.0` |
| official schedule | `…official_implementation.backoff_seconds` | `[1,3,5,7,9,11,13,15,17,19]` |
| official final wait | `…official_implementation.sleeps_after_final_failed_attempt` | `true` |
| applied schedule | `…applied_guardbound.backoff_seconds` | `[1,3,5,7,9,11,13,15,17]` |
| applied final wait | `…applied_guardbound.sleeps_after_final_failed_attempt` | `false` |
| difference list | `retry_policy.differences` | 5 entries |
| status | `retry_policy.status` | `ALIGNED WITH DOCUMENTED DEVIATIONS` |
| per-role official vs applied exhaustion | `retry_telemetry.evaluator.…` | `100.0` / `81.0` |
| permanent failure | `non_retryable_calls`, `exhausted_calls` | `1` / `0` |
| interrupted retry | `scheduled_backoff_seconds` > `elapsed_backoff_seconds` | tested |
| error vocabulary | `error_categories`, `by_category` | present |

Historical fields preserved: `attempts`, `retries`, `scheduled_backoff_seconds`,
`error_categories`, `policy`, `max_attempts`, `backoff_schedule`,
`max_scheduled_backoff_seconds`, `asr.retry_policy`, `run.retry_policy`, and the 21-field usage
record (untouched).

---

## 8. ASR denominator regression validation

Definitions unchanged from Phase 29/30:

* `asr_paper = successful_goals / expected_goals` — infrastructure failures and invalid
  evaluator outcomes remain in the denominator, counted as **not successful**;
* `asr_valid_completed = successful_goals / valid_completed_goals` — diagnostic only;
* zero denominator → `None`, never `0.0`;
* counts reported separately: `infrastructure_failure_count`, `invalid_evaluator_count`.

Worked example re-executed (`test_asr_worked_example_is_unchanged`):

| Quantity | Value |
|---|---:|
| expected goals | 200 |
| successful | 50 |
| infrastructure failures | 10 |
| invalid evaluator outcomes | 5 |
| valid completed | 185 |
| **paper ASR** | **0.2500** |
| **diagnostic ASR** | **0.2703** |

Zero-denominator case: `asr_paper is None` and `asr_valid_completed is None`.

End-to-end (offline mock, 1 failed goal of 3): `asr_paper_denominator = 3`,
`infrastructure_failure_count = 1`, `asr_paper = 0.0 (0/3)`, `asr_valid_completed = 0.0 (0/2)`
— the failure stays in the paper denominator and is excluded from the diagnostic one.

---

## 9. MTBench immutable upstream revision and provenance evidence

| Field | Value |
|---|---|
| Repository | `https://github.com/lm-sys/FastChat` |
| Tag | **`v0.2.36`** → commit `b21d0f780ca4472a13714262a0790f2ee1ade659` |
| Dataset commit (pinned) | **`b494d0c6b4e7935f1764f8439e75da3e66beccc7`** — “Release MT-bench code (#1722)”, 2023-06-16; also the last commit touching both dataset files |
| `fastchat/llm_judge/data/mt_bench/question.jsonl` | git blob **`464e2c22f7a021a0f9a584d232338a9a17c40066`**, 48 929 bytes — identical at `v0.2.36` |
| `fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl` | git blob **`8b66387a3129cbb117178fe515491df553dd52df`**, 51 109 bytes — identical at `v0.2.36` |
| `fastchat/llm_judge/gen_model_answer.py` | blob `5468dce0688933d3ccae6f3c632676d9905ff28d` |
| `fastchat/llm_judge/gen_judgment.py` | blob `3d952c8b09459f64397cb23f2f56e121f024b0a4` |
| `fastchat/llm_judge/common.py` | blob `fc223573cbf3ca2633dc022e50dca1e1d180a95f` (last modified 2024-02-04 @ `81785d7e…`, *later* than the dataset commit — judge code is not pinned by the dataset commit) |
| Hash kind | git blob SHA-1 (content-addressed, published by the GitHub API) |
| SHA-256 | **`None`** — computing it requires the bytes, which would mean downloading the dataset |
| Paper version | **`PAPER VERSION UNVERIFIED`** |

Upstream behaviour verified from source code (not inferred):

* `gen_model_answer.py` iterates `question["turns"]` and picks
  `temperature = temperature_config[question["category"]]`;
* `common.py::temperature_config` = `writing 0.7, roleplay 0.7, extraction 0.0, math 0.0,
  coding 0.0, reasoning 0.0, stem 0.1, humanities 0.1`;
* single-answer judging calls `chat_compeletion_openai(..., temperature=0, max_tokens=1024)`;
  the **2048** value only applies to the pairwise path — Phase 30's "max_tokens=2048" was a
  misread of `run_judge_pair`, corrected here;
* output format `[[rating]]`; a parse failure yields `rating = -1`;
* `NEED_REF_CATS = ["math", "reasoning", "coding"]`; reference answers come from the separate
  `reference_answer/gpt-4.jsonl`, not from `question["reference"]`;
* `gen_judgment.py` defaults: `--judge-model gpt-4`; MT-Bench scores come from the `single`
  mode (`single-v1-multi-turn`, `single-math-v1-multi-turn`);
* `API_MAX_RETRY = 8`, `API_RETRY_SLEEP = 10` in the judge helper (upstream defaults, not the
  paper's methodology).

**Distinction preserved:** these are *upstream FastChat defaults*. The paper specifies none of
them, and no default has been adopted as a paper decision.

---

## 10. XSTest immutable upstream revision and provenance evidence

| Field | Value |
|---|---|
| Repository | `https://github.com/paul-rottger/xstest` |
| License | CC-BY-4.0 (from repository metadata) |
| Tag / release | **none** — the repository publishes neither, so the commit SHA is the only immutable pin |
| Dataset commit (pinned) | **`475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5`** — “overdue cleanup”, 2025-02-11; last commit touching `xstest_prompts.csv` |
| `xstest_prompts.csv` | git blob **`ad38204d70b3964387f9690169d11afbc4756d01`**, 38 719 bytes |
| `evaluation/classify_completions_strmatch.py` | blob `aa8fa43a2043c7d400923bd4ffb628154aca1b7c`, 1 368 bytes |
| `evaluation/classify_completions_gpt.py` | blob `b9f911978a4ee5c91aa65239a59c2b979cd61576`, 2 721 bytes |
| `readme.md` | blob `2850b16498d65625fdc1886f83efd029541fcfc4` |
| Hash kind | git blob SHA-1 |
| SHA-256 | **`None`** (same reason as MTBench) |
| Paper version | **`PAPER VERSION UNVERIFIED`** |

Label and detector semantics verified from the README and the two classifier scripts:

* **Unsafe = `type` starts with `contrast_`** (README, verbatim: “unsafe prompts, which are
  those where the "type" starts with "contrast_"”).
* The CSV also carries an explicit `label` column with `safe` / `unsafe` values; the prefix rule
  and the label agree. Phase 30's note that “`type` is a prompt-type label, not a safe/unsafe
  boolean” is correct about `type`, and is now refined: a separate `label` column exists.
* Dataset size per the NAACL 2024 paper: **250 safe + 200 unsafe = 450 prompts**. Not
  re-counted locally — that would require the data.
* `classify_completions_strmatch.py`: lower-cases, strips, then prefix-matches a fixed refusal
  list → only `2_full_refusal` / `1_full_compliance`. **No partial class**; a partial refusal is
  labelled compliance.
* `classify_completions_gpt.py`: GPT-4, `temperature=0`, `max_tokens=16`, `top_p=1`,
  `frequency_penalty=0`, `presence_penalty=0`; three classes including
  `3_partial_refusal`; retries every 5 s indefinitely on exception.

---

## 11. Dataset version-verification status

| Component | Revision pin | Paper version | Local file | Status |
|---|---|---|---|---|
| MTBench | **verified** (tag + commit + blob) | `PAPER VERSION UNVERIFIED` | absent | `BLOCKED — DATA REQUIRED` |
| XSTest | **verified** (commit + blob) | `PAPER VERSION UNVERIFIED` | absent | `BLOCKED — DATA REQUIRED` |

`UPSTREAM REVISION UNVERIFIED` is **not** claimed for either dataset: both revisions are
commit-pinned and content-addressed. What remains unverified is the **paper's** choice of
release, which the paper never states.

Manifest integration: each utility component now carries
`utility.components.<name>.upstream` with `repository`, `pinned_tag`, `pinned_commit`,
`dataset_files[*].git_blob_sha1`, `dataset_files[*].size_bytes`, `code_files`,
`dataset_version_status`, `sha256: None`, `sha256_reason`, `retrieval`, and the list of
upstream defaults that are **not** paper methodology. Recorded even while the dataset is
absent, so an authorised future run knows exactly which bytes to fetch.

---

## 12. Remaining utility methodology gaps

MTBench (unresolved, unchanged):

1. exact MT-Bench release used by the authors (unstated);
2. judge model (upstream default GPT-4 is not a paper statement);
3. verbatim judge prompt (upstream `judge_prompts.jsonl::single-v1-multi-turn`, unverified);
4. judge decoding (`temperature=0`, `max_tokens=1024` upstream only);
5. per-category generation temperature (0/0.1/0.7) conflicts with the paper's “all
   temperatures 0.7”;
6. turn-2 provenance — upstream reads `turns[1]`; the project loader can generate it when
   absent, and the paper says nothing;
7. reference-answer source and use (math/reasoning/coding only);
8. whether the paper scores all questions or a subset (upstream: all).

XSTest (unresolved, unchanged):

1. exact XSTest release used by the authors (unstated);
2. over-refusal denominator — all 450 prompts, or the 250 safe prompts only;
3. whether unsafe contrast prompts are excluded;
4. detector — string-match (2 classes) vs GPT-4 (3 classes, partial refusals counted);
5. whether partial refusals count as refusals;
6. any custom filtering or normalisation;
7. refusal-detector prompt/version if the GPT path is chosen.

A working script plus a schema validator is **not** a completed utility evaluation.

---

## 13. Baseline/defence invariant verification

Unchanged and re-verified in this phase's suite (targeted run of retry, ASR, cloud-provider,
usage, NBF-integration, qualification-gate, paper-fidelity and Phase 24 P0 tests:
**401 passed**).

| Invariant | Observed |
|---|---|
| variants checked | `['models','temperature','dataset','attacks','prompt_variant','paper_reproduction']` |
| models | `gpt-4o-2024-08-06` for attacker / target / evaluator in both arms |
| temperature | `0.7 / 0.7 / 0.7` in both arms |
| attacks | `['actor_attack','crescendo_paper','opposite_day']` in both arms |
| dataset | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb`, file order, 200 goals |
| `comparison_group` | `paper_reproduction_r1_gpt4o\|ac789de8…\|official\|8` — shared by both arms |
| NBF block | `eta 0.001`, `raw_admission_threshold −0.001`, `filter_condition h + eta >= 0`, `admission_condition h < -eta`; only `enabled` differs (False vs True) |
| ActorAttack prompt bytes | 11 hashes identical to Phase 30: `extract 8cb9ffa8…`, `network 3e2dc148…`, `actor 4f223f80…`, `more_actor 2ef3718b…`, `queries 47d14b53…`, `json_format_question 13adbacf…`, `attack_modify f1c7ea4b…`, `step_judge 1c5b047e…`, `summary_with_details 8e216a6c…`, `summary_without_details 94b2f3c4…`, rubric `146589a4…` |
| Arms identical by construction | `actor_attack_prompt_provenance("official")` equal for both arms; no variant-level prompt override |
| Fail-closed qualification gate | Phase 26/27 gate tests pass unchanged |
| Utility guards | a config cannot declare a blocked component ready; `table1_ready=false` recomputed from the filesystem |

---

## 14. Full regression comparison with Phase 30

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase31_run1.xml
```

| | Phase 29 | Phase 30 | Phase 31 |
|---|---:|---:|---:|
| Collected | 1219 | 1244 | **1277** (+33) |
| Passed | 1215 | 1240 | **1273** (+33) |
| Failed | 4 | 4 | **4** |
| Errors | 0 | 0 | **0** |
| Skipped | 0 | 0 | **0** |
| Time | 127.9 s | 127.3 s | 168.2 s |

Failing node IDs are **identical, one for one**, to Phase 30 (`_phase30_run1.xml` vs
`_phase31_run1.xml`, compared programmatically):

1. `tests.test_phase14_2_rotation.TestPreRunGuardReleasesCacheFirst::test_releases_cache_before_measuring` — CUDA/VRAM environment
2. `tests.test_phase14_2_rotation.TestPreRunGuardReleasesCacheFirst::test_effective_free_after_release_is_what_is_measured` — CUDA/VRAM environment
3. `tests.test_phase8_architecture.TestThreeModelConfigValidation::test_default_config_valid` — requires `Qwen/Qwen3.5-4B`
4. `tests.test_phase8_architecture.TestConfiguredModelsPresentOnDisk::test_attacker_present` — requires `Qwen/Qwen3.5-4B`

No test was skipped, suppressed, weakened or deleted. Three assertions that encoded the
*superseded* (incorrect) Phase 30 statement were corrected:

| Test | Change |
|---|---|
| `test_phase30_retry_provenance::test_schedule_matches_the_author_release` | now asserts the 10-entry schedule and the success/exhaustion split |
| `test_phase30_retry_provenance::test_failure_after_all_ten_attempts` | asserts applied 9 waits/81 s **and** the recorded official 100 s, plus `exhausted/succeeded/actual_attempts` |
| `test_phase29_alignment::test_documented_official_retry_semantics` | corrected to the source-verified values (`sleeps_after_final_failed_attempt=True`, 100 s, `json_parse_failures_retried=True`, first-attempt caveat) and the new status string |

---

## 15. Changed-file inventory

| File | Change |
|---|---|
| `src/guardbound/llm/openai_client.py` | corrected docstring; source-derived schedule constants (10-entry official, 9-entry applied); `retry_policy_record()`; `_call_retry_record()`; `retry_summary()` now reports `actual_attempts`, `error_categories`, official/applied exhaustion totals and the final-wait flag; per-call `succeeded`/`exhausted`/`terminal_category`/`elapsed` |
| `src/guardbound/evaluation/readiness.py` | `UPSTREAM_PROVENANCE` (immutable pins + blob SHA-1s + `PAPER VERSION UNVERIFIED`) and `upstream_provenance()` |
| `scripts/_cloud_common.py` | `PAPER_CODE_RETRY_POLICY` now built from `retry_policy_record()` (single source); `retry_delta` exposes `actual_attempts`; utility manifest block embeds `upstream` provenance |
| `scripts/cloud_pilot.py` | retry console block prints official vs applied semantics and per-role `elapsed`, `exhausted`, `non_retryable` |
| `tests/test_phase31_retry_semantics.py` | **new**, 33 tests (reference-byte probe + Guardbound + manifest + provenance + ASR + goal accounting) |
| `tests/test_phase30_retry_provenance.py` | two schedule/exhaustion assertions corrected |
| `tests/test_phase29_alignment.py` | one policy assertion corrected |
| `PHASE_31_RETRY_SEMANTICS_AND_REPRODUCIBILITY_AUDIT.md` | this report |
| `_phase31_run1.xml` | junit evidence for the full run |

Untouched: attack algorithms and their control flow, official prompt bytes, NBF
architecture/checkpoint/formula/η, evaluator temperature (0.7), model identifiers, dataset
contents, configs, frozen historical artifacts, usage-schema fields.

---

## 16. Frozen-artifact integrity verification

| Artifact | SHA-256 (this phase) | Matches Phase 30 |
|---|---|---|
| HarmBench tasks (`NBF-LLM/data/test/harmbench_tasks.json`) | `ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb` | yes (`ac789de8…d014eb`) |
| NBF checkpoint (`models_best_nbf_released.pth`) | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` | yes |
| Reference `generate.py` | `8b2dcd8520def27f62a5f4a96766d9ef07b3c462d6898adb3b0ce067e07dcb29` | unchanged (read-only inspection) |
| `src/guardbound/attacks/_official_prompts.py` | `b3cd2d074506429737ca34db91e4b4ee4637cec3d81c67e22c8c0129294a92e2` | unchanged (module constants; the 11 prompt hashes in §13 are identical) |
| `data/` tree | `git status --porcelain -- data/` → empty | no tracked change |
| Local utility datasets | `data/raw/mtbench.jsonl` absent, `data/raw/xstest.jsonl` absent | unchanged |

---

## 17. Remaining discrepancies and risks

| # | Item | Sev | Note |
|---|---|---|---|
| 1 | Reference exhaustion waits 19 s after the tenth failure; Guardbound does not | P3 | documented deviation #3; both totals are in every manifest (100 s vs 81 s) |
| 2 | Reference does not retry a first-attempt API failure (`UnboundLocalError`) | P2 | Phase 30's "retries any exception" was inaccurate; the record now states the asymmetry, and Guardbound deliberately does not reproduce the bug |
| 3 | Phase 30 report §2–§4 retain the superseded retry claim | P2 | historical document; superseded by this report — cite Phase 31 for retry semantics |
| 4 | Phase 30 stated MTBench judge `max_tokens=2048` | P3 | correction: 1024 on the single-answer path; 2048 belongs to the pairwise path |
| 5 | Retry amplification | P2 | a persistently failing logical call costs 10 requests and 81 s. No budget cap exists (by design); operators must watch `retry_telemetry` |
| 6 | Evaluator temperature 0.7 (paper) vs 0.0 (author's code) | P1 | unchanged by instruction; recorded in every manifest |
| 7 | JSON parse failures are not regenerated | P2 | documented deviation #4; prevents silent extra sampling draws |
| 8 | Reference's `llama`/`deepseek` branch uses `range(100)` with **no** sleep and the same unbound-`response` bug | P3 | not exercised by any Guardbound config (the OpenAI branch applies); recorded so nobody conflates the two |
| 9 | MTBench / XSTest blocked | P1 (table) | data absent; paper versions unverified; methodology gaps enumerated in §12 |
| 10 | XSTest denominator and detector unresolved | P1 (utility) | “refused over all” vs 250 safe; strmatch vs GPT-4 |
| 11 | MTBench per-category temperatures conflict with “all 0.7” | P1 (utility) | must be decided before a utility run |
| 12 | No seed; single run; served revision not exposed | P2 | both codebases sample at 0.7 |
| 13 | 7 target calls vs the paper's “8 turns” | P3 | faithful to the author's loop |
| 14 | `--offline-mock` cannot drive ActorAttack's pre-attack JSON | P3 | pre-existing harness gap; surfaces as an accounted infrastructure failure |

---

## 18. Dataset integrity statement

**No dataset was downloaded, cloned, materialised, copied or modified.** No file was written
under `data/`; `git status --porcelain -- data/` is empty and neither `data/raw/mtbench.jsonl`
nor `data/raw/xstest.jsonl` exists. No `question.jsonl`, `reference_answer/gpt-4.jsonl` or
`xstest_prompts.csv` was fetched, saved, cached or parsed.

Full disclosure of the research mechanics: the research used GitHub **metadata** endpoints
(repository, refs/tags, per-path commit history, and file metadata including git blob SHA-1 and
byte size) plus **source-code** files (`common.py`, `gen_model_answer.py`, `gen_judgment.py`,
`readme.md`, and the two XSTest classifier scripts). Two contents-API metadata calls for
`question.jsonl` and `xstest_prompts.csv` also returned the file bodies inline in the HTTP
response; those bytes were **not** written, cached, copied or materialised anywhere, and no
dataset file exists on disk as a result. The pinned git blob SHA-1s in §9–§10 identify the exact
bytes without requiring a download.

The repository's own HarmBench file and the NBF checkpoint are byte-identical to Phase 30
(§16).

---

## 19. Authorization statement

No paid API call, credential read from the environment, qualification, smoke test, pilot or
experiment was executed. Every LLM interaction in this phase went to scripted fakes, the
offline mock transport (`--offline-mock`), or `MockChatLLM` with injected sleep and an injected
clock. The CLI runs performed were `--dry-run` (both arms) and one `--offline-mock --limit 1`
manifest-generation run, neither of which sends a request. No key was requested, printed or
exposed; the only keys in play were dummy strings inside monkeypatched tests. No budget cap
exists or was introduced, no C1/I1/R2 arm was added, and the Phase 27 fail-closed qualification
gate is unchanged.

---

## 20. Final readiness classifications

### Attack-ASR reproduction: `READY WITH DOCUMENTED DEVIATIONS`

The attempt count and the success-path schedule match the author's release (10 attempts, nine
waits, 1–17 s, 81 s, no hidden SDK retries). The exhaustion path is deliberately different and
is now reported as such: the reference waits a further 19 s (100 s total) and returns `{}`;
Guardbound raises a typed `RetriesExhausted` after 81 s and records an infrastructure failure.
Non-retryable categories fail fast, malformed JSON is never regenerated, and a first-attempt
transport failure is retried instead of aborting. Prompts are hashed from their actual bytes,
the NBF terminology is explicit and boundary-tested, dual ASR denominators are unchanged, and
both arms validate with only `nbf.enabled` differing. All deviations are enumerated in §5 and
in every manifest.

### Retry alignment: `ALIGNED WITH DOCUMENTED DEVIATIONS`

Attempt count and success-path backoff are source-exact; the two schedules are published side
by side (`81 s` applied vs `100 s` official on exhaustion), the final-wait omission is an
explicit, justified deviation, and the reference's first-attempt `UnboundLocalError` is
documented rather than imitated. Alignment is **not** claimed as exact equivalence.

### MTBench: `BLOCKED — DATA REQUIRED`

Upstream revision pinned (`v0.2.36` / `b494d0c6…`, dataset blob `464e2c22…`), schema and
upstream judge behaviour documented, but the dataset is absent, the paper's release is
`PAPER VERSION UNVERIFIED`, and §12's methodology gaps are unresolved. η remains 1e-3; no
evaluation was executed.

### XSTest: `BLOCKED — DATA REQUIRED`

Upstream revision pinned (`475f10bf…`, dataset blob `ad38204d…`), label and detector semantics
documented, but the dataset is absent, the paper's release is `PAPER VERSION UNVERIFIED`, and
the denominator/detector decisions are unresolved. η remains 1e-3; no evaluation was executed.

### Full GPT-4o Table 1: `NOT READY`

Three of five columns are implementation-ready; the two utility columns cannot be produced. A
successful three-attack ASR run would be a **partial row** and must be reported as such.

**A readiness classification is not authorization to execute or spend money. No classification
above authorises a paid run, a dataset download or an experiment. Phase 31 ends here.**
