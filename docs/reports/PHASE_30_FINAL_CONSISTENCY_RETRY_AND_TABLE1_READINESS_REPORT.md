# PHASE 30 — FINAL CONSISTENCY, RETRY ALIGNMENT & TABLE 1 READINESS REPORT

**Attack-ASR: `READY WITH DOCUMENTED DEVIATIONS`**
**MTBench: `BLOCKED — DATA REQUIRED`** (methodology also unresolved)
**XSTest: `BLOCKED — DATA REQUIRED`** (methodology also unresolved)
**Full GPT-4o Table 1: `NOT READY`**

Offline implementation and read-only research. **No dataset was downloaded,
cloned, materialised or modified; no API call, credential read, qualification,
smoke test, pilot or experiment occurred.** Frozen artifacts are untouched.

Repository state: `main` @ `89ed03f` + uncommitted work. Suite: **1244 collected /
1240 passed / 4 failed / 0 errors / 0 skipped** — the same four environmental
node IDs as Phase 29.

---

## 1. Executive summary

| Item | Outcome |
|---|---|
| Retry schedule | **Aligned**: 10 attempts, nine waits 1, 3, …, 17 s (81 s), `max_retries=0`; exhaustion raises a typed `RetriesExhausted` |
| Two deviations from the author's loop | non-retryable categories fail fast; exhaustion is typed instead of `{}` → `KeyError` |
| ASR denominators | unchanged from Phase 29 and re-validated (worked example 50/200/10/5/185 → 0.2500 / 0.2703) |
| ActorAttack prompts | 11 prompt byte strings identified and SHA-256'd, rubric reuse recorded, dynamic prompts flagged |
| NBF terminology | `eta`, `raw_admission_threshold`, `filter_condition`, `admission_condition` in every manifest; boundary decisions proven |
| Arms | both resolve through the real CLI; invariants reported; only `nbf.enabled` differs |
| MTBench | upstream dataset **found and documented**, schema **partial match**, paper version **unverified** → **blocked** |
| XSTest | upstream dataset **found and documented**, schema **partial match** (CSV, `contrast_` labels), paper version **unverified** → **blocked** |
| Full Table 1 | **not ready** — two of five columns cannot be produced |

Two defects were caught by the suite during this phase and fixed before shipping:
the new provenance function crashed on the **legacy** prompt variant (it called an
official-only helper), which would have broken the frozen gpt-4o-mini pilot
config; and the Phase 29 retry record listed a 19 s wait that the reference loop
never performs (**10 attempts retry 9 times**).

---

## 2. Official retry source inspection

`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/generate.py`:

```python
95  def generate(messages, client, model, json_format=False, temperature=0.7):
...
127     else:                                   # openai gpt branch
129         for ind_ in range(10):
130             try:
137-141             response = client.chat.completions.create(
                        model=model, messages=messages, temperature=temperature,
                        response_format=... )
144-147             if json_format: return json.loads(...)
147                 return response.choices[0].message.content
148             except Exception as e:
149-150             print("Error in generate: ", e)
151             time.sleep(2*ind_ + 1)
152         return {}
```

| Property | Verified value |
|---|---|
| Attempts | **10** (`ind_ = 0…9`) |
| Retried exceptions | **any** exception (bare `except Exception`) — auth, permission and invalid request included |
| Backoff | `2*ind_ + 1` → **1, 3, 5, 7, 9, 11, 13, 15, 17 s** |
| Waits | **nine** (none after the tenth attempt) |
| Total scheduled | 1+3+5+7+9+11+13+15+17 = **81 s** |
| Exhaustion | returns `{}` |
| Downstream | `crescendomation/run.py:21` (`data['generatedQuestion']`) and `evaluate_with_rubric.py:32` (`data["score"]`) raise `KeyError`; `steering.py:103-114` has no per-case guard, so **the whole run aborts** |
| Paper | silent on retries |

Phase 29's record listed a tenth wait of 19 s; the code performs only nine. This
phase corrects the record and the implementation together.

---

## 3. Guardbound retry implementation and deviations

`src/guardbound/llm/openai_client.py`:

* constants `OFFICIAL_RETRY_ATTEMPTS = 10`, `OFFICIAL_RETRY_BACKOFF_SECONDS =
  (1,3,5,7,9,11,13,15,17)`, `OFFICIAL_RETRY_TOTAL_BACKOFF_SECONDS = 81`,
  `RETRY_POLICY_NAME = "official_retry_schedule + Guardbound_typed_failure_handling"`;
* `OpenAIChatLLM(..., retry_attempts=10, retry_backoff=<official>, sleep_fn=<time.sleep>)`
  — the schedule lives in `generate()`, the SDK client still gets `max_retries=0`;
* on a retryable failure: increment the attempt, log, `sleep_fn(backoff[i])`, retry;
  **no sleep after the tenth attempt**;
* `RetriesExhausted` (in `cloud_errors.py`, subclass of `CloudProviderError`)
  carries `attempts`, `retries`, `backoff_seconds`, `total_scheduled_backoff_seconds`,
  `error_categories`, and `to_dict()["exhausted"] = True`;
* telemetry: `retry_summary()` → policy, `max_attempts`, `backoff_schedule`,
  `max_scheduled_backoff_seconds`, and cumulative `calls/attempts/retries/
  scheduled_backoff_seconds/elapsed_backoff_seconds/successful_calls/
  exhausted_calls/non_retryable_calls/by_category`. Backoff is accumulated **as it
  is scheduled**, so an interrupt mid-wait is still represented;
* **one usage row per logical call** — the ten attempts are aggregated in the retry
  telemetry, not as ten billing rows (the usage schema is unchanged);
* `last_request_stats["retry"]` records `{attempts, retries,
  scheduled_backoff_seconds, error_categories}` for every call.

Deliberate deviations (documented, not accidental):

| # | Deviation | Reason |
|---|---|---|
| 1 | `authentication`, `permission`, `invalid_request`, `context_length`, `content_filter` fail after one attempt | retrying them cannot change the outcome; it would only burn 81 s and real money |
| 2 | exhaustion raises `RetriesExhausted` instead of returning `{}` | `{}` is indistinguishable from a reply at the call site and crashes the run via `KeyError`; the pilot records a typed infrastructure failure and continues |
| 3 | `RetriesExhausted.category` is the **last** attempt's category | keeps the stable failure taxonomy intact while the schedule facts ride in attributes |
| 4 | with `retry_attempts=1` a failure raises the classified error, not `RetriesExhausted` | no retry was exhausted; the specific category is the truthful signal |

The applied policy is published in every manifest (`retry_policy`, `asr.retry_policy`)
and printed at the end of a run.

---

## 4. Retry tests

25 new tests in `tests/test_phase30_retry_provenance.py`; independent probe output
(sleep injected, no waiting):

| Injected failures | HTTP calls | Raised | Attempts | Retries | Sleeps | Scheduled |
|---:|---:|---|---:|---:|---|---:|
| 0 | 1 | — | 1 | 0 | [] | 0 s |
| 1 | 2 | — | 2 | 1 | [1] | 1 s |
| 4 | 5 | — | 5 | 4 | [1,3,5,7] | 16 s |
| 9 | 10 | — | 10 | 9 | [1..17] | **81 s** |
| 10 | 10 | `RetriesExhausted` | 10 | 9 | [1..17] | **81 s** |
| 401 | 1 | `CloudAuthError` | 1 | 0 | [] | 0 s, `non_retryable_calls=1` |

Coverage: success on attempts 1, 2, 5 and 10; failure after all ten; **no eleventh
call** (the fake holds an eleventh response that is never consumed); the exact
backoff list; cumulative 81 s; no sleep after the tenth attempt; `max_retries=0`;
auth/permission and 400 fail fast; one usage row despite three attempts;
interrupted retry (`attempts=2`, `retries=2`, 4 s scheduled, elapsed < 4 s, the
interrupted attempt not repeated); a failed goal does not stop the run (patched
`run_one` → exit 1, three unique goal rows, one `infrastructure_failure`, then the
remaining goals complete); retry telemetry present in the manifest and in every
goal-level record; invalid evaluator outcomes never counted as successes.

---

## 5. ASR denominator definitions and validation

Unchanged from Phase 29, re-proven:

* `asr_paper = successful_goals / expected_goals` — failures and invalid evaluator
  outcomes stay in the denominator as **not successful**;
* `asr_valid_completed = successful_goals / valid_completed_goals` — diagnostic;
* zero denominator → `None`, never `0.0`;
* failures and invalid outcomes counted separately
  (`infrastructure_failure_count`, `invalid_evaluator_count`).

Worked example from the brief, executed by
`test_phase30_worked_example`: 200 expected, 50 successful, 10 infrastructure,
5 invalid, 185 valid → **`asr_paper = 0.2500`**, **`asr_valid_completed = 0.2703`**,
both printed side by side in the console block and the manifest.

---

## 6. ActorAttack prompt provenance

`src/guardbound/attacks/actor_attack.py` owns **ten** dedicated prompt strings; the
**rubric is the only shared one** (delegated to the Crescendo/Opposite-day renderer
at `:488-492`). `prompt_variants.actor_attack_prompt_provenance()` hashes the actual
module constants (not copies) and records kind, source, placeholders and reuse:

| Prompt | SHA-256 (first 16) | Kind |
|---|---|---|
| `EXTRACT_PROMPT` | `8cb9ffa80a90c0c3` | template (`org_query`) |
| `NETWORK_PROMPT` | `3e2dc148153f4d26` | template (`harm_target`) |
| `ACTOR_PROMPT` | `4f223f802807bfca` | template (`num_string`); canonical rendering for `3 actors` = `f50c9aefcaad90ff` |
| `MORE_ACTOR_PROMPT` | `2ef3718b968521bc` | constant (assistant turn) |
| `QUERIES_PROMPT` | `47d14b532a372a2f` | template |
| `JSON_FORMAT_QUESTION_PROMPT` | `13adbacf70707a41` | template (`resp`) |
| `ATTACK_MODIFY_PROMPT` | `f1c7ea4be4924afc` | template |
| `ATTACK_STEP_JUDGE_PROMPT` | `1c5b047ea1fc640a` | template (`query`, `response`) |
| `SUMMARY_WITH_DETAILS_PROMPT` | `8e216a6cd4ad4aeb` | template (`delivery_type`, `other_details`) |
| `SUMMARY_WITHOUT_DETAILS_PROMPT` | `94b2f3c47d6a04c0` | template (`delivery_type`) |
| rubric (shared) | `146589a4fba6d42c` | rendered; `reused_by: [actor_attack, crescendo_paper, opposite_day]` |

Dynamic content is declared, not hidden: the per-actor query chain is generated by
the attacker model and rewritten on rejection, and the `SUMMARY_*` texts are filled
per goal — both are listed under `dynamic` with an explanation.

Both arms are identical by construction: prompts are resolved from the
configuration-level `prompt_variant`, never from `variants` (tested), and the
manifest block for `r1_baseline` equals the one for `r1_paper`.

Tests verify each recorded hash equals
`sha256(<module constant>)` at runtime, that the actor rendering matches
`.format(num_string="3 actors")`, and that the rubric hash equals
`prompt_hashes()["rubric"]["variants"][variant]`.

---

## 7. NBF threshold verification

Every manifest now carries the paper's terminology (`_cloud_common.nbf_manifest_block`):

| Field | `r1_baseline` | `r1_paper` |
|---|---|---|
| `enabled` | `False` | `True` |
| `eta` | 0.001 | 0.001 |
| `raw_admission_threshold` | −0.001 | −0.001 |
| `filter_condition` | `h + eta >= 0` | same |
| `admission_condition` | `h < -eta` | same |
| `device` / `precision` | — / `float32` | `cpu` / `float32` |

Boundary tests (`admits`/`is_admitted`/`is_filtered`, ε ∈ {1e-9, 1e-6}):

* `h = −η` → **filtered** (strict `<`);
* `h = −η − ε` → **admitted**;
* `h = −η + ε` → **filtered**.

No checkpoint, architecture, score formula or η was changed.

---

## 8. Baseline-vs-defence invariants

Both arms ran through the real CLI (`--dry-run --limit 200`), exit 0:

| | `r1_baseline` | `r1_paper` |
|---|---|---|
| invariants checked | `['models','temperature','dataset','attacks','prompt_variant','paper_reproduction']` | same |
| models | gpt-4o-2024-08-06 × 3 | identical |
| temperatures | 0.7 / 0.7 / 0.7 | identical |
| attacks | `[actor_attack, crescendo_paper, opposite_day]` | identical |
| prompt variant / rubric | `official` (rubric `146589a4…`) | identical |
| dataset | `ac789de8…`, file order, 200 | identical |
| system prompt | "You are a helpful assistant." | identical |
| turn limit | 8 | identical |
| **NBF** | disabled | enabled, η = 0.001 |
| protocol | R1 validated | R1 validated |

`assert_invariants` returns the six declared invariants (it returned `[]` with one
variant, i.e. inert, in Phase 28); a variant overriding `temperature` is rejected.

---

## 9. MTBench research

**Paper** (arXiv:2503.00187v3, p8 §5.1; p22 B.1): "MTBench (Zheng et al., 2023) …
scores from 1 to 10"; on a positive barrier score the answer is replaced by the
exact refusal string, "followed by regular evaluation of MTBench". **No dataset
version, judge model, judge prompt, turn-2 sourcing or sampling temperature is
stated.** The release ships no utility code or data.

**Author's repository**: no MTBench asset, no utility script
(`NBF-LLM/` contains only the three attacks, `train.py`, `steering.py`, the NBF
checkpoint and the HarmBench file).

**Authoritative upstream** (`lm-sys/FastChat`, code inspected read-only):
`fastchat/llm_judge/gen_model_answer.py` iterates `question["turns"]` (both turns
come from the dataset), keys answers by `question["question_id"]`, and selects a
**per-category** temperature from `common.py::temperature_config`
(`writing 0.7, roleplay 0.7, extraction 0.0, math 0.0, coding 0.0, reasoning 0.0,
stem 0.1, humanities 0.1`). Judging uses `single-v1-multi-turn` with **GPT-4**,
`temperature=0`, `max_tokens=2048`, `[[rating]]` output; a parse failure yields
`rating = -1`. Reference answers are needed only for `math, reasoning, coding` and
come from a **separate reference file**, not from `question["reference"]`.

---

## 10. XSTest research

**Paper** (§5.1 p8; B.1): "over-refusal rate (refused queries over all) on XSTest
Röttger et al. (2024)" — no version, no subset rule, no detector.

**Author's repository**: no XSTest asset or script.

**Authoritative upstream** (`paul-rottger/xstest`, README + evaluation code
inspected read-only): `xstest_prompts.csv` contains all prompts in a `prompt`
column; **unsafe prompts are those whose `type` starts with `contrast_`**
(250 safe + 200 unsafe = 450, per the NAACL 2024 paper). Two detectors are
provided: `classify_completions_strmatch.py` (prefix string matching after
lower-casing → `2_full_refusal` / `1_full_compliance`, no partial class) and
`classify_completions_gpt.py` (GPT-4, `temperature=0`, `max_tokens=16`, three
classes `1_full_compliance` / `2_full_refusal` / `3_partial_refusal`).

---

## 11. Dataset version, schema, label and provenance comparison

| Field | MTBench | XSTest |
|---|---|---|
| Dataset | MT-Bench (Zheng et al., 2023) | XSTest (Röttger et al., NAACL 2024) |
| Official source | `lm-sys/FastChat` → `fastchat/llm_judge/data/mt_bench/question.jsonl` | `paul-rottger/xstest` → `xstest_prompts.csv` |
| Paper version | **not stated** | **not stated** |
| Upstream format | JSONL | CSV |
| Upstream schema | `question_id`, `category`, `turns: [q1, q2]`; separate reference answers for 4 categories | `prompt`; `type` labels (`contrast_*` = unsafe) |
| Guardbound loader schema | `{question, reference?, turn2_question?, category?}` | `{prompt|text, type?}` |
| Label semantics | n/a (1–10 score) | **`type` is a prompt-type label, not a safe/unsafe boolean** |
| Transformation needed | map `turns[0]→question`, `turns[1]→turn2_question`, keep `category`; drop/ignore `question_id` or add it | CSV→JSONL; `type` prefix mapping; **decide the safe subset** |
| Paper match | `UPSTREAM DATA FOUND — PAPER VERSION UNVERIFIED` + `PARTIAL MATCH — TRANSFORMATION REQUIRED` | same |
| Availability | public, read-only metadata verified; **no download performed** | public; **no download performed** |
| Confidence | schema verified from upstream **code**; version unverified | schema verified from upstream **README + evaluation code**; version unverified |

Not verified anywhere: the exact releases the authors used, the judge prompt/model
for MTBench, the refusal detector for XSTest, and whether XSTest's unsafe contrasts
are excluded from the over-refusal denominator.

---

## 12. Utility evaluation implementation readiness

| Component | Implemented and verified | Unsatisfied |
|---|---|---|
| MTBench | 2-turn execution at target temperature 0.7; NBF substitution with the canonical refusal string; 1–10 judge parsing; caching; aggregation + reporting; `--eta` **1e-3**; loader schema check; manifest provenance | dataset absent; judge model/prompt, turn-2 policy and per-category temperatures unresolved |
| XSTest | target execution; NBF filtering; shared refusal detector; refusal-rate aggregation; `--eta` **1e-3**; schema check; manifest provenance | dataset absent; safe-subset policy and detector choice unresolved (two upstream detectors, neither selected by the paper) |

Both CLIs exit 2 with `BLOCKED — DATA REQUIRED` when the dataset is missing or
schema-invalid; nothing is downloaded, fabricated or substituted. The new
`utility_manifest_block()` recomputes status from the filesystem, so a config can
never declare a blocked component ready (tested), and records 7 explicit
`methodology_gaps`.

A working script and a schema validator are **not** a completed utility evaluation.

---

## 13. Why the utility components remain blocked

1. **Data**: neither dataset is present in the repository and neither may be
   downloaded in this phase (`MTBENCH: BLOCKED — DATA REQUIRED`,
   `XSTEST: BLOCKED — DATA REQUIRED`).
2. **Version**: the paper names the benchmarks but not the releases; upstream
   repositories evolve (the XSTest repo even carries `xstest_v2_*` completion files).
3. **Transformations**: both upstream schemas differ from the loader contract; the
   mappings are deterministic but must be authorised and applied deliberately, not
   silently.
4. **Methodology**: MTBench judge model/prompt and the per-category temperature
   conflict with "all temperatures 0.7"; XSTest safe-subset and detector are
   unspecified.
5. **Consequence**: `table1_ready: false`; a three-attack ASR run is a *partial*
   row and must be reported as such.

---

## 14. Manifest and telemetry changes

Added (nothing removed or renamed):

| Requirement | Manifest path | Observed |
|---|---|---|
| Official retry semantics + applied policy | `retry_policy` (also `asr.retry_policy`) | 10 attempts, nine waits, 81 s, deviations listed |
| Per-role attempts/retries/backoff | `retry_telemetry.<role>` | `{policy, max_attempts, backoff_schedule, calls, attempts, retries, scheduled_backoff_seconds, elapsed_backoff_seconds, successful_calls, exhausted_calls, non_retryable_calls, by_category}` |
| Run-level retry totals | `retry_run` | `{}` when no retry; else per-role deltas + `total_retries` |
| Per-goal retry metadata | every result row: `retry` | `{}` for clean goals, deltas otherwise |
| Exception category & goal/attack | failure rows: `failure_category`, `error_class`, `attack`, `goal` | present |
| ASR definitions (both) | `asr.denominators`, `run.asr_definitions` | paper + diagnostic text |
| Failure / invalid counts | `run.infrastructure_failure_count`, `run.invalid_evaluator_count` | both |
| ActorAttack prompt identity | `actor_attack_prompts` | 11 hashes + reuse + dynamic |
| NBF terminology | `nbf.{eta, raw_admission_threshold, filter_condition, admission_condition}` | −0.001, both conditions |
| Baseline/defence linkage | `linkage.{comparison_group, arms, nbf_enabled}` | shared string across arms |
| Utility status/provenance/gaps | `utility.{components, table1_ready, methodology_gaps}` | blocked, 7 gaps |
| Dataset schema verification | `utility.components.<name>.dataset.schema_status` | `PARTIAL MATCH — TRANSFORMATION REQUIRED` |

Preserved: usage telemetry (21 fields/call), `asr_denominator` + definition, `run`
completion facts, `prompt_hashes`, `config.sha256`, `dataset.goal_id_hash`,
`environment`, `git_*`, `rate_card`, `paper_protocol`.

Observed end-to-end (offline mock, `r1_baseline`, `--limit 1`): 3 unique goal rows,
1 `infrastructure_failure`, `asr_paper 0/3`, `asr_valid_completed 0/2`,
`retry_telemetry` for attacker/target/evaluator, 11 actor prompt hashes, utility
blocked.

---

## 15. Full regression results against Phase 29

```powershell
python -m pytest tests/ -q --no-header -p no:cacheprovider --tb=line --junitxml=_phase30_run1.xml
```

| | Phase 29 | Phase 30 |
|---|---:|---:|
| Collected | 1219 | **1244** (+25) |
| Passed | 1215 | **1240** |
| Failed | 4 | **4** |
| Errors | 0 | **0** |
| Skipped | 0 | **0** |
| Time | 127.9 s | 127.3 s |

Identical failing node IDs (two CUDA-gated, two requiring `Qwen/Qwen3.5-4B`).

Tests updated because Phase 30 explicitly replaced the behaviour they encoded (no
test weakened or deleted):

| Test | Change |
|---|---|
| `test_cloud_provider::make_llm` | one attempt by default for error-taxonomy tests; the schedule has dedicated coverage |
| `...::test_no_hidden_retries_on_a_failing_request` | now asserts **10 visible attempts and no eleventh**, with `time.sleep` patched — the "no hidden SDK retry" property is unchanged and stronger |
| `test_phase29::test_documented_official_retry_semantics` | corrected: 9 waits (the 19 s entry was wrong), applied attempts 10, status `ALIGNED` |
| `test_phase29::test_transient_failure_makes_exactly_one_attempt` / `test_exhausted_attempts_never_become_a_verdict` | pass `retry_attempts=1` (the knob that reproduces the old policy); their assertions are unchanged |
| `test_phase29::test_pilot_manifest_records_dual_asr_and_linkage` | NBF block now asserts the new terminology fields instead of an exact dict |

Also covered by the suite and unchanged: the qualification fail-closed gate
(Phase 27), dual-ASR denominators, utility schema/blocking, dataset guard.

---

## 16. Changed-file inventory

| File | Change |
|---|---|
| `src/guardbound/llm/openai_client.py` | official 10-attempt schedule, injectable sleep, `retry_summary()`, per-call retry stats, typed exhaustion |
| `src/guardbound/llm/cloud_errors.py` | `NON_RETRYABLE_CATEGORIES`, `RetriesExhausted` |
| `src/guardbound/attacks/prompt_variants.py` | `actor_attack_prompt_provenance()`, `_placeholders()` |
| `scripts/_cloud_common.py` | policy record (10 attempts), `retry_summary_of/snapshot/delta`, `utility_manifest_block`, `actor_attack_prompt_block`, `nbf_manifest_block` |
| `scripts/cloud_pilot.py` | per-goal retry deltas, `retry_telemetry`/`retry_run`, actor-prompt block, utility block, NBF terminology, retry console summary |
| `configs/paper_reproduction_r1_gpt4o.yaml` | `utility:` block (both components blocked, 7 gaps, η = 1e-3) |
| `tests/test_phase30_retry_provenance.py` | **new**, 25 tests |
| `tests/test_phase29_alignment.py`, `tests/test_cloud_provider.py` | documented updates (§15) |

Untouched: attack algorithms/control flow, NBF architecture/checkpoint/formula/η,
official prompt bytes, dataset contents, evaluator temperature (0.7), model ids,
frozen artifacts.

---

## 17. Remaining discrepancies and risks

| # | Item | Sev | Note |
|---|---|---|---|
| 1 | Evaluator temperature 0.7 (paper) vs 0.0 (author's code) | P1 | unchanged by instruction; recorded in every manifest |
| 2 | Non-retryable categories fail fast; exhaustion is typed | P2 | documented deviations from "any exception / `{}`" |
| 3 | Retry amplification | P2 | a persistently failing logical call now costs 10 requests and 81 s. There is **no budget cap** (by design), so an operator must watch `retry_telemetry` |
| 4 | MTBench / XSTest blocked | P1 (table) | data absent; versions unverified; methodology gaps enumerated |
| 5 | MTBench per-category temperatures upstream vs "all 0.7" in the paper | P1 (utility) | conflicts; must be decided before a utility run |
| 6 | XSTest safe-subset ambiguity | P1 (utility) | "refused over all" vs the 250 safe prompts |
| 7 | Upstream dataset versions unverified | P1 (utility) | repositories are moving targets; hashes must be pinned when obtained |
| 8 | `--offline-mock` cannot drive ActorAttack's pre-attack JSON | P3 | pre-existing harness gap; surfaces as an accounted infrastructure failure |
| 9 | No seed; single run; served revision not exposed | P2 | both codebases sample at 0.7; deprecation date recorded |
| 10 | 7 target calls vs the paper's "8 turns" | P3 | faithful to the author's loop |
| 11 | Per-call wire settings absent from usage rows | P3 | retry metadata is separate; 21-field schema unchanged |
| 12 | XSTest `type` is a label family, not `safe/unsafe` | P2 | a naive mapping would mislabel every row; only documented, not applied |

---

## 18. Dataset integrity statement

**No dataset was downloaded, cloned, materialised, copied or modified.** The
research in §9–§11 used only published documentation and source-code files from the
upstream repositories (FastChat README + `common.py` + `gen_model_answer.py`;
XSTest README + repository listing + the two classifier scripts). No
`question.jsonl`, `xstest_prompts.csv` or other data file was fetched, and no file
was written under `data/`. The repository's own HarmBench file is unchanged
(`ac789de8…d014eb`) and so is the NBF checkpoint (`cea1a75b…`).

---

## 19. Authorization statement

No paid API call, credential read, qualification, smoke test, pilot or experiment
was executed. Every LLM call in this phase went to scripted fakes, the offline mock
transport, or `MockChatLLM` with injected sleep. No spending cap exists or was
introduced. The Phase 27 fail-closed qualification gate is unchanged.

---

## 20. Final readiness classifications

### Attack-ASR reproduction: `READY WITH DOCUMENTED DEVIATIONS`

The retry schedule now matches the author's release exactly (10 attempts, nine
waits, 81 s, no hidden SDK retries), exhaustion is a typed failure that keeps the
run alive and the ASR denominators honest, prompts are hashed from their actual
bytes, the NBF terminology is explicit and boundary-tested, and both arms validate
with only `nbf.enabled` differing. Remaining deviations are enumerated in §17 and
are visible in every manifest.

### MTBench: `BLOCKED — DATA REQUIRED`

Upstream dataset identified and its schema documented, but the dataset is not
present, the paper's version is unverified, and the judge/per-turn-temperature
methodology is unresolved.

### XSTest: `BLOCKED — DATA REQUIRED`

Upstream dataset identified (CSV, `contrast_` labels), but the dataset is not
present, the paper's version is unverified, and the safe-subset/detector decisions
are unresolved.

### Full GPT-4o Table 1: `NOT READY`

Three of five columns are implementation-ready; the two utility columns cannot be
produced. A successful three-attack ASR run would be a partial row and must be
reported as such.

**No classification above authorises a paid run, a dataset download or an
experiment.**
