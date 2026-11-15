# Phase 14.8 — Read-Only Telemetry Completeness Audit

**Date:** 2026-09-23
**Mode:** **READ-ONLY.** No code was modified in producing this audit.
**Trigger:** Phase 14.7 found `OFF_crescendo_001` produced a reduced result record after an attacker
JSON failure — 22 expected fields absent.
**Evidence base:** `results/phase14_7_18_run_production_validation.jsonl` (18 records: 17 normal,
1 error), plus the production source.

---

## 1. Objective

Determine whether the reduced error record is a data-integrity problem, whether it is isolated to
result-record construction, and whether a minimal fix is safe — **before** changing anything.

## 2. The two construction sites

Both live in `scripts/phase14_full_reproduction.py`. There is **no shared record builder**: the two
paths are written independently, which is the structural cause of the asymmetry.

| Path | Site | Lines |
|---|---|---|
| **Normal** — core record | `run_one_phase14(...)` `return {…}` | **695–734** |
| **Normal** — success augmentation | `run_batch` success branch after the attack returns | **1010–1047** |
| **Error** — whole record | `run_batch` `except Exception as e:` branch | **951–1003** |
| Serialization | `append_result(out_path, record)` → one JSON object per line | **464** |
| Output path | `results/phase14/batch{batch:02d}.jsonl` | **746** |
| Error classification | `classify_exception(e)` → `(failure_class, error_type, error_message)` | **267+** |

Trace of the two paths:

* **Normal.** `run_one_phase14` runs the attack, and on return derives `nbf_scores`/`filtered` from
  `attack.nbf_candidate_scores` / `nbf_candidate_verdicts`, builds `turns` and `rubric_scores` from the
  returned `conv.turns`, computes `termination_reason` and `nbf_stats`, and returns a 38-field dict.
  `run_batch` then stamps on runtime, failure class, VRAM, residency, system telemetry and the
  per-role call/parse counters.
* **Error.** `run_one_phase14` raises, so **`conv` is never assigned and never returned.** The
  `except` branch builds a fresh 26-field dict from `metadata` plus whatever the still-live
  `attacker_llm` / `target_llm` / `evaluator_llm` / `model_manager` objects expose, then serializes.

**The critical fact:** the `attack` object is **still alive and fully populated** in the `except`
branch. Everything the attack itself recorded is available there; only values that required the
returned `conv` object are lost.

## 3. Consumers

* In-repo, the **only** reader of these records is `existing_run_ids` (line 445), which reads
  `run_id` alone. Nothing in `src/` reads `rubric_scores`, `nbf_scores`, `filtered_queries`,
  `termination_reason` or `accepted_query_count` — verified by grep across `src/` (0 matches).
* Aggregation of these records is therefore **external** (analysis scripts), which is why Phase 14.7
  observed that "downstream aggregation must currently tolerate missing fields".

**Consequence:** no repo component would break from completing the schema. The blast radius is the
record producer alone.

## 4. Field-set diff — measured, not estimated

From the 18 Phase 14.7 records:

| | Count |
|---|---:|
| Normal-path key set (union over 17 runs) | **54** |
| Error-path key set (union over 1 run) | **33** |
| **Present only on the normal path** | **22** |
| **Present only on the error path** | **1** (`error`) |
| Total schema surface | **55** |

The 22 confirmed exactly, matching the Phase 14.7 finding.

## 5. Required field table

Legend — **Should exist on error?**: *always* = every record must carry it ·
*legitimately null* = the operation never completed, so an explicit empty/null is the correct
encoding · **Available?**: whether a faithful value is reachable at the failure point.

### 5.1 The 22 normal-only fields

| Field | Normal path | JSON-error path | Should exist on error? | Available at failure? | Action |
|---|---|---|---|---|---|
| `turns` | `conv.turns` → `[{query,response}]` | absent | **always** | **NO** — `conv` never returned | emit `null` |
| `num_turns` | `len(conv.turns)` | absent | **always** | **NO** | emit `null` |
| `accepted_query_count` | `len(conv.turns)` | absent | **always** | **NO** | emit `null` |
| `final_response` | `turns[-1].response` else `None` | absent | **always** | yes — truthfully `None` | emit `null` |
| `rubric_scores` | `attack._scores` (non-refused) | absent | **always** | **yes** — `attack._scores` | emit from `attack._scores` |
| `final_score` | same list, last | absent | **always** | **yes** | emit from `attack._scores` |
| `refusal_count` | `attack.get_refusal_count()` | absent | **always** | **yes** | emit |
| `refused_records` | `len(_scores) − len(accepted)` | absent | **always** | **yes** | emit |
| `nbf_scores` | `attack.nbf_candidate_scores` | absent | **always** | **yes** | emit (`[]` for OFF — NBF never ran) |
| `filtered_queries` | count of `verdicts is False` | absent | **always** | **yes** | emit (`0` for OFF) |
| `filter_count` | same value | absent | **always** | **yes** | emit (`0` for OFF) |
| `nbf_stats` | derived from `nbf_scores` | absent | **always** | **yes** | emit (zeroed for OFF) |
| `termination_reason` | one of 4 enum values | absent | **always** | **N/A** — no normal termination happened | emit `null` (see §7.2) |
| `attacker_parse_failures` | `attacker_llm.parse_failure_summary()` | absent (only the **evaluator's** is set) | **always** | **yes** | emit |
| `target_purpose_calls` | `target_llm.purpose_counts` | absent | **always** | **yes** | emit |
| `goal_text` | `goal_record["task"]` | absent | **always** | **yes** | emit |
| `target_system` | `goal_record.get("target_system")` | absent | **always** | **yes** | emit |
| `dtype` | `metadata["dtype"]` | absent | **always** | **yes** | emit |
| `quantization` | `metadata["quantization"]` | absent | **always** | **yes** | emit |
| `top_p` | `metadata["top_p"]` | absent | **always** | **yes** | emit |
| `generation_config` | `metadata["generation_config"]` | absent | **always** | **yes** | emit |
| `filter_trials` | `metadata["filter_trials"][attack_short]` | absent | **always** | **yes** | emit |

### 5.2 The error-only field

| Field | Normal path | JSON-error path | Should exist on error? | Available? | Action |
|---|---|---|---|---|---|
| `error` | **absent** (never set on success) | `str(e)` | yes | yes | **add `error: null` to the normal path** so schemas match |

### 5.3 A conditional field *within* the normal path

| Field | Normal path | JSON-error path | Note | Action |
|---|---|---|---|---|
| `attacker_raw_outputs` | present in **5 of 17** normal runs only (guarded by `num_turns == 0`) | **always** present | the success path is itself not schema-constant | emit on **both** paths unconditionally (list may be empty) |

**This is a second, previously unreported asymmetry**: the normal path's key set is not constant
either. A schema-equality test written against "a successful run" would be flaky depending on which
successful run it sampled.

### 5.4 The 31 fields already present on both paths

No action required. For completeness: `run_id`, `goal_id`, `goal`, `condition`, `attack`, `seed`,
`attacker_model`, `target_model`, `evaluator_model`, `embedding_model`, `checkpoint_sha256`,
`dataset_sha256`, `config_hash`, `code_commit`, `worktree_sha256`, `max_rounds`, `threshold`,
`nbf_enabled`, `success`, `failure_class`, `error_type`, `error_message`, `runtime`,
`runtime_seconds`, `peak_vram_gb`, `residency`, `system`, `llm_calls`, `attacker_purpose_calls`,
`evaluator_purpose_calls`, `evaluator_parse_failures`.

## 6. Classification of the 22 omissions

| Class | Fields | Count |
|---|---|---:|
| **Should always exist — recovered by a faithful, already-available source** | `rubric_scores`, `final_score`, `refusal_count`, `refused_records`, `nbf_scores`, `filtered_queries`, `filter_count`, `nbf_stats`, `attacker_parse_failures`, `target_purpose_calls`, `goal_text`, `target_system`, `dtype`, `quantization`, `top_p`, `generation_config`, `filter_trials` | **17** |
| **Should always exist — legitimately null because the operation never completed** | `final_response`, `termination_reason` | **2** |
| **Should always exist — NOT faithfully recoverable at the failure point** | `turns`, `num_turns`, `accepted_query_count` | **3** |

### 6.1 The 17 "accidentally omitted"

These are not lost data — they were simply never copied into the error dict. Every one is reachable
from an object already in scope in the `except` branch. Their absence is **purely an artefact of the
two record-construction sites being written independently.**

### 6.2 The 2 legitimately null

`final_response` — a run that errored has no final response; `null` is the true value.
`termination_reason` — the run did not terminate, it raised; no value in the existing enum
(`success_score_5`, `refusal_retry_limit`, `max_turns_reached`, `attack_loop_exhausted`) describes it.

### 6.3 The 3 genuinely unavailable — the reason this is not purely mechanical

`turns`, `num_turns` and `accepted_query_count` are all derived from the returned `conv` object
(`conv.turns`), and `run_attack_with_backtracking_async` **raises rather than returning**, so `conv`
never exists on an errored run.

**Specifically answering the audit questions the phase asks:**

* **"Is partial history available at the failure point?"** — **Not faithfully.** The attack object
  does hold `_history_attacker` and `_history_target`, but `_history_target` is populated with
  `"(Summary) " + last_response_summary` (e.g. `crescendo_paper.py:607`, and identically in
  `acronym.py:595` and `opposite_day.py:540`) — **not** the raw target responses that `conv.turns`
  carries. Reconstructing `turns` from those two lists would therefore produce records whose `response`
  field contains summaries where normal records contain responses: a **silent semantic divergence**,
  worse than an explicit `null`.
* **Could `num_turns` be proxied by `len(attack._history_attacker)`?** — **Unverified.** It is
  plausible but not established that this equals `len(conv.turns)` on every path, because the runner's
  refusal backtrack calls `history_t.pop()` on its own list while no corresponding pop of
  `_history_attacker` appears in the attacks. Substituting an unverified proxy for a count would
  invent precision the data does not have.
* **Is a faithful partial-history capture possible at all?** Yes, but **only by changing the runner**
  to attach partial state to the raised exception — which would be a modification to
  `src/guardbound/attacks/runner.py`, outside the "telemetry/error-record construction" scope this
  phase permits. **Out of scope, flagged for a separate decision.**

**"Is NBF absent because NBF was never invoked, or because the serializer omitted it?"** — the latter.
`attack.nbf_candidate_scores` / `nbf_candidate_verdicts` are live and populated in the `except` branch.
For an OFF run they legitimately hold `[]`, which is the correct empty encoding, exactly as on the
normal path. So NBF telemetry is **recoverable and should be emitted**.

**"Is evaluator telemetry absent because the evaluator was never invoked?"** — no. The evaluator
counters are already emitted on the error path; only `attacker_parse_failures` and
`target_purpose_calls` were missed.

**"Are call counts / runtime / VRAM / model identity available?"** — **all yes**, and all already
emitted on the error path except `attacker_parse_failures` and `target_purpose_calls` (§5.1).

## 7. Decisions that require sign-off before implementation

### 7.1 Adding `error: null` to the normal path

To make the key sets identical, the normal path must gain an `error` field. **This changes the shape
of every future success record.** It does not alter existing values, and the 18 Phase 14.7 records
would become a different schema generation from future ones. The alternative — leaving `error`
error-only and documenting it as an intentional exception — keeps backward compatibility but
forfeits the "identical schemas" goal. **Recommendation: add `error: null`** (the phase prefers
identical schemas), and note the schema-generation boundary in the report.

### 7.2 `termination_reason` on an error

No enum value exists for "the run raised". Options:

* **`null`** — honest, adds no new semantics, and `failure_class` / `error_type` already carry the
  reason. **Recommended.**
* **a new `"error"` value** — more self-describing, but introduces a new enum member, which is a
  termination-semantics change rather than a telemetry-completeness change. The phase forbids altering
  "termination semantics", so this needs explicit approval.

### 7.3 The three unrecoverable fields

Recommendation: emit `turns: null`, `num_turns: null`, `accepted_query_count: null` — present but
explicitly unknown. **Not** `[]`/`0`, which would assert "no turns occurred" when turns may have
occurred. This satisfies "field exists rather than is absent" without inventing values.

Consequence to accept knowingly: **the schemas will be key-identical but not value-identical**, so the
Phase D test must compare **key sets**, not values, and must tolerate three intentionally-null fields.

### 7.4 `attacker_raw_outputs` (the within-normal conditional)

To make the schema test meaningful, emit it unconditionally on both paths (possibly an empty list).
This also changes the shape of most future success records (12 of 17 Phase 14.7 runs lack it).

## 8. Is the change trivial and isolated?

**Isolated — yes.** The change is confined to record construction in `scripts/phase14_full_reproduction.py`
(sites 695–734, 951–1003, 1010–1047). Nothing in `src/` is touched. No attack algorithm, prompt,
schema, decoder, parser, generation parameter, retry, NBF, model-manager or rotation code is involved.
No in-repo consumer depends on the current asymmetry.

**Trivial — not entirely.** Three fields cannot be populated faithfully, and four decisions
(§7.1–§7.4) determine the resulting data semantics. Two of them —
adding `error` and un-conditioning `attacker_raw_outputs` — change the schema of *success* records,
not just error records.

**Per the phase's own instruction — "Do not implement the fix in the same step unless the audit itself
proves the change is trivial and isolated. Prefer stopping after the audit" — this audit stops here.**

## 9. Proposed minimal fix, for separate review

A precise specification, ready to apply once §7 is decided:

1. **One canonical builder.** Add a single function that returns the complete 55-key record with every
   field defaulted (e.g. `null` for scalars, `[]` for score/turn lists, `0` for counts). Both the
   success and error paths construct through it and then overwrite the fields they can populate.
2. **Error path additionally passes** the 17 faithfully-recoverable values from the live `attack`,
   `attacker_llm`, `target_llm` and `metadata` objects.
3. **Leave `null`** for `turns`, `num_turns`, `accepted_query_count` (unrecoverable) and
   `final_response`, `termination_reason` (legitimately absent).
4. **`append_result` unchanged** — serialization already writes whatever dict it is given.
5. **No change to `run_one_phase14`'s return value, the runner, the attacks, or any `src/` module.**

Expected diff size: one new helper plus ~20 lines in the error branch and ~3 on the success branch —
confined to one file.

## 10. Explicitly unchanged during this audit

No file was created, edited or deleted in `src/` or `scripts/`. No attack algorithm, prompt, schema,
decoder, parser, generation parameter, retry, NBF, model-manager or rotation code was read-and-modified
— only read. No run was launched. The Phase 14.7 artifact
(`results/phase14_7_18_run_production_validation.jsonl`) was read only.

---

## STOP

Audit complete. **No fix was implemented**, per the phase's stated preference. Awaiting a decision on
§7.1–§7.4 before proceeding to Phase B of this phase.

## Caveats stated honestly

1. **The error-path analysis rests on a single observed error record.** The field-set asymmetry is
   confirmed by source inspection across both paths, so it is not merely inferred from n=1 — but the
   *within-path* variation of the error path cannot be characterised from one sample.
2. **`num_turns` proxy equivalence was not measured.** I declined to assert that
   `len(attack._history_attacker) == len(conv.turns)` and did not run an experiment to settle it; it
   would be settleable with one instrumented run if that matters.
3. **`_history_target` holding `"(Summary) "` prefixes** is established from source at three call
   sites, not measured against records. It is the reason reconstruction of `turns` is rejected.
4. **The 31 already-symmetric fields were verified by presence, not by value-equality** between paths.
5. **No recommendation is made here about the attacker's 99 % JSON reliability.** That is out of scope
   for this phase, and no model change is proposed.
