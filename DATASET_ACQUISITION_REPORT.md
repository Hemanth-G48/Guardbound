# DATASET ACQUISITION REPORT

**Phase:** acquisition executed under an explicit, acquisition-only authorization.
**Predecessors:** `PHASE_34_UTILITY_PROTOCOL_FINAL_AUDIT.md` (unchanged),
`PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md` (unchanged).

**Acquisition status: `COMPLETE — 4/4 FILES ACQUIRED AND BLOB-VERIFIED`**
**Dataset pin status: `UPSTREAM REVISION VERIFIED` / `PAPER VERSION UNVERIFIED`**
**Methodology approval: `NOT GRANTED`**
**Implementation: `NOT AUTHORIZED` / `NOT IMPLEMENTED`**
**Paid evaluation: `NOT AUTHORIZED`**
**Full Table 1: `NOT READY`**

Acquisition and verification only. **No source code, configuration, test or historical artifact was
modified. No model, judge or classifier was invoked. No paid API call was made and no credential
was read. No evaluation, qualification, smoke test or experiment was run. No dataset was
transformed, converted or reshaped.** Four data files and this report are the only new paths.

---

## 1. Executive summary

Four files were acquired from the pinned upstream revisions, verified against their published Git
blob SHA-1 values, hashed with SHA-256, and structurally validated.

| # | File | Bytes | Git blob SHA-1 | Blob match | SHA-256 |
|---|---|---:|---|---|---|
| 1 | MT-Bench `question.jsonl` | 48 929 | `464e2c22f7a021a0f9a584d232338a9a17c40066` | **yes** | `119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7` |
| 2 | MT-Bench `reference_answer/gpt-4.jsonl` | 51 109 | `8b66387a3129cbb117178fe515491df553dd52df` | **yes** | `f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f` |
| 3 | FastChat `judge_prompts.jsonl` (harness) | 10 457 | `d1e499f12d18c57338d883d43f5735e512c00d8c` | **yes** | `8baddf43f95f8c1a88166163b171f1cd68e9d8ce8606f867734d4b0efe38b98d` |
| 4 | XSTest `xstest_prompts.csv` | 38 719 | `ad38204d70b3964387f9690169d11afbc4756d01` | **yes** | `11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a` |

149 214 bytes total. All four byte counts match the values recorded in Phase 32 from metadata
alone, and all four content hashes match the published blobs — so the pins were correct before
acquisition and the acquired bytes are the pinned bytes.

Structural validation passed on every file: MT-Bench 80 questions (ids 81–160, 8 categories × 10,
two turns each); reference answers 30 rows covering exactly math/reasoning/coding; judge prompt
harness 8 entries with all four required single-answer templates; XSTest 450 prompts with a
250-safe / 200-unsafe split and a consistent `contrast_` ⇔ `unsafe` label rule.

**Three findings** materially affect the pending specification and are recorded in §8:

* **F1 — The referenced Phase 34.2 specification does not exist.** The authorization names
  "the approved Phase 34.2 specification", but no `PHASE_34_2_*` file exists in the repository.
  Acquisition proceeded against the pins established in Phases 31–32, which Phases 34 and 34.1
  did not alter. This is a documentation gap, not an acquisition ambiguity.
* **F4 — The judge system message is not a single string.** It is 28 bytes for `single-v1` and
  `single-math-v1`, 613 bytes for `single-v1-multi-turn`, and 640 bytes for
  `single-math-v1-multi-turn`. The Phase 34/34.1 identity's single `judge.system_message_sha256`
  field is therefore under-specified and must become a per-template map.
* **F5 — The dataset's own `reference` field is a 2-element list, not a turn-2 question or a
  reference answer.** It is non-null on 39 of 80 rows across six categories. The frozen MT-Bench
  loader's `turn2_question or reference or …` fallback would therefore inject a *list* into the
  turn-2 message. The Phase 34 rule "never derive turn 2 from `reference`" is now backed by a
  concrete type-level failure path, not only by provenance reasoning.

Nothing else changed. The datasets are now present; no protocol is approved, no code is
authorized, and no paid call is permitted.

---

## 2. Authorization, scope, and the Phase 34.2 discrepancy

### 2.1 What was authorized

| # | Authorized item | Scope |
|---|---|---|
| 1 | MT-Bench `question.jsonl` and `reference_answer/gpt-4.jsonl` from the pinned revision | two data files |
| 2 | XSTest `xstest_prompts.csv` from the pinned commit | one data file |
| 3 | FastChat `judge_prompts.jsonl` from the pinned blob | one harness file |

Requirements imposed: verify exact revisions and Git blob hashes; compute SHA-256 for every file;
record sizes and provenance; preserve historical artifacts; do not modify source, config or tests;
do not invoke models or judges; no paid calls; no implementation or evaluation; one consolidated
report.

### 2.2 The Phase 34.2 reference

The authorization refers to "the approved Phase 34.2 specification". Repository inspection shows:

```
PHASE_34_UTILITY_PROTOCOL_FINAL_AUDIT.md             66 428 bytes
PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md        49 960 bytes
(no PHASE_34_2_* file)
```

**No Phase 34.2 document was ever produced.** The work that phase was scoped to perform — request
accounting, schema closure, arm-configuration/telemetry separation, ID-set semantics — is
therefore still outstanding as a *specification* matter. It does not affect this acquisition,
because every value used here was fixed before it: Phase 31 pinned the FastChat tag, dataset
commit and per-file blobs; Phase 31/32 pinned the XSTest commit and blob; Phase 32 fixed the
target local layout; Phases 34 and 34.1 changed none of them.

This report does not create Phase 34.2 and does not substitute for it. It is recorded here so the
authorization chain is not retroactively rewritten.

### 2.3 What was not authorized, and was not done

No methodology approval, no implementation, no evaluation, no paid call, no credential read, no
dataset transformation. §10 records compliance item by item.

---

## 3. Pinned revisions

| Item | Value | Source of the pin |
|---|---|---|
| FastChat repository | `https://github.com/lm-sys/FastChat` | Phase 31 |
| FastChat release tag | `v0.2.36` | Phase 31 |
| Tag commit | `b21d0f780ca4472a13714262a0790f2ee1ade659` | Phase 31 |
| MT-Bench **data** commit | `b494d0c6b4e7935f1764f8439e75da3e66beccc7` | Phase 31 |
| XSTest repository | `https://github.com/paul-rottger/xstest` | Phase 31 |
| XSTest commit | `475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5` | Phase 31 (no tags exist) |

Why two FastChat references: `v0.2.36` is the release whose **code** defines the judging protocol;
`b494d0c6…` is the commit that last touched the **data** files. Fetching the data at
`b494d0c6…` and matching the blobs recorded against `v0.2.36` simultaneously confirms the Phase 31
claim that the data files are identical at both references.

`judge_prompts.jsonl`, previously described only as "a harness file", is present at the **dataset
commit** `b494d0c6…` (HTTP 200, blob matched). It does not require a separate revision pin.

**Paper version remains unverified for both benchmarks.** Every file here is pinned to a
defensible upstream revision, not to a revision proven to be the one used by the NBF authors. The
status string `PAPER VERSION UNVERIFIED` is carried forward unchanged.

---

## 4. Acquisition records

| Field | Value |
|---|---|
| Retrieval started / completed (UTC) | 2026-11-11T12:17:31Z |
| Transport | HTTPS `raw.githubusercontent.com`, `Invoke-WebRequest` writing raw bytes |
| Authentication | none |
| Staging | session scratch directory (outside the repository) |
| Install step | byte copy into `data/utility/…` **only after** the Git blob SHA-1 matched |
| HTTP results | 4 × HTTP 200 |
| Files requested | 4 |
| Files received | 4 |
| Files rejected | 0 |

Exact request URLs:

```
https://raw.githubusercontent.com/lm-sys/FastChat/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl
https://raw.githubusercontent.com/lm-sys/FastChat/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl
https://raw.githubusercontent.com/lm-sys/FastChat/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/judge_prompts.jsonl
https://raw.githubusercontent.com/paul-rottger/xstest/475f10bf0a3d6a9dfb174b6de1a38afbfdff98a5/xstest_prompts.csv
```

The two-step (stage → verify → install) procedure was used so that a hash mismatch would abort
before any file entered the repository. No mismatch occurred, so no file was rejected or deleted.

No GitHub contents API, git-trees API, or archive download was used. No repository was cloned.
Only the four named files were requested; no directory listing was fetched.

---

## 5. Hash verification

Method: the Git blob identity is
`SHA-1("blob " + decimal_length + "\0" + content)`, computed over the exact received bytes.
SHA-256 is computed over the same bytes. Byte counts are the on-disk size.

| Local path | Upstream path | Ref | Expected blob | Computed blob | Match |
|---|---|---|---|---|---|
| `data/utility/mtbench/questions/question.jsonl` | `fastchat/llm_judge/data/mt_bench/question.jsonl` | `b494d0c6…` | `464e2c22f7a021a0f9a584d232338a9a17c40066` | `464e2c22f7a021a0f9a584d232338a9a17c40066` | **✓** |
| `data/utility/mtbench/reference_answers/gpt-4.jsonl` | `fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl` | `b494d0c6…` | `8b66387a3129cbb117178fe515491df553dd52df` | `8b66387a3129cbb117178fe515491df553dd52df` | **✓** |
| `data/utility/mtbench/harness/judge_prompts.jsonl` | `fastchat/llm_judge/data/judge_prompts.jsonl` | `b494d0c6…` | `d1e499f12d18c57338d883d43f5735e512c00d8c` | `d1e499f12d18c57338d883d43f5735e512c00d8c` | **✓** |
| `data/utility/xstest/prompts/xstest_prompts.csv` | `xstest_prompts.csv` | `475f10bf…` | `ad38204d70b3964387f9690169d11afbc4756d01` | `ad38204d70b3964387f9690169d11afbc4756d01` | **✓** |

| Local path | Bytes | Expected bytes | Match | SHA-256 |
|---|---:|---:|---|---|
| `…/mtbench/questions/question.jsonl` | 48 929 | 48 929 | **✓** | `119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7` |
| `…/mtbench/reference_answers/gpt-4.jsonl` | 51 109 | 51 109 | **✓** | `f957a5bc977badb66885ec970e6cd08527845780313f0995764260e5777b9b3f` |
| `…/mtbench/harness/judge_prompts.jsonl` | 10 457 | 10 457 | **✓** | `8baddf43f95f8c1a88166163b171f1cd68e9d8ce8606f867734d4b0efe38b98d` |
| `…/xstest/prompts/xstest_prompts.csv` | 38 719 | 38 719 | **✓** | `11783fb294ed017473ee53c207d71f2161c7672c8d0b037501e78387f801cb5a` |

Encoding facts recorded during verification: all four files are **UTF-8 without BOM** and contain
**no CRLF** sequences. Each of the two JSONL files and the CSV ends with a trailing newline
(MT-Bench: 80 lines; XSTest: 451 lines = 1 header + 450 rows). Non-ASCII characters occur in the
MT-Bench data (e.g. typographic apostrophes); byte identity is what the blob check establishes, so
display encoding is not a provenance risk.

**SHA-256 values above are new in this report.** Phase 32 could not compute them because it held
no bytes. They are the values a future integrity check must compare against.

---

## 6. Structural validation

### 6.1 MT-Bench `question.jsonl`

| Check | Expected | Observed | Result |
|---|---|---|---|
| Rows | 80 | 80 | **✓** |
| Distinct `question_id` | 80 | 80 | **✓** |
| ID range | 81–160 | 81–160 | **✓** |
| `turns` length per row | exactly 2 | exactly 2 for all | **✓** |
| Categories | 8 × 10 | coding, extraction, humanities, math, reasoning, roleplay, stem, writing — 10 each | **✓** |
| Top-level keys | — | `category`, `question_id`, `reference`, `turns` | recorded |

### 6.2 MT-Bench `reference_answer/gpt-4.jsonl`

| Check | Expected | Observed | Result |
|---|---|---|---|
| Rows | — | **30** | recorded |
| Distinct `question_id` | — | 30 | **✓** |
| Every ID exists in `question.jsonl` | yes | yes | **✓** |
| `choices` length | 1 | 1 for all | **✓** |
| `choices[0].turns` length | 2 | 2 for all | **✓** |
| Category coverage | exactly math, reasoning, coding | exactly those 3 (10 each) | **✓** |
| Questions without a reference | 50 (the other 5 categories) | 50 | **✓** |

This confirms upstream's `NEED_REF_CATS` scope empirically: reference-guided judging is
constructible for 30 of 80 questions and impossible for the other 50. A protocol that required a
reference for every question would fail on 50 rows.

### 6.3 FastChat `judge_prompts.jsonl`

| Check | Expected | Observed | Result |
|---|---|---|---|
| Entries | — | 8 | recorded |
| `single-v1` present | yes | yes | **✓** |
| `single-v1-multi-turn` present | yes | yes | **✓** |
| `single-math-v1` present | yes | yes | **✓** |
| `single-math-v1-multi-turn` present | yes | yes | **✓** |
| `output_format` for the four | `[[rating]]` | `[[rating]]` | **✓** |
| `type` for the four | `single` | `single` | **✓** |

All eight names, for completeness: `pair-v2`, `pair-v2-multi-turn`, `pair-math-v1`,
`pair-math-v1-multi-turn`, `single-v1`, `single-math-v1`, `single-v1-multi-turn`,
`single-math-v1-multi-turn`. The four pairwise entries are not used by the per-turn scoring
protocol.

### 6.4 XSTest `xstest_prompts.csv`

| Check | Expected | Observed | Result |
|---|---|---|---|
| Rows | 450 | 450 | **✓** |
| Columns | `id, prompt, type, label, focus, note` | identical | **✓** |
| Distinct `id` | 450 | 450 | **✓** |
| `label == safe` | 250 | 250 | **✓** |
| `label == unsafe` | 200 | 200 | **✓** |
| Prompt types | 18 × 25 | 18 × 25 | **✓** |
| `type` starts `contrast_` | 200 rows | 200 | **✓** |
| `contrast_` ⇔ `unsafe` on every row | yes | yes (0 violations) | **✓** |

Safe types (10): `definitions`, `figurative_language`, `historical_events`, `homonyms`,
`nons_group_real_discr`, `privacy_fictional`, `privacy_public`, `real_group_nons_discr`,
`safe_contexts`, `safe_targets`.
Unsafe (8, all `contrast_`): `definitions`, `discr`, `figurative_language`, `historical_events`,
`homonyms`, `privacy`, `safe_contexts`, `safe_targets`.

---

## 7. Provenance metadata

| Field | MT-Bench questions | MT-Bench references | Judge harness | XSTest prompts |
|---|---|---|---|---|
| Repository | `lm-sys/FastChat` | `lm-sys/FastChat` | `lm-sys/FastChat` | `paul-rottger/xstest` |
| Ref | commit `b494d0c6…` (also tag `v0.2.36`) | commit `b494d0c6…` | commit `b494d0c6…` | commit `475f10bf…` |
| Upstream path | `fastchat/llm_judge/data/mt_bench/question.jsonl` | `fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl` | `fastchat/llm_judge/data/judge_prompts.jsonl` | `xstest_prompts.csv` |
| Role | benchmark prompts | reference answers (3 categories) | judging harness (not a dataset) | benchmark prompts |
| Bytes | 48 929 | 51 109 | 10 457 | 38 719 |
| Git blob SHA-1 | `464e2c22…` | `8b66387a…` | `d1e499f1…` | `ad38204d…` |
| SHA-256 | `119565ad…` | `f957a5bc…` | `8baddf43…` | `11783fb2…` |
| Encoding | UTF-8, no BOM, LF | UTF-8, no BOM, LF | UTF-8, no BOM, LF | UTF-8, no BOM, LF |
| Licence | inherited from FastChat (not re-verified in this phase) | idem | idem | **CC-BY-4.0** (Phase 31 metadata) |
| Retrieval | 2026-11-11T12:17:31Z, HTTPS raw | idem | idem | idem |
| Paper version | `PAPER VERSION UNVERIFIED` | idem | idem | idem |

Licence note: FastChat's repository licence was **not** re-verified during this acquisition (it
requires reading a file that was not in the authorized set). XSTest's CC-BY-4.0 status comes from
the repository metadata recorded in Phase 31 and is unaffected. Attribution obligations should be
confirmed before any publication, not before evaluation.

Newly computed judge-template identities (from the acquired harness file):

| Template | `system_prompt` bytes | `system_prompt` SHA-256 | `prompt_template` bytes | `prompt_template` SHA-256 |
|---|---:|---|---:|---|
| `single-v1` | 28 | `75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de` | 641 | `0935c7ff79f38d3f9b135e14b2cf3bc05ddfbcb96cfd4094b1681a8044e9ccd0` |
| `single-v1-multi-turn` | 613 | `d1e77aa107f02dd829e2e9caee2ad227d0a501853fefa74ccd4b51f758cbd432` | 212 | `33067cf18206b4abdc13b27f53b5240b4ef64e9efa5a2fb4b51d91fb0aca60fd` |
| `single-math-v1` | 28 | `75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de` | 767 | `8e11dbbf609a3d6f69cc75c8f1f2a4932160a64499f6fd6e5f3916e8e54aa2b7` |
| `single-math-v1-multi-turn` | 640 | `4b8fe8086dbdf4a8bc64cdeec10ff6b348ec6c0578d668ad01604e47ae517baf` | 405 | `6a2b8ad2961c275c0afd7206cbf6bb22379309ded1032798464f3f4ae5176b03` |

Placeholder analysis: `system_prompt` contains **no** placeholders in any of the four templates, so
it is a static per-template string. `prompt_template` carries the per-example placeholders —
`{question}`, `{answer}` for `single-v1`; `{question_1}`, `{answer_1}`, `{question_2}`,
`{answer_2}` for `single-v1-multi-turn`; plus `{ref_answer_1}` /
`{ref_answer_1}`+`{ref_answer_2}` for the two math variants.

---

## 8. Findings

### F1 — The referenced Phase 34.2 specification does not exist

`PHASE_34_2_FINAL_ACCOUNTING_AND_SCHEMA_AUDIT.md` is absent from the repository. The authorization
names it as approved. Acquisition proceeded because every pin predates it and is unambiguous; this
report does not claim that Phase 34.2 was performed. **Consequence:** the request-accounting and
schema-closure work remains outstanding as a specification task, and the authorization register
still has no Phase 34.2 revision.

### F2 — The MT-Bench reference file covers 30 of 80 questions

`reference_answer/gpt-4.jsonl` contains 30 rows, exactly the 10 math + 10 reasoning + 10 coding
questions. The other 50 questions have no reference answer. This validates the category-conditional
design (reference-guided templates only for `NEED_REF_CATS`) and means a global reference-guided
protocol would be unimplementable from these bytes.

### F3 — `question.jsonl` carries a `reference` field of a different kind

| Property | Observed |
|---|---|
| Type | `null` or **list of 2 strings** |
| Rows non-null | 39 / 80 |
| Categories | coding 9, extraction 5, math 10, reasoning 10, roleplay 1, stem 4 |

It is neither a turn-2 question nor the reference-answer source. Two consequences:

1. It confirms the Phase 34 rule that turn 2 must come from `turns[1]`.
2. It exposes a concrete defect path in the frozen loader: `q2 = ex.turn2_question or ex.reference
   or "Please elaborate…"` would assign a **list** to a user message's `content`. Depending on the
   transport, that either raises or silently serializes the list. This is no longer only a
   provenance objection; it is a type error waiting for the first reference-bearing row.

### F4 — The judge system message is not a single string

Phase 34 and 34.1 model the judge system message as one field,
`judge.system_message_sha256`. The acquired harness shows four different static system messages:
28 B (`single-v1`), 613 B (`single-v1-multi-turn`), 28 B (`single-math-v1`), 640 B
(`single-math-v1-multi-turn`).

The multi-turn templates put the judging *instructions* in the system message and a short
conversation-opening string in the user template (212 B / 405 B), which is why the sizes invert
between families. A single hash field cannot represent this. **Required specification change (not
performed here):** `judge.system_message_sha256` must become a four-entry map, and
`judge.template_set` should carry both the user-template and the system-message hash per template.

### F5 — The same 28-byte string appears in three distinct roles

`You are a helpful assistant.` (28 bytes, SHA-256
`75357d685f238b6afd7738be9786fdafde641eb6ca9a3be7471939715a68a4de`) is:

1. the **target** generation system message chosen in Phase 34 §3.2 (a reproduction decision);
2. the **judge** system message for `single-v1` and `single-math-v1` (upstream fact);
3. the **XSTest detector** system message (upstream fact).

Three fields, one byte string. They are separately sourced and must remain separately recorded; a
manifest that collapses them would lose the ability to say *why* each is what it is. This confirms
the Phase 34 decision to keep the target and judge system messages as distinct identity fields.

### F6 — `judge_prompts.jsonl` is pinnable at the dataset commit

The harness file resolved at `b494d0c6…` with blob `d1e499f1…`, matching the value Phase 32
recorded from metadata. No additional revision pin is needed, and A15's acquisition requirement is
satisfied by the same commit as the MT-Bench data.

### F7 — A1/A2/A15 acquisition obligations are now discharged at the file level

The three acquisition items now have bytes on disk with verified identity. This discharges the
*fetching* obligation only. It does not approve the revisions as paper-faithful, does not approve
any methodology, and does not authorize implementation.

---

## 9. Local layout and integrity procedure

### 9.1 Installed layout

```
data/utility/
  mtbench/
    questions/question.jsonl            48 929 B
    reference_answers/gpt-4.jsonl       51 109 B
    harness/judge_prompts.jsonl         10 457 B
  xstest/
    prompts/xstest_prompts.csv          38 719 B
```

Differences from the Phase 32 §8.3 proposal, both deliberate:

* the harness file sits under `mtbench/harness/` rather than alongside the questions, so it is
  never mistaken for a dataset file;
* no `provenance.json`, `checksums.sha256` or transformed `xstest_prompts.jsonl` sidecar was
  created, because the authorization requested one consolidated report. The checksums above are the
  machine-readable record and can be materialized later without re-downloading.

### 9.2 Verification procedure for any future phase

1. Recompute `SHA-256` for each file and compare with §5. Any mismatch invalidates the run.
2. Optionally recompute the Git blob SHA-1 and compare with §3/§5 — it proves the bytes are the
   upstream bytes, not merely stable locally.
3. Assert the structural invariants of §6 (80 / 30 / 8 / 450; 250+200; `contrast_` ⇔ unsafe).
4. Abort on any mismatch; never regenerate, re-download silently, or "fix" a dataset in place.

No transformation is authorized. In particular, the XSTest CSV has **not** been converted to
JSONL; that conversion is an implementation step under A25 and must be deterministic and
byte-recorded when it happens.

---

## 10. Prohibited-operation compliance

| Constraint | Status |
|---|---|
| Modify source code | **complied** — no `.py` file touched |
| Modify configuration | **complied** — no YAML/TOML/`.gitignore` change |
| Modify tests | **complied** — none touched |
| Modify historical artifacts | **complied** — Phases 33, 34, 34.1 byte-identical |
| Download additional datasets | **complied** — exactly the 4 authorized files |
| Clone a repository | **complied** — none |
| Use a dataset-content API | **complied** — no API used; raw file URLs only |
| Invoke a model, judge or classifier | **complied** — none |
| Read credentials / make paid calls | **complied** — none |
| Run evaluation, qualification, smoke test, pilot or experiment | **complied** — none |
| Run pytest | **complied** — not run |
| Implement any change matrix | **complied** — none |
| Treat approval as extending beyond acquisition | **complied** — §11 and §13 bound it explicitly |

---

## 11. Effect on the pending specification (recorded, not implemented)

| Item | Before | After acquisition | Still required |
|---|---|---|---|
| A1 MT-Bench revision | pending + acquisition | **files present, blob-verified** | methodology approval of the pin; `PAPER VERSION UNVERIFIED` remains |
| A2 XSTest revision | pending + acquisition | **file present, blob-verified** | methodology approval; `PAPER VERSION UNVERIFIED` remains |
| A15 judge harness | pending + acquisition | **file present, blob-verified** | methodology approval of template adoption |
| `judge.system_message_sha256` | single field, value "pending A15" | now resolvable as **four** hashes (F4) | schema change → requires a specification phase; not done |
| `judge.template_set` | four template hashes | user-template hashes now known (§7) | adoption approval |
| Turn-2 sourcing | must use `turns[1]` | **reinforced** by F3 (list-typed `reference`) | implementation under A25 |
| Reference-guided scope | math/reasoning/coding | **empirically confirmed** (30/80) | implementation under A25 |

Nothing above is an approval. The acquisition satisfies "the bytes exist and are what we said they
are"; it does not satisfy "the protocol is approved" or "the code is authorized".

**One specification defect is now newly evidenced and must be resolved before the identity schema
can be called closed:** the per-template judge system-message map (F4). Until it is fixed, a
`score_protocol_identity.v1` instance cannot faithfully record the judge configuration.

---

## 12. Repository integrity record

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `PHASE_34_UTILITY_PROTOCOL_FINAL_AUDIT.md` | 66 428 | `c87511bf1baf9378c176f90602a6ec4e7b6b859d6b5fe1608b7150df174229d2` | **unchanged** |
| `PHASE_34_1_FINAL_SPECIFICATION_CORRECTIONS.md` | 49 960 | `49f32395cd5255507d5d7e8366a6723c4aa2ceb062860b2bb2b9f16a462fad84` | **unchanged** |
| `PHASE_33_UTILITY_DECISION_LOCK_AND_IMPLEMENTATION_SPEC.md` | 42 122 | `66f98ed82a522dfec18ea8fca526ba5ebd141095fe3d5a2118618594e12bfcd6` | **unchanged** |

New paths created by this phase:

```
?? data/utility/                      (4 files, 149 214 bytes)
?? DATASET_ACQUISITION_REPORT.md      (this report)
```

`git status --porcelain` count: **107 → 109** (two new entries; Git collapses the acquired tree
into the single `data/utility/` entry). No tracked file was modified; `git status --porcelain` for
every previously modified path is unchanged.

Note on tracking: `.gitignore` covers `data/raw/*` and `data/processed/*` but **not**
`data/utility/*`, so the acquired files appear as untracked. `.gitignore` was **not** modified —
that would be a configuration change outside this authorization. Whether these bytes should be
committed, ignored, or stored via DVC/LFS is a separate decision for a future phase.

No dataset body was altered, re-encoded, normalized or re-saved after hashing: the installed files
were copied with `Copy-Item` and re-hashed at the destination, where all four SHA-256 values
matched the staged values.

---

## 13. Final status and next authorization boundary

| Component | Status | Basis |
|---|---|---|
| **Dataset acquisition** | **`COMPLETE — 4/4 VERIFIED`** | blob SHA-1, byte size and SHA-256 all verified (§5); structure validated (§6) |
| **Dataset provenance** | **`PINNED`** | exact revisions and per-file identities recorded (§3, §7) |
| **Paper version fidelity** | **`UNVERIFIED`** | the authors' releases are still unstated in the paper |
| **Phase 34.2 specification** | **`ABSENT`** | referenced by the authorization but never produced (§2.2) |
| **MT-Bench methodology** | **`NOT APPROVED`** | pin is available; A3/A6/A7/A8/A22/A26 unchanged |
| **XSTest methodology** | **`NOT APPROVED`** | pin is available; A10–A14, A20, A21 unchanged |
| **Shared score identity** | **`BLOCKED — JUDGE SYSTEM-MESSAGE MAP REQUIRED`** | F4; needs a specification correction |
| **Utility implementation** | **`NOT IMPLEMENTED`** | no code change authorized or made |
| **Paid evaluation** | **`NOT AUTHORIZED`** | no call made |
| **Full Table 1** | **`NOT READY`** | data present, protocol unapproved, implementation absent |

**What this authorization did:** put the four pinned files on disk with verified identity.
**What it did not do:** approve a protocol, authorize code, authorize spending, or close the Phase
34.2 specification gap.

The next authorization boundary is unchanged: methodology approval, implementation authorization
and paid evaluation remain three separate decisions, and none is implied by this acquisition.
