# PHASE 38 — FINAL PRE-EXECUTION HARDENING FOR DELIVERABLE 1

**Scope:** the P2 items retained by the Phase 37 independent audit.
**Mode:** implementation + offline validation. No paid call, no qualification, no experiment, no
model inference, no GPU allocation.

| Item | Outcome |
|---|---|
| **P2-1** ASR reporting | **IMPLEMENTED** — per-attack table + labelled overall micro-average |
| **P2-6** Dataset row-count validation | **IMPLEMENTED** — fail-closed count, shape and limit checks |
| **P2-3** R1 qualification runbook | **IMPLEMENTED** — `docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` |
| **P2-4** Identity completeness | **PARTIALLY IMPLEMENTED** — `structured_output` bound; five other fields assessed and deliberately not bound |
| **P2-2** Gate placement | **ASSESSED, NOT CHANGED** — construction-level enforcement would require a factory-signature redesign; residual risk documented |

**Regression:** 31 new tests + 590 existing offline tests = **621 passed, 0 failed, 0 skipped**.
**Methodology:** unchanged (`git diff -- configs/` empty; checkpoint and dataset hashes unchanged).

**Final verdict: `READY FOR SEPARATELY AUTHORIZED QUALIFICATION`.**

---

## A. Executive summary

Phase 37 found no P0/P1 defects and listed ten P2s. This phase resolves the four that affect how
results are read and how the run starts, leaves the rest documented, and adds no methodology
change.

**Implemented**

1. **P2-1 — ASR reporting.** `cloud_pilot.render_asr_report()` renders one row per attack with that
   attack's own 200-behaviour denominator and both rates, then a separately labelled
   `Overall micro-average` block that states the 600 attack-behaviour pair count. Values are read
   verbatim from `aggregate_asr_report`; the renderer performs no arithmetic. The manifest and JSON
   schemas are untouched — `run.asr_by_attack` and the aggregate fields were already present.
2. **P2-6 — dataset integrity.** `load_goals` now refuses a dataset with fewer rows than
   `dataset.num_goals`, rejects non-positive counts, negative limits, and selected rows that are
   not mappings carrying a non-empty `task`/`goal`. The deterministic prefix order is preserved,
   and the check runs before any client is constructed.
3. **P2-3 — runbook.** `docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` documents the exact R1 stack, the
   paid qualification command, the expected report path and fields, how staleness is rejected, how
   the report is passed to the experiment, and the four execution modes. It marks qualification
   and the experiment as paid, separately authorized operations.

**Assessed, deliberately limited**

4. **P2-4 — identity completeness.** Only `structured_output` was added to the qualification
   identity, because it changes the API contract the probes exercise. `max_turns`, dataset
   path/hash, NBF activation, checkpoint and threshold were assessed and left out with reasons
   (§F.2). The writer now records the field; reports produced before this change would be rejected
   as stale — none exist.
5. **P2-2 — gate placement.** The gate stays in the CLI entry point. Moving it into
   `build_role_llms` would require threading a report path through a shared factory, would
   special-case the offline-mock and test paths, and risks double-gating the qualification probes.
   That is an architectural change beyond this phase. Residual risk documented in §F.1.

**Nothing was executed.** No qualification, no paid client, no experiment. The next step is the
separately authorized qualification run described in the runbook.

---

## B. Change inventory

Six files. `git diff -- configs/` is empty.

| # | Path | Change | Phase 37 finding |
|---|---|---|---|
| 1 | `scripts/_cloud_common.py` | `load_goals`: added `num_goals >= 1`, `len(cases) >= num_goals`, non-negative `limit`, and per-row shape checks; raised explicit `ValueError`s | P2-6 |
| 2 | `scripts/_cloud_common.py` | `expected_qualification_identity`: added `structured_output`; `require_qualification`: compares it and reports it in the staleness message | P2-4 |
| 3 | `scripts/cloud_model_qualification.py` | `_base_report` writes `configuration_under_qualification.structured_output` | P2-4 |
| 4 | `scripts/cloud_pilot.py` | new `render_asr_report(asr, attacks)`; the three legacy ASR print lines are replaced by one call | P2-1 |
| 5 | `tests/test_phase36_implementation.py` | fixtures updated to carry `structured_output: "json_object"` so the new identity check is satisfied | P2-4 (compatibility) |
| 6 | `tests/test_phase38_hardening.py` | **new**, 31 tests | P2-1 / P2-3 / P2-4 / P2-6 |
| 7 | `docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` | **new** operational document | P2-3 |

No attack code, prompt, NBF component, evaluator, dataset or configuration was modified.

---

## C. ASR reporting verification

### C.1 Console output

Exact output of `render_asr_report` for a synthetic 3 × 200 aggregate (40/195, 50/200, 30/185):

```
ASR by attack (primary denominator = configured behaviours per attack)
  attack           successful  expected   primary   valid  diagnostic
  actor_attack             40       200    0.2000     195      0.2051
  crescendo_paper          50       200    0.2500     200      0.2500
  opposite_day             30       200    0.1500     185      0.1622
  primary = successful / expected; diagnostic = successful / valid completed

Overall micro-average across 3 attacks (600 expected attack-behaviour pairs)
  primary ASR     0.2000 (120/600)
  diagnostic ASR  0.2069 (120/580)
```

* Every attack name appears with its own `expected` column (`200`).
* `primary` is `successful_goals / expected_goals` — the paper's per-attack denominator.
* `valid` and `diagnostic` are the independent diagnostic denominator.
* The aggregate is labelled **`Overall micro-average`** and states the **600** pair count.
* A zero diagnostic denominator renders `n/a`, never `0.0000`.

### C.2 Source

`scripts/cloud_pilot.py:55-88` (`render_asr_report`), called at `:799`; `_fmt` at `:50-52` already
maps `None` to `n/a`. The function reads only fields produced by
`_cloud_common.aggregate_asr_report`: `by_attack[<name>].{successful_goals, expected_goals,
valid_completed_goals, asr_paper, asr_valid_completed}` and `overall.{...}`.

The previous three console lines (overall-only) are replaced; no other output changed.

### C.3 Tests

| Test | Asserts |
|---|---|
| `test_asr_report_lists_all_three_attacks` | all three names present |
| `test_each_attack_shows_its_own_200_denominator` | token 2 of each row is `200` |
| `test_primary_asr_uses_the_per_attack_denominator` | `0.2000` / `0.2500` / `0.1500`, equal to the aggregate values |
| `test_diagnostic_asr_is_reported_separately` | `195` / `0.2051` and differs from primary |
| `test_overall_is_labelled_micro_average_with_600_pairs` | label text, `600`, `(120/600)`, `(120/580)` |
| `test_zero_diagnostic_denominator_renders_not_available` | `n/a (0/0)`, no fabricated zero |
| `test_renderer_does_not_change_the_aggregation_values` | the dict is byte-identical before/after; overall counts exact |
| `test_renderer_reads_only_keys_the_aggregator_provides` | no new schema dependency |

All pass. JSON/manifest compatibility is unchanged: no key was added, removed or renamed.

---

## D. Dataset integrity verification

### D.1 Enforcement

`scripts/_cloud_common.py::load_goals` (Phase 38 block):

```python
requested = int(num_goals)
if requested <= 0:                       raise ValueError(...)
if len(cases) < requested:               raise ValueError("dataset contains N behaviours ...")
selected = cases[:requested]             # deterministic file-order prefix (unchanged)
if limit is not None:
    if int(limit) < 0:                   raise ValueError("--limit must be >= 0")
    selected = selected[:limit]
for index, case in enumerate(selected):
    if not isinstance(case, dict) or not (case.get("task") or case.get("goal")):
        raise ValueError("selected row {index} ...")
```

* The count check uses `dataset.num_goals` (200 for R1), **not** `--limit`, so a partial run cannot
  mask a short dataset.
* Selection remains the file-order prefix; ordering and contents are untouched.
* Validation happens at `cloud_pilot.main` line 427, which is **before** `--dry-run`, the
  qualification gate and `build_role_llms`. A short dataset therefore cannot construct a client.
* `dataset_provenance` still records the dataset SHA-256; nothing about hashing changed.

### D.2 Failure behaviour

| Case | Result |
|---|---|
| exactly 200 rows | accepted, 200 selected |
| more than 200 (e.g. 250) | accepted, first 200 selected in file order |
| 199 rows, `num_goals: 200` | `ValueError: dataset contains 199 behaviours but dataset.num_goals requires 200` |
| empty file, `num_goals: 200` | `ValueError: dataset contains 0 behaviours …` |
| empty file, no `num_goals` | `ValueError: dataset.num_goals must be >= 1` |
| selected row is not a mapping | `ValueError: selected row 1 is not a mapping …` |
| row lacks `task`/`goal` | `ValueError: … non-empty 'task' or 'goal'` |
| `num_goals: 0` or negative | `ValueError: num_goals must be >= 1` |
| negative `--limit` | `ValueError: --limit must be >= 0` |

### D.3 Tests

`test_exactly_200_rows_is_accepted`, `test_more_than_200_rows_keeps_the_deterministic_prefix`,
`test_199_rows_is_rejected`, `test_empty_dataset_is_rejected`,
`test_empty_dataset_without_an_explicit_count_is_rejected`, `test_invalid_row_shape_is_rejected`,
`test_row_without_a_goal_field_is_rejected`, `test_zero_requested_count_is_rejected`,
`test_negative_requested_count_is_rejected`, `test_negative_limit_is_rejected`,
`test_limit_selects_a_prefix_without_weakening_the_count_check`,
`test_dataset_failure_precedes_client_construction`, and
`test_real_r1_dataset_still_loads_200_goals` (the real pinned file loads 200). All pass.

The frozen HarmBench file was not modified: SHA-256 remains
`ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb`.

---

## E. Qualification runbook

`docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` documents:

* **A — required R1 configuration**: the full table (three roles at `gpt-4o-2024-08-06`,
  `official`, `baseline`, 0.7/0.7/0.7, `json_object`, HarmBench 200, the three attacks, Acronym
  training-only, 8 turns, checkpoint hash, η, `h < −η`, `steer_target: false`, both ASR
  denominators).
* **B — qualification**: the exact command

  ```bash
  python scripts/cloud_model_qualification.py \
    --config configs/paper_reproduction_r1_gpt4o.yaml \
    --confirm-paid-request
  ```

  marked **paid and separately authorized**; the expected report path
  `results/qualification/gpt-4o-2024-08-06.baseline.qualification.json`; the required report fields
  (`overall`, `probes_complete`, models, prompt variant, temperatures, structured output); a local
  inspection snippet; the identity-based staleness rule; and the `--qualification-report` flag for
  the experiment.
* **C — experiment authorization**: a QUALIFIED report is a prerequisite, not authorization.
* **D — four execution modes**: offline mock, dry run, qualification and paid experiment, with a
  table of cost / client construction / qualification requirement. The two paid commands appear
  only under a "require separate authorization" heading.
* **E/F**: a no-spend preflight checklist and how to read the ASR table.

The document explains why the default `configs/cloud_qualification.yaml` is *not* suitable for R1
(`gpt-4o-mini`, evaluator 0.0 → correctly rejected as stale).

**Runbook consistency tests** (all pass): exact qualification command present; report path and
verdict present; paid/authorization wording present; four modes distinguished; both ASR
denominators documented; and a cross-check that the values in the runbook match
`configs/paper_reproduction_r1_gpt4o.yaml`.

**Not executed.** No command from the runbook was run.

---

## F. Gate and identity assessment

### F.1 P2-2 — construction-level gate placement: **not changed**

Investigated.

* `_cloud_common.build_role_llms` (`:376-398`) is a generic factory used by `cloud_pilot.main`
  (`:493`) and by `tests/test_attacker_arms.py` (`:238, :254`). It accepts
  `(cfg, ledger)` only — no qualification report, no experiment identity.
* The qualification harness builds its clients through `build_single_role_llm`, not through this
  factory, so moving the gate would not double-gate qualification.
* Enforcing inside the factory would require either a new required parameter (breaking both
  existing callers and every script that uses it), or reading the default report path implicitly —
  which would break offline-mock injection and any test that constructs clients without a report.

**Decision: leave the CLI gate in place; do not attempt a partial construction-level check.**
Residual risk: a programmatic caller that invokes `run_one`/`build_role_llms` directly, bypassing
`main`, would not pass the gate. That is outside the supported entry point (`scripts/cloud_pilot.py`)
and is recorded as P2-2 in §I. The existing CLI gate is not downgraded.

### F.2 P2-4 — identity completeness: **one field added**

| Field | Classification | Decision |
|---|---|---|
| `cloud.structured_output` | **Required for API-contract identity** — the probes parse JSON contracts; `json_object` and `json_schema` are different request contracts | **Bound**: writer records it, gate compares it |
| `attacks.max_turns` | Experiment topology; does not change whether the model can serve the role | Not bound |
| `dataset.path` / hash | Experiment reproducibility. The run records its own hash and the loader now fails closed on a short file | Not bound |
| `nbf.enabled` | Defense configuration; both arms use the same cloud stack, and qualification is about the stack | Not bound |
| `nbf.checkpoint` | Local probe identity. The run verifies its own checkpoint SHA-256 in `verify_nbf_checkpoint` before any client is constructed | Not bound; covered by checkpoint verification |
| `nbf.eta` / threshold | Inference-time margin, not model identity | Not bound |

Binding all of these would make a legitimate report stale for irrelevant reasons (e.g. changing
`max_turns` for a smoke run) without adding evidence about the model roles. The one field with a
genuine API-contract justification was added and tested.

Consequence to note: reports produced before this phase do not carry `structured_output` and would
be rejected as stale. No such report exists (`results/qualification/` is absent), so there is no
migration burden.

---

## G. Regression results

Environment: Windows, `PYTHONDONTWRITEBYTECODE=1`, `KMP_DUPLICATE_LIB_OK=TRUE`,
`-p no:cacheprovider`. Every test module used here was inspected first: scripted fakes, synthetic
fixtures, monkeypatched client builders, no network, no model load, no GPU.

### G.1 New tests only

```
python -m pytest tests/test_phase38_hardening.py -q --no-header -p no:cacheprovider
```
**31 collected / 31 passed / 0 failed / 0 skipped.**

### G.2 Phase 36 + Phase 38

```
python -m pytest tests/test_phase38_hardening.py tests/test_phase36_implementation.py -q
```
**63 passed** (31 + 32).

### G.3 Relevant offline union

```
python -m pytest \
  tests/test_phase38_hardening.py tests/test_phase36_implementation.py \
  tests/parity/test_actor_attack_parity.py tests/parity/test_crescendo_parity.py \
  tests/parity/test_opposite_day_parity.py tests/parity/test_refusal_threshold_parity.py \
  tests/test_attacks.py tests/test_phase2_attacks.py tests/test_phase14_8_telemetry.py \
  tests/test_phase22_paper_fidelity.py tests/test_phase24_p0_fixes.py \
  tests/test_phase26_qualification_gate.py tests/test_phase27_fail_closed_gate.py \
  tests/test_phase29_alignment.py tests/test_phase30_retry_provenance.py \
  tests/test_phase31_retry_semantics.py tests/test_attacker_arms.py \
  tests/test_cloud_provider.py tests/test_cloud_pricing_and_snapshots.py \
  tests/test_cloud_research_integration.py tests/test_cloud_qualification_harness.py \
  -q --no-header -p no:cacheprovider
```

| Metric | Value |
|---|---|
| Collected | 621 |
| Passed | **621** |
| Failed | 0 |
| Errors | 0 |
| Skipped | 0 |
| Duration | 21.9 s |

### G.4 Not executed

The four Phase 31 environment failures were not run: two require CUDA/VRAM allocation
(`test_phase14_2_rotation…`) and two require the absent local `Qwen/Qwen3.5-4B`
(`test_phase8_architecture…`). Their classification is unchanged. This is **not** a full-suite
pass: 621 of 1 277 tests were executed.

---

## H. Methodology integrity

| Parameter | Frozen value | Status |
|---|---|---|
| Target model | `gpt-4o-2024-08-06` | unchanged |
| HarmBench behaviours | 200 | unchanged; loader now fails closed below it |
| Main attacks | ActorAttack, Crescendo, Opposite-day | unchanged |
| Acronym | training-only | unchanged |
| NBF checkpoint SHA-256 | `cea1a75b…` | unchanged (recomputed) |
| NBF η | 0.001 | unchanged |
| Runtime admission | `h < −η` | unchanged |
| `steer_target` | `false` | unchanged |
| Attacker / target / evaluator temperature | 0.7 / 0.7 / 0.7 | unchanged |
| Maximum conversation length | 8 | unchanged |
| Primary ASR denominator | 200 per attack | unchanged |
| Diagnostic ASR | valid completed evaluations | unchanged |

`git diff -- configs/` is empty. No attack prompt, NBF component, evaluator or dataset byte
changed. The HarmBench file and checkpoint hashes match Phase 35/37.

---

## I. Remaining deviations

| ID | Item | Impact | Status |
|---|---|---|---|
| P2-2 | Gate is CLI-scoped; a direct programmatic caller could bypass it | None for the supported entry point; defence-in-depth gap | **open, documented** |
| P2-5 | `attacker_arm_name` is not canonicalised on the consumer side while the writer uses `resolve_arm` | Alias configs would be rejected (fail-closed) | open |
| P2-7 | `attacks.N_retry` is declared but not read (effective constant 3) | None behaviourally | open |
| P2-8 | `invariants_checked` records declared names, not verified values | Manifest wording | open |
| P2-9 | `acronym.py:188` generates without a temperature | Acronym is training-only | open, out of scope |
| P2-10 | Attack-ASR `comparison_group` omits models/temperatures | Full manifest carries them | open |
| DEV-1 | Four environment-only test failures | Unrelated to Deliverable 1 | unchanged |
| PREREQ-1 | No QUALIFIED report exists for the R1 stack | Paid path blocks by design | **next step, requires authorization** |

No P0 or P1 remains. P2-1, P2-3, P2-6 are closed; P2-4 is closed for `structured_output` with the
remaining fields assessed and justified.

---

## J. Final verdict

# `READY FOR SEPARATELY AUTHORIZED QUALIFICATION`

The ASR output is now unambiguous: one row per attack with the paper's 200-behaviour denominator,
a separately labelled diagnostic rate, and an explicitly labelled overall micro-average over 600
attack-behaviour pairs. Dataset selection fails closed before any client is constructed. The exact
R1 qualification procedure is documented, including the reason the default qualification config
must not be used.

No paper methodology or experiment behaviour changed. No qualification, paid client, experiment or
model inference was executed in this phase.

**Next step:** authorize, then run the qualification command in
`docs/DELIVERABLE_1_EXECUTION_RUNBOOK.md` §B. A `QUALIFIED` report does not authorize the
experiment; that is a separate decision.
