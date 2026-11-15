# Phase 15 — 30-Goal Pilot Report (Track A, Frozen)

**Date:** 2026-09-23 → 2026-09-24
**Command:** `python scripts/phase14_full_reproduction.py --batch 0 --limit 30 --attack all --condition both --config configs/reproduction_phase14_frozen.yaml --structured-output-mode constrained_json`
**Status:** **PILOT ABORTED at 108 / 180 records (goals 0–17 of 30).** See §14.

---

## 0. Correction to the record

**The pilot did not complete.** It ran for **12.9 hours** (16:53Z → 05:45Z) and then **exited with
code 1** at the start of goal 18. Final state:

| | |
|---|---|
| Records produced | **108 / 180 (60.0 %)** |
| Goals completed | **0–17 of 30** |
| Last record written | `OFF_acronym_017` |
| Next run that never started | `ON_crescendo_018` |
| Exit code | **1** (unhandled exception → clean abort) |

This report replaces the earlier INTERIM version, which was written at 11/180 while the run was
still healthy.

## 14. GO / NO-GO DECISION

```
BLOCKED — per-run VRAM guard false positive aborted the pilot at 108/180 (goals 0–17):
          the gate read raw driver-free VRAM (4.04 GB) without first releasing PyTorch's
          reusable cached blocks, while effective availability was 15.83 GB.
```

**This is an `INFRASTRUCTURE_FAILURE` in my own guard implementation — not a model-capability
failure, not a parser failure, not an algorithm failure, and not data corruption.** The distinction
matters and is evidenced in §14.2: every experiment-integrity check passed on all 108 records; the
abort came from a *conservative gate* reading the wrong quantity.

**Do NOT launch the 1,200-run Track A study.** Fix the gate first (§14.4), then resume the pilot
(it is resumable) to 180/180.

## 1. Pilot objective

A production-pipeline validation gate: prove that the frozen pipeline records, executes and
distinguishes results correctly before scaling from 30 goals to 200. Not an ASR study, and no tuning
of any kind.

## 2. Exact configuration — pre-flight verified

All **22 checks PASS** before the first run:

| Item | Value | Status |
|---|---|---|
| attacker | `Qwen/Qwen3-4B-Instruct-2507` | OK |
| target | `microsoft/Phi-4-mini-instruct` | OK |
| evaluator | `meta-llama/Llama-3.2-3B-Instruct` | OK |
| three distinct model ids | 3/3 | OK |
| dtype / quantization | `bfloat16` / `none` | OK |
| max_rounds / threshold / seed | `8` / `0.0` / `42` | OK |
| embedder | `all-mpnet-base-v2` | OK |
| **NBF checkpoint SHA256** | `cea1a75b…2a136fe` — **matches the frozen spec** | OK |
| initial state / dims / classes | `zeros(768)` / 768 / 5 | OK |
| **filter_trials** | `{crescendo: 3, opposite_day: 3, acronym: 3}`, `actor_attack` absent | OK |
| dataset SHA256 | `ac789de8…f125d014eb` | OK |

**Track A only** — no attacker retry, no parser repair, no fallback, no reuse. A malformed attacker
reply is a `JSON_PARSE_ERROR` and nothing else.

## 3. 30-goal dataset verification

Official HarmBench test set (200 goals) from the frozen path, **original file order**, no shuffle, no
reordering, no difficulty selection. Pilot slice = **goals 0–29**. Goals 0–17 were reached.

## 4. 180-run matrix and output separation

`30 goals × 3 attacks × 2 conditions = 180`; ActorAttack excluded. Per-goal order: Crescendo ON/OFF,
OppositeDay ON/OFF, Acronym ON/OFF — the runner's existing semantics, unmodified.

The runner writes to `results/phase14/batch00.jsonl` (hardcoded; no `--out` flag). To honour §5
without modifying the runner:

1. the Phase 14.2 evidence was **backed up** first (`batch00_phase14_2_backup.jsonl`, 18 records);
2. the pilot wrote to a clean `batch00.jsonl` — **required**, because the old file's
   `run_id`s (`ON_crescendo_000` etc.) collide with goal 0's and the resume logic keys on `run_id`,
   so leaving it would have silently **skipped goal 0's six runs**;
3. **not yet done** (the pilot aborted before completion): relocate to
   `results/phase15_pilot_30/trackA_frozen/`, annotate with `experiment`/`track`, restore the 14.2
   backup.

## 5. Execution integrity — 108 records

| Check | Result |
|---|---|
| records | **108** |
| unique run_ids | **108** |
| duplicate run_ids | **0** |
| **schema violations** | **0 / 108** — 55/55 canonical fields every record |
| run_ids unexpected vs the expected matrix | **0** |
| **NBF OFF contamination** | **0 / 54** |
| **`filter_trials` nulls** | **0 / 108** — `{3: 108}` |
| model identity violations | **0** |
| dtype / quantization deviations | **0** |
| **load refusals** | **0 / 1901 loads** |
| loads / evictions | **1901 / 1900** (balanced, no leak) |
| OOM / CUDA errors | **0** |

## 6. JSON failure analysis

**4 of 108 runs** ended in `JSON_PARSE_ERROR`. Every one satisfies the frozen telemetry contract:

| Requirement | Observed |
|---|---|
| classified `JSON_PARSE_ERROR`, never `ATTACK_FAILURE` | **4 / 4** |
| **never encoded as `turns = 0`** | **4 / 4 have `turns = null`** |
| raw attacker output preserved | **4 / 4** |
| schema complete | **4 / 4** (55/55 keys) |

Per-call reliability over the whole pilot:

| Role | JSON calls | parse failures | reliability |
|---|---:|---:|---:|
| **evaluator** (`Llama-3.2-3B-Instruct`) | 1581 | **0** | **100.0 %** |
| **attacker** (`Qwen3-4B-Instruct-2507`) | 1164 | **14** | **98.8 %** |

Consistent with Phase 14.7 (attacker 99.0 %, evaluator 100 %) — the attacker is the
reliability-limiting component, handled correctly every time, with **no retry or repair**.

## 7. NBF OFF / ON validation

| Condition | n | `nbf_enabled` | total `nbf_scores` | total `filtered_queries` |
|---|---:|---|---:|---:|
| **OFF** | 54 | all `False` | **0** | **0** |
| **ON** | 54 | all `True` | populated | populated |

**Zero OFF contamination** across 54 OFF runs — the §7/§13 hard stop never triggered.
The recurring `ON_acronym_*` runs stopping at 0 turns with 7 filtered are the barrier working as
designed (every candidate filtered), matching Phase 13/14.2/14.7.

## 8. `filter_trials` validation

`{3: 108}` — **`filter_trials = 3` on every record, zero nulls**, including all 4 failure records.
Crescendo previously reported `null`; the Phase 14.8.x fix holds under 12.9 hours of production load.

**Semantics reminder:** `3` is the declared frozen configuration value, not a claim that three NBF
evaluations run per candidate. Execution is carried by `nbf_scores` / `filtered_queries` /
`filter_count`. The `range(3)` loops are untouched.

## 9. Model / GPU validation

| Check | Result |
|---|---|
| attacker / target / evaluator identity | single value each, across all 108 |
| substitution / duplicate identity | **none** |
| loads / evictions / refusals | 1901 / 1900 / **0** |
| **peak VRAM (max over 108 runs)** | **19.904 GB** of 25.763 (77 %) |
| max GPU temperature | ~66 °C (driver target 84 °C) |
| OOM / CUDA errors | **0** |

## 10. Schema validation

**55/55 canonical fields on all 108 records**, including the 4 failure records. Zero violations.

## 11. Runtime summary

| Metric | Value |
|---|---:|
| model runtime total | **12.86 h** |
| mean per run | 429 s |
| wall-clock total | 12.9 h |
| **extrapolated for 180 runs** | **≈ 21.5 h** |

Per-goal cost was ~43 min of model runtime plus rotation overhead (~17.6 loads/run × ~9 s).
The earlier 30-hour estimate was too pessimistic; **21.5 h** is the measured rate.

## 12. Attack outcome summary

| Outcome | Count |
|---|---:|
| `SUCCESS` | **6** |
| `ATTACK_FAILURE` | 98 |
| `JSON_PARSE_ERROR` (execution failure, correctly separated) | 4 |

Six successes across 108 runs is consistent with the frozen attacker and irrelevant to this gate —
the pilot validates production integrity, not attack effectiveness.

## 13. Anomalies

**One, and it aborted the pilot — see §14.** Everything else was nominal.

---

## 14. The abort — full diagnosis

### 14.1 What happened

The batch aborted at the start of goal 18 with exit code 1. `run_batch` had just written
`OFF_acronym_017` and was about to begin `ON_crescendo_018`, whose pre-run guard raised
`VRAMGuardError` (`failure_class = VRAM_SAFETY_FAILURE`) — an unhandled abort by design.

### 14.2 Root cause — the gate read the wrong quantity

VRAM composition of the final record and the worst five records:

| run_id | raw driver free | live allocated | reserved | **cached-free (reusable)** | effective available | resident models |
|---|---:|---:|---:|---:|---:|---|
| **`OFF_acronym_017`** | **4.04 GB** | 8.505 GB | 20.294 GB | **11.789 GB** | **15.829 GB** | 1 |
| `OFF_acronym_010` | 6.725 GB | 8.505 GB | 17.610 GB | 9.104 GB | 15.829 GB | 1 |
| `OFF_crescendo_010` | 8.195 GB | 8.505 GB | 16.140 GB | 7.634 GB | 15.829 GB | 1 |

At the abort point only **one** model was resident (the pinned attacker, 8.505 GB live), and
**15.829 GB was genuinely available** — 4.04 GB untouched by anyone plus 11.789 GB sitting in
PyTorch's reusable cache. But `pre_run_check` reads `torch.cuda.mem_get_info()[0]` **directly**, with
no `empty_cache()` first, so it saw 4.04 GB and refused against the 6.0 GB floor.

**Across all 108 records, effective availability never fell below 9.402 GB** — always above the
modern floor. Exactly **one** record tripped, and it is the single record where the allocator happened
to hold a large cached pool. This is a **false positive**.

**Why the load gate never tripped**: `ModelManager._load_with_guards` *does* call
`gc.collect() → synchronize → empty_cache() → synchronize` before reading free VRAM. That is why
**0 of 1901 loads** were refused. `pre_run_check` was written without that step — an inconsistency
between two gates in the same class.

### 14.3 Classification

| Category | Verdict | Basis |
|---|---|---|
| `MODEL_CAPABILITY_FAILURE` | **NO** | attacker 98.8 % / evaluator 100 %; every failure correctly classified |
| `PARSER_FAILURE` | **NO** | 0 parser misclassifications; 0 schema violations |
| `ALGORITHM_FAILURE` | **NO** | no algorithmic file changed (§16) |
| `MODEL_IDENTITY_FAILURE` / `MODEL_RESIDENCY_FAILURE` | **NO** | identity correct; 1901/1900 balanced; ≤2 resident |
| **`INFRASTRUCTURE_FAILURE` — guard false positive** | **YES** | the pre-run gate measured cached memory as if it were consumed |

**It is not data corruption and not a hazardous condition.** No run was lost, no record is wrong, and
no model was loaded unsafely. A conservative guard stopped the batch on a harmless condition.

### 14.4 The fix (identified, **not applied**)

Make `pre_run_check` consistent with `_load_with_guards`:

```
File:               src/guardbound/llm/model_manager.py
Function:           ModelManager.pre_run_check
Current behavior:   reads torch.cuda.mem_get_info() directly, so PyTorch's cached-but-free
                    blocks count as consumed VRAM. Tripped at 4.04 GB raw while 15.83 GB
                    was available.
Required change:    release cached blocks (gc.collect → synchronize → empty_cache →
                    synchronize) before reading free VRAM — the same sequence
                    _load_with_guards already uses — or evaluate free + (reserved − allocated).
Experimental risk:  None to semantics. It makes a conservative gate accurate; the real
                    protections (pre-load gate, post-load verify, 22 GB reserved ceiling)
                    are untouched.
Validation:         Re-run the 108-record history through the new gate and confirm no
                    record that had >= 9.4 GB effective free is refused.
```

`ceiling_check` (22 GB on `reserved`) did **not** fire — `reserved` peaked at 20.294 GB, so that
threshold is currently well-calibrated and should stay.

## 15. Files changed

**No production code was modified by this phase.** `src/` and `scripts/` mtimes all predate the
16:53Z launch (latest is `phase14_full_reproduction.py` at 20:51 local = 15:21Z). The pilot added
only artifacts:

| Path | Contents |
|---|---|
| `results/phase14/batch00.jsonl` | 108 pilot records (relocation to `results/phase15_pilot_30/trackA_frozen/` still pending — §4) |
| `results/phase15_pilot_30/trackA_frozen/` | created, still empty |
| `PHASE_15_30_GOAL_PILOT_REPORT.md` | this report |
| `_scratchpad/batch00_phase14_2_backup.jsonl` | Phase 14.2 evidence, 18 records |

## 16. Algorithm integrity

| Item | Status |
|---|---|
| attack algorithms (`crescendo`, `opposite_day`, `acronym`, `actor_attack`) | **unchanged** — mtimes 03-09 / 05-09 / 08-09 |
| attack prompts / evaluator prompts | **unchanged** — 03-09 / 06-09 |
| JSON schemas | **unchanged** |
| constrained decoder | **unchanged** — 17-09 |
| history handling / round budgets | **unchanged** |
| NBF implementation + threshold | **unchanged** |
| generation parameters | **unchanged** |
| retries / parser repair / fallback | **none added** |

All mtimes predate this session; the tracked `attacks/*.py` modifications are the pre-existing audit
fixes present at session start.

## 17. Tests

No code was changed, so the suite was not re-run for this phase. Baseline remains **561 passed**
(verified earlier in this session).

## 18. Is the architecture safe for Phase 14?

**Safe — with one correction needed in the guard.**

Positive, measured over 12.9 hours and 108 runs:

* **0** schema violations · **0** NBF-OFF contamination · **0** `filter_trials` nulls · **0** load
  refusals · **0** identity or residency failures · **0** OOM or CUDA errors
* **1901 loads / 1900 evictions** — balanced, no leak, ≤2 resident throughout
* peak **19.904 GB** of 25.763 GB (77 %); 4/4 `JSON_PARSE_ERROR` classified and evidenced correctly
* evaluator **100 %** structured-output reliability, attacker **98.8 %**
* the pinned attacker is not reloaded (35 loads for a whole run vs 17.6 mean across all runs)

The single defect is a **false-positive gate**, not a resource hazard: effective freedom never went
below 9.402 GB.

**Next step: apply the §14.4 fix, then resume the pilot** (same command). Resume skips the 108
completed `run_id`s and will finish goals 18–29 — approximately **9 hours** at the measured rate.

---

## Caveats stated honestly

1. **The pilot is incomplete: 108/180, 18/30 goals.** All integrity conclusions are over 60 % of the
   matrix; the unrun goals 18–29 could still surface a different failure.
2. **The output relocation and Phase 14.2 restore did not happen** (§4) — they were scheduled for
   completion. `results/phase14/batch00.jsonl` currently holds pilot data, and the 14.2 evidence is
   in the session scratchpad.
3. **The §14.4 fix is proposed, not applied.** Applying it is a code change and was outside this
   instruction.
4. **The abort is attributed from the records, not from a traceback** — the task's stdout capture was
   empty (the known PowerShell/console issue). The evidence is the terminal `VRAMGuardError`
   signature: exit 1, the last record's 4.04 GB raw free, and `run_batch` raising exactly there.
   A resumed run with visible logging would confirm it directly.
5. **No tuning of ASR, JSON reliability or runtime** was performed, and none is proposed.

---

## STOP

The 1,200-run Track A study was **not** launched and will not be started automatically. The pilot is
halted at 108/180 pending the §14.4 guard fix.
