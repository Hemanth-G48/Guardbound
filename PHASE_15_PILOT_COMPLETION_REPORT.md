# Phase 15 — 30-Goal Pilot Completion Report (Track A, Frozen)

**Date:** 2026-09-24
**Scope:** infrastructure correction (VRAM guard) → validation → resume of the frozen pilot → 180-run integrity audit
**Command (unchanged from the aborted pilot):**

```
python scripts/phase14_full_reproduction.py --batch 0 --limit 30 --attack all --condition both \
    --config configs/reproduction_phase14_frozen.yaml --structured-output-mode constrained_json
```

**Status:** **PILOT COMPLETE — 180/180. GATE: PASS.** The 1,200-run study was **not** launched.

---

## A. Code change

| | |
|---|---|
| File | `src/guardbound/llm/model_manager.py` |
| Function | `ModelManager.pre_run_check` |
| Other production files touched | **none** |
| Attack / NBF / model / prompt files touched | **none** (verified by mtime scan) |

**The VRAM measurement bug.** `pre_run_check` read `torch.cuda.mem_get_info()[0]`
directly. That is the *raw driver-level free* value, which counts PyTorch's
**reusable cached blocks** as consumed memory, so the per-run floor could trip on a
harmless allocator high-water mark. At the abort (`ON_crescendo_018`, after
`OFF_acronym_017`) it read **4.04 GB** free while **11.789 GB** sat in the reusable
cache — effective availability **15.829 GB** against a 6.0 GB floor.
`_load_with_guards` already released that cache before measuring, which is why
0 of 1901 loads was ever refused. The two gates in the same class disagreed.

**The minimal fix.** `pre_run_check` now performs the same release sequence
`_load_with_guards` already used, *before* measuring:

```python
if torch.cuda.is_available():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

free = _free_vram_gb()
if free is None:
    return
if free < self.min_free_per_run_gb:          # unchanged threshold
    ...raise VRAMGuardError(...)             # unchanged refusal
```

Nothing else moved. Thresholds (pre-load 3.0 GB, post-load 1.5 GB, per-run 6.0 GB,
ceiling 22.0 GB reserved), the pre-load gate, the post-load verification, the
pinned-slot residency policy and the load behaviour are all untouched.

**Timeline note (stated honestly).** The edit is present in the working tree with
mtime `2026-09-24 11:43:07`, i.e. it was applied by the previous session *after* it
wrote the 11:20 interim report (which lists the fix as "identified, not applied").
This session opened with a read-only audit, found the fix already in place, and
confirmed it byte-for-byte against the required sequence. **No further production
change was made, because none was needed.** A repository-wide mtime scan confirms
that only two files changed after the pilot started at 22:23 on 23-09:

```
2026-09-24 11:43:07  src/guardbound/llm/model_manager.py      (the fix)
2026-09-24 11:43:45  tests/test_phase14_2_rotation.py         (test-only: 6 new guard tests)
```

All attack modules (`crescendo_paper.py`, `opposite_day.py`, `acronym.py`,
`actor_attack.py`, `runner.py`), the judge prompts, the NBF checkpoint, the
predictor/dynamics code and the frozen config predate the pilot by days. The
recomputed config hash `61ea203c20e68c51` is identical to all 180 records.

## B. Tests

| Run | Result |
|---|---|
| `pytest tests/ -q` (full suite) | **615 passed, 0 failed** (88.12 s) |
| `pytest tests/test_phase14_2_rotation.py -q` | **32 passed** |
| `-k PreRunGuard -v` (Phase 15 guard tests) | **6 passed, 0 failed** |

The 6 targeted tests cover exactly the required properties: cache released *before*
the measurement (`empty_cache` precedes `measure`), the numeric Phase 15 repro
(raw 4.04 GB + 11.789 GB cached → 15.829 GB → **admitted**), genuine 5.9 GB still
**refused**, 6.1 GB admitted, thresholds unchanged (3.0 / 1.5 / 6.0 / 22.0), and
post-load verification still enforced.

**Real-CUDA verification** (not mocked) reproduced the abort condition on the card:

```
live allocated      =  8.001 GB   (resident pinned-attacker analogue)
reusable cached     = 12.004 GB
driver free (raw)   =  4.348 GB   <- what the old guard read (would REFUSE)
effective available = 16.353 GB

after pre_run_check(): ADMITTED
driver free (post)  = 16.353 GB   -> cache released to driver: +12.004 GB  PASS
live allocated      =  8.001 GB   -> delta +0.000000 GB, tensor intact    PASS
```

Caveat on "before/after": the interim report recorded a 561-passed baseline at
11:20, before this session. I did not re-derive that historical figure, so only
measured current values are reported above.

## C. Pilot completion

```
records completed: 180 / 180
goals completed:    30 / 30
remaining:           0
```

| Half | Records | Window |
|---|---:|---|
| Pre-fix (goals 0–17) | 108 | 23-09 22:23 → 24-09 11:15 (aborted) |
| Post-fix (goals 18–29) | 72 | 24-09 12:11:52 → 21:25:23 (exit code 0) |

The resume skipped all 108 completed `run_id`s (validated by `--dry-run` before
launch: **108 DONE / 72 RUN**, and the 72 were exactly goals 018–029 in the frozen
order). `append_result` only appends, so no completed record could be rewritten —
confirmed by the unchanged SHA256 of the pre-resume 108-record snapshot.

## D. Failure taxonomy

| Class | Count |
|---|---:|
| `SUCCESS` | 11 |
| `ATTACK_FAILURE` | 162 |
| `JSON_PARSE_ERROR` | 6 |
| `MODEL_LOAD_FAILURE` | 1 |
| other | 0 |

**The single `MODEL_LOAD_FAILURE` (`ON_crescendo_024`) — explained, not a guard
condition.** The evaluator's pipeline load raised a **network** timeout:

```
loading evaluator (meta-llama/Llama-3.2-3B-Instruct) failed: ReadTimeout:
[WinError 10060] A connection attempt failed because the connected party did not
properly respond after a period of time...
```

It is a transient HuggingFace-hub HTTP read timeout during a load (1 of 3291 loads
in the whole run), classified by the runner's pre-existing `classify_exception`
through `ModelLoadError.failure_class`. It is **not** a VRAM refusal: that record
shows `load_refusals = 0`, peak VRAM 16.31 GB, and its 23 preceding loads in the
same run all succeeded at 6.4–7.7 GB measured. No retry, no repair, no fallback —
the run was recorded as a failure with full telemetry, as the freeze requires.

The 6 `JSON_PARSE_ERROR` records are legitimate attacker failures: `turns = null`
(never `0`), raw attacker output preserved, `error_type = AttackGenerationError`,
complete 55-field schema. Per the freeze, no retry or repair was added.

## E. Integrity

| Requirement | Result |
|---|---|
| 180 / 180 records accounted for | **PASS** (0 malformed lines) |
| duplicate run IDs | **0** |
| missing run IDs | **0** |
| unexpected run IDs | **0** |
| schema violations | **0** — 55 fields, exactly 1 key-set across all 180 |
| NBF-OFF contamination | **0 / 90** (`nbf_scores == []`, `filtered_queries == filter_count == 0`, `nbf_enabled == False`) |
| OFF `nbf_stats` | identical null skeleton on all 90: `{n:0, filter_count:0, min:null, max:null, mean:null}` |
| `filter_trials` nulls | **0** (`{3: 180}`) |
| model identity violations | **0** (single value for attacker / target / evaluator / embedder across 180) |
| residency violations | **0** — max `resident_count` = 2; attacker pinned in all 180 |
| load refusals | **0 / 3291 loads** |
| loads / evictions | **3291 / 3289** (balanced, no leak) |
| OOM / CUDA failures | **0** |
| peak VRAM | **20.281 GB** of 25.763 GB (79 %) |
| unexpected retries / repairs / fallbacks | **0** (no such fields exist in any record) |
| invariant breaches | **0** |
| seeds | match the frozen derivation (42 + goal×1000 + attack offset + condition offset) |

Conservative checks that were written, failed on the data, and were then
**corrected as over-strict rather than explained away**: the first version of the
OFF-`nbf_stats` check rejected any non-empty stats dict; inspection showed the dict
is the canonical skeleton with `null` summaries and counted `n = 0` — i.e. exactly
the required `null = unavailable / 0 = observed zero` semantics, identical across
all 90 OFF records including the 54 from before the fix.

## F. Runtime

| Metric | Post-fix (72 runs) | All 180 |
|---|---:|---:|
| wall-clock | **9 h 13 min 31 s** | 22.1 h (12.9 h + 9.23 h) |
| model runtime | 9.216 h | 22.078 h |
| rotation load time | 3.826 h / 1390 loads | 8.759 h / 3291 loads |
| mean run time | 460.8 s | 441.6 s |
| median run time | 410.5 s | 388.9 s |
| maximum run time | 1035.6 s | 1035.6 s |
| minimum run time | 14.0 s | 14.0 s |

The pilot's own earlier estimate of ≈21.5 h for 180 runs was met: **22.1 h**.
`worktree_sha256` differs between the halves by design — `53119886489e9c5a` on the
108 pre-fix records, `55b9a2a1fbe8fea6` on the 72 post-fix records — because the
hash covers `git diff HEAD`, which now includes the guard fix. Every other identity
field (`code_commit`, `config_hash`, `checkpoint_sha256`, `dataset_sha256`, model
ids, dtype, quantization, `max_rounds`, `threshold`, `top_p`, `filter_trials`) is
identical across all 180.

## G. Gate decision

```
PHASE 15 30-GOAL PILOT: PASS
READY FOR 200-GOAL / 1200-RUN TRACK A REVIEW
```

All integrity requirements pass: 180/180 accounted for, 0 duplicates, 0 missing,
0 schema violations, 0 NBF-OFF contamination, 0 `filter_trials` nulls, 0 model
identity violations, 0 residency violations, 0 unexplained infrastructure
failures, 0 OOM/CUDA errors, 0 unintended retries/repairs/fallbacks.

**Not blocked.** The one non-core failure class is explained (transient hub network
timeout during a model load) and satisfied every telemetry requirement.

**The 1,200-run study was NOT launched.** Per the instruction, the pilot stops here
for review. No thresholds, quantization, offloading or retry behaviour was changed
at any point, and no ASR ranking or tuning was performed.

## Evidence locations

| Path | Contents |
|---|---|
| `results/phase14/batch00.jsonl` | Live runner output, 180 records (SHA256 `FCB952F1598B697B033D605B0791CFBB…`) |
| `results/phase15_pilot_30/trackA_frozen/` | Final 180-record copy (same SHA256), pre-resume 108-record snapshot, interim report, evidence manifest |
| `results/phase14_2_evidence/` | Preserved Phase 14.2 scratchpad + `batch00_phase14_2_backup.jsonl` / `_preserved.jsonl` (recovered from the prior session's temp scratchpad before it aged out) |

No existing Phase 14 or Phase 14.2 evidence was overwritten or deleted.
