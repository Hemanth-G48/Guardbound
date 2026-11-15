# Phase 16.3 — Ornith-1.5-9B Final Attacker Qualification Gate

**Date:** 2026-09-28
**Type:** read-only qualification review. **No generation experiment was run, no ASR
was computed, and the 90-run pilot was not launched.**
**Evidence:** `results/phase16_3_ornith_final_qualification/` (this report +
`summary.json`); source evidence in `results/phase16_2_ornith_config_b/`,
`results/phase16_1_ornith_diagnostic/`, `results/phase15_pilot_30/trackA_frozen/`.

---

## 1. Executive conclusion

Configuration B of Ornith-1.5-9B is **technically sound and exactly reproducible**,
and its output failures are **already representable by the frozen pipeline** with
no retry, repair, fallback, prompt change or denominator change. It is **not**
qualified without qualification, because its measured per-generation rejection
rate (15 % isolated, 21 % multi-turn) is a **recurring** pattern that does not meet
the project's documented PASS criterion for attacker JSON behaviour, and at the
project's measured call counts it would end the large majority of runs on an
attacker JSON failure.

```text
Attacker qualification: CONDITIONALLY QUALIFIED
90-run ASR pilot:       NOT AUTHORIZED
```

## 2. Phase 16.2 evidence verification (programmatic)

Twenty checks over the raw evidence files, all PASS:

| Check | Result |
|---|---|
| Stage A has exactly 20 records (planned count ⇒ no retries) | PASS |
| Stage A valid JSON = 17 / rejected = 3 | PASS |
| Stage A failure classes = `{VALID_JSON: 17, EXTRA_TEXT: 3}` (matches the report) | PASS |
| Stage B attempted = 19, successful generations = 15 | PASS |
| Stage B context failures = 4 | PASS |
| Stage B complete 3-turn sequences = 4 | PASS |
| No duplicate `(seq, turn)` pair ⇒ no regeneration | PASS |
| Every record preserved a raw output field | PASS |
| All 32 accepted replies parsed by **direct `json.loads`** | PASS |
| No accepted reply needed the structural extraction (0/32) | PASS |
| run-1 vs run-2 raw text identical (Stage A 20/20, Stage B 19/19) | PASS |
| Turn role ordering correct on every generated turn (15/15) | PASS |
| Goal preserved on every generated turn | PASS |
| 16.2 harness contains no prompt text (no non-docstring literal > 200 chars) | PASS |
| 16.2 harness calls the production attack entry points | PASS |
| Track A 180-record file hash `FCB952F1598B697B033D605B0791CFBB…` | PASS |
| Frozen config hash `27B660A7052054521A94E3430013D986…` | PASS |

No retry, no parser repair, no fallback, no prompt modification, no attack
modification and no NBF modification is evidenced anywhere in the 16.2 artifacts:
the record counts are exactly the planned counts, the accepted replies never
required extraction, and the harness calls the frozen contract functions.

**Determinism claim verified.** Phase 16.2's two independent runs produced
byte-identical raw output text (20/20 Stage A, 19/19 Stage B) and Phase 16.1's
Configuration B is byte-identical to both on all 39 generations. The claim rests
on byte-level comparison of the preserved raw outputs, not on summary figures.

## 3. Reproducibility assessment

```text
REPRODUCIBILITY = PASS
```

* 16.1 → 16.2: Stage A 17/20, Stage B 15/19, 4 context failures, 4/8 full
  sequences — all identical; raw text byte-identical on all 39 generations.
* 16.2 run 1 → run 2: raw text byte-identical on all 39 generations.
* The failures are reproducible in the strongest sense: **the same goals fail**.

Caveat carried from 16.2 and repeated here: because seeds and every generation
parameter are held identical by design (the task requires paired comparability),
this demonstrates determinism of the model + harness path and confirms the earlier
measurement. It is not an independent statistical replication and adds no new
sampling evidence.

## 4. Interface reliability assessment

### 4.1 Is there an existing reliability gate?

**Yes — a documented, qualitative gate exists**; a numeric threshold does **not**.

| Source | Criterion |
|---|---|
| `results/PHASE14_7_18_RUN_PRODUCTION_VALIDATION.md` §17 "Decision Gate" | PASS criterion: **"No *systematic* JSON/parser issue — 1.03 % per-call on the attacker, isolated to 1 run; not a recurring pattern (Phase 14.2 was 64.7 % per-call failure)"** |
| ibid. §17 | The decision vocabulary is `PASS` / `CONDITIONAL` / `BLOCKED`; CONDITIONAL exists for "documented limitations [that] warrant explicit consideration before committing 1,200 runs" |
| `results/PHASE14_3_MODEL_ROLE_SWAP_QUALIFICATION.md` | The role-swap qualification methodology: call the **production contract functions unchanged**, use a pass-through proxy to separate model failure from parser failure, report contract success rate. Strictness: "JSON is accepted only if the backend returned a `dict` or a string that `json.loads` parses to a dict. No fenced-JSON recovery, no prose stripping, no partial-object extraction." |
| `PHASE_15_PILOT_COMPLETION_REPORT.md` (gate) | "Classified `JSON_PARSE_ERROR` records are acceptable as experimental failures if their telemetry is complete and the failure is legitimate." |

```text
Existing reliability gate: FOUND (qualitative)
Numeric acceptance threshold: NOT DEFINED
```

No numeric attacker-reliability threshold exists anywhere in the project's
documentation, configuration or tests, and **none was invented here**. The `≥ 0.5`
figure that appears in `scripts/qualify_glm_attacker.py` is a harness constant
introduced in Phase 15 for that phase's staged gating; it is not a project gate
and is not used for this decision.

### 4.2 Applying the existing criterion to Ornith Configuration B

| Attacker | per-call JSON failure | recurring pattern? | Phase 14.7 criterion |
|---|---:|---|---|
| Qwen3.5-4B (Phase 14.2) | **64.7 %** | yes | not acceptable — model swapped out |
| Qwen3-4B-Instruct-2507 (frozen Track A, measured over the 180 pilot records) | **0.91 %** | no (6 runs of 180) | PASS |
| **Ornith-1.5-9B Config B** | **15 % (isolated) / 21 % (multi-turn)** | **yes — same goals fail every run** | **PASS criterion NOT met** |

Ornith Configuration B sits between the accepted reference and the rate that
triggered a model swap: ~16–23× the frozen attacker's per-call rate, ~3–4× below
Qwen3.5-4B's. Because the failures recur on the same goals rather than being
isolated to a single run, the gate's PASS criterion ("not a recurring pattern")
is not satisfied; the case falls into the gate's CONDITIONAL territory, which
requires the cost to be acknowledged rather than tuned away.

## 5. Existing failure-handling behaviour (verified in code, unchanged)

| Question | Answer (frozen behaviour) |
|---|---|
| Does a JSON failure terminate the current attack? | **Yes.** `AttackGenerationError` propagates out of the attack; the run ends immediately. |
| Is the raw output retained? | **Yes.** `attacker_raw_outputs` is captured by the telemetry proxy on both paths, and the raw text is embedded in `error_message`. |
| Is the run still counted in the denominator? | **Yes.** The failure record is appended with the same 56-field schema and the same `run_id`; nothing is excluded. |
| Any retry? | **No.** One generation per turn; no re-generation on parse failure. |
| Any parser repair? | **No.** `json.loads` → the pre-existing structural extraction → raw string; the parser is unchanged. |
| Any fallback? | **No.** No alternate model, prompt or decoding mode. |
| Can a failed generation alter later turns? | **No.** The run terminates; there is no later turn to contaminate. |
| Does the success criterion change? | **No.** `evaluation.success_score = 5` and `AttackGenerationError → JSON_PARSE_ERROR` are untouched. |

So Ornith's failures are representable **transparently and without any
methodological exception** — they are the pipeline's own first-class outcome.

## 6. Expected measurement / coverage implications

**The planned matrix, using the project's actual inclusion rules.** The frozen
study excludes two attacks (`EXCLUDED_ATTACKS`: `actor_attack` →
`LOCAL_MODEL_CAPABILITY_LIMIT`, `red_queen` → `OFFICIAL_IMPLEMENTATION_NOT_PRESENT`),
so the matrix is **200 goals × 3 attacks × 2 NBF conditions = 1,200 runs**, not the
1,600 theoretical cells.

**Measured attacker call counts (frozen Track A records, n=180):** 1,978 attacker
generation calls, mean **10.99 calls/run**, 18 parse failures ⇒ **0.91 % per call**;
6/180 runs (3.3 %) recorded `JSON_PARSE_ERROR`.

Compounding the per-call rate across a run — the same arithmetic the project
itself used in Phase 14.7 §17 — the share of runs surviving their attacker calls is:

| per-call failure rate (source) | survives N=8 | survives N=11 | survives N=15 |
|---|---:|---:|---:|
| 0.91 % — Qwen3-4B, frozen Track A (measured) | 92.9 % | **90.0 %** | 87.2 % |
| **15 % — Ornith Config B, Stage A (3/20)** | 27.2 % | **15.4 %** | 8.7 % |
| **21 % — Ornith Config B, Stage B (4/19)** | 15.1 % | **6.6 %** | 2.9 % |
| 64.7 % — Qwen3.5-4B, Phase 14.2 (reference) | ≈0 % | ≈0 % | ≈0 % |

At the project's mean of ~11 calls per run this implies roughly **85–93 % of the
1,200 runs** would terminate on an attacker JSON failure under Configuration B,
against ~10 % for the frozen Qwen attacker (consistent with the project's own
documented 10–17 % estimate for 1.03 %).

**What this does and does not mean.**

* It is a **coverage** statement, not an effectiveness statement. No ASR is
  computed or predicted here.
* A rejected generation is **not** an attack failure. The frozen taxonomy already
  separates them: these runs would be `JSON_PARSE_ERROR` records, not
  `ATTACK_FAILURE`, and they remain in the denominator exactly as the four Track A
  pilot `JSON_PARSE_ERROR` records did.
* The consequence is a **measurement** consequence: with ~85–93 % of runs ending
  on an execution failure, multi-turn trajectories would rarely reach the 8-round
  budget, and the attack-outcome signal (SUCCESS vs ATTACK_FAILURE) would rest on
  a small surviving minority. Any downstream analysis would have to report those
  runs separately — which the existing taxonomy and reporting already do.
* These are **estimates from a small diagnostic** (20 isolated generations, 19
  multi-turn turns) compounded under an independence assumption the project
  itself flags as unproven (Phase 14.7 §414). They are not measurements at scale.

## 7. Comparison with previously tested attacker backends

Documented evidence only; **no ranking and no "best" claim** — the columns expose
methodological tradeoffs.

| | Qwen3-4B-Instruct-2507 | Qwen3.5-4B | GLM-4.6V-Flash | Ornith-1.5-9B |
|---|---|---|---|---|
| Backend status | existing `hf_local` (frozen Track A) | existing `hf_local` (Phase 12/14.2 stack) | **new** `glm4v` (Phase 15) | **new** `ornith15` (Phase 16) |
| Technical qualification | production attacker since Phase 14.7 | earlier attacker, replaced | PASS (native path verified, 20.586 GB) | PASS (native class verified, 18.82 GB) |
| JSON / interface behaviour | 0.91 % per-call failure over 180 frozen runs; 48/48 contract success in the Phase 14.3 gate | 64.7 % per-call failure (6/17 JSON ok) in Phase 14.2 — the reason for the role swap | Stage A 5/10 (50 %); Stage C 9/9 runs with 3 `JSON_PARSE_ERROR` | Config A 15/20, **B 17/20**, C 7/20, D 0/20; 15–21 % recurring rejection |
| Multi-turn behaviour | mean 10.99 attacker calls/run; 180-run pilot completed | runs ended early (mean 0.17 turns) | Stage B 5/5 sequences, 0 context failures; Stage C 9/9 runs, short trajectories | Stage B 15/19 turns, 4/8 sequences reached turn 3 |
| VRAM behaviour | ~8.045 GB; pinned-slot rotation, 0 refusals, peak 20.281 GB in the pilot | 2-model stack, peak 17.465 GB | 20.586 GB → forces sequential residency (guidance: refuses target load otherwise) | 18.82 GB → sequential residency; 0 refusals, peak 19.963 GB |
| Known blockers | none | abandoned (JSON reliability) | needs `enable_thinking=false` (else 0/8 correct schema) + residency rewrite | recurring malformed JSON (trailing comma, truncation, space-prefixed key) |
| Track A pilot completed? | **YES — 180/180** | no | no (9-run Stage C interface test only) | no |

## 8. Integrity verification

```text
Track A records       FCB952F1598B697B033D605B0791CFBB…   unchanged
Track A snapshot      C12A562A9454D83B6CC0EE93B2653F4A…   unchanged
Frozen config         27B660A7052054521A94E3430013D986…   unchanged
frozen config_hash    61ea203c20e68c51                    unchanged
Attack / NBF code     mtimes predate Phase 16.1            unchanged
16.1 / 16.2 evidence  not overwritten                      unchanged
```

No production, backend, parser, prompt, attack, NBF, target, evaluator or
configuration file was modified in this phase. This phase wrote only
`results/phase16_3_ornith_final_qualification/`.

## 9. Qualification decision

```text
CONDITIONALLY QUALIFIED
```

**Why not QUALIFIED:** the existing gate's PASS criterion for attacker JSON
behaviour is "no *systematic* JSON/parser issue … isolated to 1 run; not a
recurring pattern". Ornith Configuration B's 15 %/21 % rejection is recurrent —
the same goals fail on every run — so it does not meet that criterion, and no
numeric threshold exists that it could be said to satisfy. No threshold was
invented to bridge the gap.

**Why not NOT QUALIFIED:** nothing Ornith requires would change the experiment.
It needs no parser repair, no retry, no regeneration, no fallback, no prompt or
generation-parameter change, no success-criterion or denominator change, and no
special handling of failures — the frozen pipeline already records exactly this
outcome class with complete telemetry.

**Exact condition attached to this qualification:**

1. The frozen failure handling must be preserved verbatim: a rejected generation
   terminates the run as `JSON_PARSE_ERROR`, the raw output is retained, the run
   stays in the denominator, and no retry, repair, fallback or prompt/parameter
   change is introduced.
2. The expected coverage loss — on the order of **85–93 % of ~1,200 runs
   terminating on an attacker JSON failure** at ~11 calls/run, versus ~10 % for
   the frozen attacker — must be **declared and accepted in advance** of any run,
   exactly as Phase 14.7 required for the 1.03 % case ("should be accepted
   knowingly, not discovered mid-study"), and the `JSON_PARSE_ERROR` records must
   be reported separately from attack outcomes.

## 10. Statement regarding the 90-run pilot

No ASR experiment and no 90-run pilot were launched in this phase, and nothing was
tuned or retried. The 1,200-run Track A study remains **not authorized** for an
Ornith attacker: under the decision above it may only proceed after the condition
in §9 is explicitly accepted, and it must never be launched automatically.

---

```text
PHASE 16.3 DECISION

Technical backend:
PASS

Reproducibility:
PASS

Frozen-interface compatibility:
PARTIAL

Existing reliability gate:
FOUND

Methodological exceptions required:
NO

Attacker qualification:
CONDITIONALLY QUALIFIED

90-run ASR pilot:
NOT AUTHORIZED

Reason:
Configuration B is technically sound and exactly reproducible (byte-identical raw
outputs across 16.1, 16.2 run 1 and 16.2 run 2; 17/20 isolated and 15/19
multi-turn generations accepted by the unmodified frozen interface, all via direct
json.loads with no repair or extraction), and its failures are already represented
transparently by the frozen pipeline with no retry, repair, fallback or denominator
change. However, the measured 15-21% per-generation rejection is a *recurring*
pattern on specific goals and does not meet the project's documented PASS criterion
("no systematic JSON/parser issue ... not a recurring pattern", Phase 14.7 §17),
sitting ~16-23x above the frozen attacker's 0.91% per-call rate. At the measured
~11 attacker calls per run that implies ~85-93% of the 1,200 planned runs would end
on an attacker JSON failure. Under the gate's own vocabulary this is the
CONDITIONAL case: it is qualified only if the frozen failure handling is preserved
verbatim and that coverage cost is declared and accepted before any run.
```
