# Phase 16.5 — F1/F3 Remediation + Attacker JSON Failure Forensics

**Status:** diagnosis and controlled methodology correction complete.
**No ASR was computed. No 90-run pilot was launched. No 1,200-run study was launched.**
Frozen Track A evidence was read only; nothing historical was overwritten.

Baseline captured *before* any change: `results/phase16_5_forensics/baseline_manifest.json`
(commit `b46d6fd7ed37b7a6ee9e643476ae2b2263b85624`, worktree `47015ef18ab01831`,
frozen config hash `61ea203c20e68c51`, dataset `ac789de8859e755c…`, NBF checkpoint
`cea1a75bcef4fc51…`).

---

## A. F1 resolution

```text
Old target max_new_tokens : 256                 (configs/reproduction_phase14_frozen.yaml)
New target max_new_tokens : NOT EXPLICITLY SET  (null, configs/reproduction_phase16_5_forensics.yaml)
```

```text
No arbitrary replacement fixed token limit : YES   (512/1024/2048/4096 deliberately NOT used)
Natural EOS/stopping enabled              : YES   (nothing but the context window bounds it)
```

**Other implicit generation-length limits (full backend audit, §4).**
The complete path was traced — target → tokenizer → chat template → gen kwargs →
`generate()` → stopping criteria → EOS → decode:

| limit site | value found | imposes a cap? |
| --- | --- | --- |
| `target max_new_tokens` | was 256, now unset | was the *only* cap |
| `target max_length` | not passed | no |
| `generation_config.max_new_tokens` | `None` (Phi-4-mini) | no |
| `generation_config.max_length` | `None` (Phi-4-mini) | no |
| EOS token | `[200020, 199999]` | termination signal, present |
| custom stopping criteria | none on the target-text path (the frozen `}` criterion fires only when `json_format=True`, i.e. attacker/evaluator) | no |
| context window | `config.max_position_embeddings = 131072` | **yes — the only remaining bound** |
| processor/tokenizer limit | `tokenizer.model_max_length = 131072` | the same bound |
| pipeline limit | the pipeline's `GenerationConfig` is built by the loader and carries only the audited values | no |

No hidden ceiling was found: no model's `generation_config.json` sets `max_length`
or `max_new_tokens`, so removing the loader's 256 exposes nothing else.

**Token distribution (target-generation-only probe — no attack run, no ASR).**
8 prompts rebuilt from the frozen records (same system prompt, same history, same
query) and regenerated with the cap removed:

```text
recorded (capped)  : mean 254  median 256  max 256
regenerated (uncapped): mean 1191  median 1300  min 450  max 1310
```

```text
Cap-hit rate before : 95.0 %   (929/929 responses, Phase 16.4)
Cap-hit rate after  :  n/a — no cap exists; 7/8 stopped only by the probe's 45 s bound
Natural-EOS rate    : 12.5 %   (1/8 — at 450 tokens)
Mid-word rate       : 87.5 %   (7/8 — still mid-sentence when the probe's bound hit)
VRAM impact         : 7.98 GB peak (target alone), 11.3 s load; no OOM, no CUDA error
Latency impact      : ~29 tok/s → 45 s ≈ 1,300 tokens → a context-bound generation
                      (131,072 tok) would need ≈ 75 min per turn
```

**F1 = BLOCKED.**

**Why.** Removing the artificial 256 cap proved that the cap *was* cutting off real
content (regenerated replies ran ~5× longer). But it also proved the local target
**does not terminate naturally on this input distribution**: in 7 of 8 cases it was
still generating — mid-sentence — when the probe's wall-clock bound stopped it, and
it never emitted EOS. The only remaining bound is the model's own context window
(131,072 tokens ≈ 75 min/turn), which is not executable for 929 turns and would
silently convert "no cap" into "no practical stopping rule at all".

Per §3 this is therefore **stopped and documented rather than papered over with a new
arbitrary ceiling**. The configuration keeps the honest `null` declaration and carries
an explicit do-not-run warning. The constraint's origin is the substituted local
target itself, not the loader: this differs from the paper, whose GPT-4o target ends
its turns on its own under a server-side maximum that it reaches only rarely.

---

## B. F3 resolution

Verified by driving `generate()` through a recording pipeline
(`results/phase16_5_forensics/f3_sampling_validation.json`):

| | declared (config) | frozen config: actually passed | Phase 16.5 config: actually passed |
| --- | --- | --- | --- |
| **temperature** | 0.7 / 0.7 / 0.0 | 0.7 / 0.7 / 0.0 | 0.7 / 0.7 / 0.0 |
| **top_p** | 1.0 | **not passed — `inherited_from_generation_config`** | **1.0 passed explicitly** |
| **top_k** | 0 (top-k disabled) | **not passed — `inherited_from_generation_config`** | **0 passed explicitly** |
| **do_sample** | true | true (from the temperature heuristic) | true (declared) |

```text
Declared temperature: 0.7 (attacker/target), 0.0 (evaluator)
Effective temperature: 0.7 / 0.7 / 0.0            — matches declaration

Declared top_p: 1.0
Effective top_p: frozen runs 0.8 (Qwen3-4B attacker), 0.9 (Llama-3.2-3B evaluator/target inherited nothing)
                 Phase 16.5 config 1.0            — now matches the declaration

Declared top_k: 0 (disabled)
Effective top_k: frozen runs 20 (Qwen3-4B attacker), None (others)
                 Phase 16.5 config 0             — now matches the declaration

Declared do_sample: true
Effective do_sample: true (all roles, both configurations)
```

**F3 = FIXED.** The declared values are now what is passed, and — independently of any
config — the runtime metadata *reports* inherited parameters as inherited instead of
hardcoding `top_p: 1.0`, so the mismatch can never again be invisible. Note honestly:
the 180 frozen Track A records were produced under `top_p=0.8/0.9`; their recorded
`top_p: 1.0` field was inaccurate. Their bytes are untouched.

---

## C. JSON failure forensics

Existing raw attacker outputs only — nothing regenerated for this stage.

### Frozen Track A (Qwen3-4B attacker) — 1,978 generations, 100 % captured

| attack | total | valid JSON | failed | failure rate | truncation | trailing comma | extra text | wrong key | missing field | other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crescendo_paper | 854 | 848 | 6 | 0.70 % | 4 | 0 | 0 | 0 | 0 | 2 (`MALFORMED_SYNTAX`) |
| opposite_day | 566 | 565 | 1 | 0.18 % | 1 | 0 | 0 | 0 | 0 | 0 |
| acronym | 558 | 547 | 11 | 1.97 % | 11 | 0 | 0 | 0 | 0 | 0 |
| **all** | **1,978** | **1,960** | **18** | **0.91 %** | **16** | 0 | 0 | 0 | 0 | **2** |

The frozen parser rejected nothing else: no fenced JSON, no trailing commas, no
extra prose, no wrong keys — i.e. none of the "structured-output incapacity" shapes.

### Ornith attacker (Phase 16.1 four configurations, and 16.2 config B)

| source | n | valid | failed | rate | dominant classes |
| --- | --- | --- | --- | --- | --- |
| 16.1 (configs A–D pooled) | 137 | 70 | 67 | 48.9 % | `NATURAL_LANGUAGE` 27, `TRAILING_COMMA` 18, `TRUNCATED_JSON` 10, `MALFORMED_SYNTAX` 8, `REASONING_ONLY` 2 |
| 16.2 (config B) | 39 | 32 | 7 | 17.9 % | `TRAILING_COMMA` 5, `TRUNCATED_JSON` 1, `NATURAL_LANGUAGE` 1 |

**The two models fail differently:** the frozen attacker fails almost entirely by
*truncation at its token ceiling*; Ornith fails by *syntax* (trailing commas, prose).

### Concentration (§11) — DISTRIBUTED, not concentrated

18 failures across **15 distinct goals of 30**, at most **2 on any goal**, in **18 of
180 runs**. Failures are not tied to particular goals, turns or attack prompts; they
are stochastic draws against a length boundary. `NBF condition` is not a factor: the
failures split across the ON/OFF conditions in proportion to their run counts.

---

## D. Failure attribution

| failure class | attribution | evidence | confidence |
| --- | --- | --- | --- |
| `TRUNCATED_JSON` (16/18 of frozen failures) | **TOKEN BUDGET** (attacker's own 256 cap) | median 255 output tokens, 16/16 at ≥250 tokens, all ending mid-word (`s`, `r`, `y`); valid outputs median 156 tokens with only 3/1,960 above 250 | **high** |
| `MALFORMED_SYNTAX` (2/18) | **MODEL** | median 195 tokens, both end with `}` — complete but syntactically invalid; not truncation | medium (n=2) |
| attack-specific skew: acronym 1.97 % vs opposite_day 0.18 % | **ATTACK-SPECIFIC INTERFACE — but via length, not wording** | every acronym failure is a truncation, and acronym's required step output is the longest; the JSON contract and `json_format=True` call are identical across attacks | medium-high |
| Ornith `NATURAL_LANGUAGE` / `REASONING_ONLY` | **MODEL** | whole replies in prose, no JSON attempted; absent from the frozen attacker | medium |
| Ornith `TRAILING_COMMA` (the largest Ornith class) | **MODEL / GENERATION INTERFACE** | persists *under constrained decoding* (see §E), so it is not merely a prompt problem | medium |
| parser/interface asymmetry (Crescendo raised, OppositeDay/Acronym absorbed) | **PARSER/INTERFACE** | direct code inspection; fixed as F2 | high |
| `prompt` as a cause | **not supported** | the JSON contract is identical across all three attacks; the only cross-attack difference tracks output *length* | — |

No causal claim is made beyond the evidence: truncation is demonstrated for 16 of 18
frozen failures (they sit exactly at the cap), and the Ornith syntax classes are
demonstrated to survive constrained decoding, but the sample for the other classes is
small.

---

## E. Structured-output diagnostic (§16/§17)

Paired, attack-step-only, Ornith 15, identical prompts/goals/seeds/temperature/budget/
revision; the only variable is the decoding mode. 21 generations per condition
(7 per attack). Constrained mechanism = the project's existing Phase 14.1
`JSONGrammarLogitsProcessor` + `}` stopping criterion — the backend's own native
mechanism, no ad-hoc grammar engine, no repair, no retry.
(`results/phase16_5_forensics/structured_output_comparison.json`)

| | A: unconstrained | B: native constrained |
| --- | --- | --- |
| valid JSON | **15/21 (71.4 %)** | **8/21 (38.1 %)** |
| failure rate | 28.6 % | 61.9 % |
| failure types | `TRAILING_COMMA` 6 | `TRAILING_COMMA` 10, `MALFORMED_SYNTAX` 1, `MISSING_REQUIRED_FIELD` 1, `TRUNCATED_JSON` 1 |
| by attack | crescendo 6/7, opposite_day 5/7, acronym 4/7 | crescendo 4/7, **opposite_day 0/7**, acronym 4/7 |
| mean latency | 5.28 s | 5.62 s |
| peak VRAM | 19.51 GB | 19.51 GB |

**Native constrained decoding did not eliminate the failures — it made them worse**,
and the same direction was already observed independently in Phase 16.1 (constrained
15/20 vs unconstrained 17/20 on Crescendo). The constrained path still emits trailing
commas, so the mechanism is **SUPPORTED_BUT_INSUFFICIENT**: it is available and runs,
but it does not enforce the syntax this interface needs, and it is not a remedy for
the local attacker's structured-output behaviour. No repair layer was added to rescue it.

Interpretation per §18: **Case E** (constrained generation also fails) combined with
**Case A** for the frozen attacker (truncation dominates there).

---

## F. F2 resolution

```text
before: OppositeDay.generate_opposite_day_step  → malformed reply → ("", "")
        Acronym.generate_acronym_step           → malformed reply → ("", "")
        Crescendo.generate_crescendo_step       → malformed reply → AttackGenerationError
        => the same defect was classified three different ways; two of them
           booked it as an ordinary attack outcome.

after:  all three raise AttackGenerationError → classified JSON_PARSE_ERROR.
        A parsed dict keeps `.get(..., "")`, so a *valid* empty query stays a
        legitimate empty query and the two cases are not conflated.
        Crescendo's stricter both-fields-required contract is unchanged.

tests:  tests/test_phase16_5_forensics.py — 34 tests:
        malformed → JSON_PARSE_ERROR for all 3 attacks; exactly one generation
        attempt (no retry); raw reply retained in the error for telemetry;
        valid JSON → normal flow; valid empty query → not a parser failure;
        lenient `.get` preserved; Crescendo stays strict; plus F1 and F3 tests.
```

**Semantic effect on the frozen evidence** (measured, not assumed): of the 18 affected
runs, 6 Crescendo runs were already correctly recorded as `JSON_PARSE_ERROR`. The other
**12 (OppositeDay/Acronym) were silently absorbed**: the empty query ended the loop
early and each was recorded as an ordinary `ATTACK_FAILURE`
(`attack_loop_exhausted` 11, `refusal_retry_limit` 1) with **mean 2.7 turns vs the
cohort's 5.2**. Under the corrected classification they are execution failures, not
attack outcomes. None of them had succeeded, so the *pilot ASR is unaffected* — what
was wrong was the failure accounting, on 6.7 % of Track A runs. Their bytes are
untouched; only future runs are classified correctly.

---

## G. Regression status

```text
focused F1 tests        : PASS   (4 tests)
focused F2 tests        : PASS   (18 tests)
focused F3 tests        : PASS   (3 tests)
attack parity tests     : PASS
NBF parity tests        : PASS
prompt parity tests     : PASS
model backend tests     : PASS
telemetry tests         : PASS
full test suite         : 728 passed, 0 failed   (694 pre-existing + 34 new)
```

One pre-existing test required a documented update —
`tests/parity/test_refusal_threshold_parity.py::test_paper_runner_backtracking_pops_refused_exchange`.
It relied on the old absorbing behaviour: its mock returned prose once its scripted
replies ran out, OppositeDay swallowed it into an empty query, and the loop ended as a
side effect. It now ends the loop with a *valid* JSON reply carrying an intentionally
empty query — the legitimate termination path — so the test's actual subject
(refusal-backtracking popping the refused exchange) is unchanged and still asserted.
No unrelated failure was fixed or concealed; the full suite is green.

---

## H. Track A integrity

```text
results/phase15_pilot_30/trackA_frozen/batch00_goals000-029_180records.jsonl
    SHA256 FCB952F1598B697B033D605B0791CFBB64A010F8E43CD1ADF3D0EDC359D8A092   UNCHANGED
results/phase14/batch00.jsonl
    SHA256 FCB952F1598B697B033D605B0791CFBB64A010F8E43CD1ADF3D0EDC359D8A092   UNCHANGED
results/phase15_pilot_30/trackA_frozen/batch00_goals000-017_108records.jsonl
    SHA256 C12A562A9454D83B6CC0EE93B2653F4AA5AB058C5AC3E50CEED4EF7A8EC57FB3   UNCHANGED
configs/reproduction_phase14_frozen.yaml
    config_hash 61ea203c20e68c51                                             UNCHANGED
record counts: 180 records / 1,978 attacker generations, 100 % captured       UNCHANGED
```

`results/phase16_1_ornith_diagnostic/`, `results/phase16_2_ornith_config_b/` and
`results/phase16_3_ornith_final_qualification/` were read only; every Phase 16.5
artefact was written to the new `results/phase16_5_forensics/` directory.

---

## Deliverables

```text
results/phase16_5_forensics/baseline_manifest.json            pre-change baseline
results/phase16_5_forensics/failure_forensics.jsonl           2,154 classified attacker generations (raw retained)
results/phase16_5_forensics/failure_summary.json              per-source/per-attack distribution + concentration
results/phase16_5_forensics/structured_output_comparison.json §16 paired diagnostic
results/phase16_5_forensics/f1_natural_stopping_probe.json    F1 target-generation evidence
results/phase16_5_forensics/f3_sampling_validation.json       F3 effective-parameter proof
results/phase16_5_forensics/code_changes.md                   F1/F2/F3 changes
results/phase16_5_forensics/phase16_5_report.md               this report
configs/reproduction_phase16_5_forensics.yaml                 separate forensic config (Track A untouched)
scripts/phase16_5_{failure_forensics,f1_natural_stopping_probe,f3_validation,structured_output}.py
tests/test_phase16_5_forensics.py                             34 tests
```

---

```text
PHASE 16.5 DECISION

F1 target truncation:
BLOCKED
  The artificial 256 cap is removed and no replacement ceiling was substituted, but
  the local target does not emit EOS on this input distribution (1/8 natural stops;
  median 1,300 tokens still generating when the wall-clock probe stopped it). The
  only remaining bound is the 131,072-token context window (~75 min/turn), so
  reference-equivalent target generation cannot be claimed for this backend.

F3 sampling mismatch:
FIXED
  Declared top_p/top_k/do_sample are now passed explicitly, and the metadata reports
  inherited values as inherited instead of asserting 1.0. Historical frozen records
  were produced under the models' own top_p (0.8/0.9); their bytes are untouched.

F2 failure classification:
FIXED
  Malformed attacker output now raises AttackGenerationError -> JSON_PARSE_ERROR for
  Crescendo, OppositeDay and Acronym alike; valid empty queries remain ordinary empty
  queries; no retry, no repair, no regeneration.

Primary JSON failure mechanism:
  Truncation at the attacker's own token ceiling. 16 of the 18 malformed frozen
  outputs (89 %) terminate at 250-256 tokens, mid-word — the JSON was still being
  written. Distributed across 15 goals of 30 and 18 of 180 runs, so it is a boundary
  drawn stochastically, not a specific goal, prompt or attack defect. The two
  remaining failures are complete-but-invalid syntax.

Primary suspected cause:
TOKEN BUDGET for the frozen attacker (high confidence);
MODEL / GENERATION INTERFACE for Ornith's syntax failures (medium confidence).
Not prompt wording, not the JSON contract, and not the attack algorithms.

Native constrained generation:
SUPPORTED_BUT_INSUFFICIENT
  Native constrained decoding is available and runs, but the valid-JSON rate fell
  from 71.4 % to 38.1 % on the same paired prompts — a direction replicated from
  Phase 16.1. It emits trailing commas under the grammar and is not a remedy.

Implementation fidelity after changes:
CONDITIONAL
  F3 and F2 are corrected with no methodology change. F1 could not be corrected for
  this backend without inventing a ceiling, so the paper's target-generation behaviour
  is not reproduced; that gap must be declared in any future result.

Track A integrity:
PASS
  180-record hash FCB952F1598B697B…, 108-record snapshot C12A562A9454D83B…, frozen
  config 61ea203c20e68c51 — all unchanged. No historical result overwritten.

ASR experiment:
NOT RUN

90-run pilot:
NOT RUN

NEXT STEP:
  Decide the target's generation bound explicitly before any further run: the measured
  evidence (target does not self-terminate; 89 % of attacker failures are its own
  256-token truncation) means the single highest-value controlled change is to give
  BOTH roles a declared, paper-documentable output bound — the reference GPT-4o
  target is subject to a server-side maximum output too — and to re-run the F1
  validation against that declared bound rather than against "unbounded". That
  decision is the operator's; Phase 16.5 does not make it.
```
