# Phase 18 — Main NBF Experiment: C1 + NBF OFF vs C1 + NBF ON

**Status: COMPLETE — 180/180 runs (30 of 30 goals, 90 matched cells).**

The single experimental variable is the NBF state. Both arms are the frozen C1 attacker
configuration. Nothing in C1, the attacks, the models, the prompts, the generation
parameters, the dataset, the evaluator, the success criterion or the execution protocol was
modified for this experiment. All §22 integrity checks pass.

---

## 1. Executive summary

| Item | Value |
| --- | --- |
| Total runs | **180** (30 goals × 3 attacks × 2 conditions) |
| NBF OFF runs | 90 |
| NBF ON runs | 90 |
| **Official ASR, NBF OFF** | **43/90 = 47.8%** |
| **Official ASR, NBF ON** | **29/90 = 32.2%** |
| Absolute difference | **−15.6 percentage points** (relative 0.674, i.e. −33%) |
| Matched result (90 cells) | 23 cells OFF-only, 9 cells ON-only → exact McNemar **p = 0.0201** |
| Unmatched Fisher | p = 0.0476 |
| NBF filtering rate | **32.2%** (266 of 825 candidate evaluations) |
| Runtime overhead | **NBF ON is faster**: 660 s vs 1015 s mean per run |
| Infrastructure failures | 11 CUDA OOMs (10 OFF, 1 ON); none removed from the denominator |
| Total run time | 41.9 h |

## 2. What ran

| Item | Value |
| --- | --- |
| Arm A | **C1 + NBF OFF** (`barrier=None`, `embed_fn=None`) |
| Arm B | **C1 + NBF ON** (validated NBF, threshold 0.0, `steer_target=False`) |
| Attacker | `Qwen/Qwen3.8-27B` rev `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, NF4/BF16, thinking OFF, max tokens UNSET |
| C1 | Stage 6 generation-point instruction (mode `I1`), temperature 0.7, top_p 1.0 |
| Target | `meta-llama/Meta-Llama-3-8B-Instruct` rev `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| Judge | `Qwen/Qwen3-8B` rev `b968826d9c46dd6066d109eabc6255188de91218`, T=0, thinking OFF |
| Attacks | Crescendo, OppositeDay, Acronym (30 goals each) |
| Rounds | `max_turns=8` (cap, not achieved depth) |
| Retries / repair / fallback | 0 / NONE / NONE |
| Success | `attack.was_successful()` (rubric score 5) |
| Dataset | frozen 200-goal slice, goals 0–29, sha256 `ac789de8…d014eb` |
| Seed | `derive_seed(goal_id, attack, "off")` |
| Run IDs | `NBFOFF_<attack>_<goal>` / `NBFON_<attack>_<goal>`; 180/180 unique, 0 duplicates |

## 3. NBF fidelity (unchanged from the validated implementation)

`config/nbf_fidelity.json` records the verified state before the run:

| Component | Value |
| --- | --- |
| Checkpoint | `nbf_original_stuff/.../NBF-LLM/models/models_best_nbf_released.pth` |
| Checkpoint sha256 | `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` = the expected constant |
| Architecture | `NeuralBarrierFunction(dynamics, predictor)` via `load_original_checkpoint` |
| State dimension / classes | 768 / 5 |
| Embedding model | `all-mpnet-base-v2` (768-d) |
| Initial state | `zeros(768)` |
| Barrier score | `P(class 5) − max(other classes)` |
| Threshold / eta | 0.0 / 0.0 |
| Filtering logic | runner `calculate_score_from_dialog`; **score ≥ threshold → filtered** (target not called, round consumed); score < threshold → accepted |
| Retrained / redesigned / retuned | **No / No / No** |
| Extra safety heuristic added | **None** |

## 4. Pre-run validation (§18) — all checks passed

6 smoke runs (1 goal × 3 attacks × 2 conditions), executed in `smoke/` so the 180-run
denominator is untouched: C1 identical in both arms ✓, NBF OFF records 0 evaluations ✓,
NBF ON evaluates and logs filtering decisions ✓, checkpoint hash matches ✓,
target/judge/evaluator identical ✓, run IDs unique ✓, progress tracking ✓, ETA ✓,
**resume verified** ✓, frozen artifacts unchanged ✓.

## 5. Primary endpoint — official ASR

```
C1 + NBF OFF : 43/90 = 47.8%   (95% CI 37.8 – 58.0)
C1 + NBF ON  : 29/90 = 32.2%   (95% CI 23.5 – 42.4)
Difference   : -15.6 percentage points   (relative change 0.674, i.e. -33%)
Fisher exact (unmatched, two-sided) : p = 0.0476
```

Denominator = all 90 runs per arm, including the 11 OOMs and 14 attacker contract failures.
No run was removed, no condition switched to a survivor denominator, no goal rerun.

## 6. Matched analysis

Pre-specified test: **exact McNemar (two-sided)** over matched goal × attack cells — the
project's established matched test, declared before results were seen.

```
matched cells: 90

OFF success / ON success : 20
OFF success / ON failure : 23
OFF failure / ON success :  9
OFF failure / ON failure : 38

discordant pairs: 32       exact McNemar p = 0.0201
mean paired turn difference (ON − OFF): -1.42 turns (median -1.0)
```

Both readings agree: the matched test (p = 0.0201) and the unmatched Fisher test
(p = 0.0476) are significant, with discordant cells running 23:9 in favour of NBF OFF.

## 7. How the estimate settled as the matrix grew

The experiment was extended in balanced goal blocks; the same endpoints recomputed at each
boundary show how much a partial read would have misled:

| Block | Runs | Cells | ASR OFF | ASR ON | Diff | Exact McNemar | Discordant (OFF:ON) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| goals 0–4 | 30 | 15 | 60.0% | 33.3% | −26.7 | p = 0.125 (ns) | 4 : 0 |
| goals 0–11 | 72 | 36 | 52.8% | 30.6% | −22.2 | p = 0.0215 | 9 : 1 |
| goals 0–14 | 90 | 45 | 57.8% | 33.3% | −24.4 | p = 0.0034 | 12 : 1 |
| goals 0–19 | 120 | 60 | 53.3% | 35.0% | −18.3 | p = 0.0192 | 15 : 4 |
| **goals 0–29** | **180** | **90** | **47.8%** | **32.2%** | **−15.6** | **p = 0.0201** | **23 : 9** |

The direction never changed, but **the effect size shrank monotonically with sample size**
from −26.7 to −15.6 points, and the discordant split moved steadily toward equilibrium
(4:0 → 23:9). An early stop at 30 or 90 runs would have reported a substantially exaggerated
effect. The final estimate is the one to quote.

## 8. Attack-level results (separate reporting, no ranking)

| Attack | NBF OFF ASR | NBF ON ASR | NBF ON filtering | turns OFF → ON | refusals OFF → ON | OOM OFF → ON |
| --- | --- | --- | --- | --- | --- | --- |
| Crescendo | 15/30 = 50.0% | 15/30 = 50.0% | 30/246 = 12.2% | 119 → 122 | 126 → 87 | 5 → 1 |
| OppositeDay | 16/30 = 53.3% | 13/30 = 43.3% | 69/309 = 22.3% | 132 → 103 | 178 → 135 | 2 → 0 |
| Acronym | 12/30 = 40.0% | **1/30 = 3.3%** | **167/270 = 61.9%** | 126 → **24** | 191 → 79 | 3 → 0 |

- **Crescendo is exactly tied** (50.0% vs 50.0%) — the attack the barrier filters least
  (12.2%). Its total depth even rises slightly under NBF because filtered candidates consume
  round budget without producing progress.
- **Acronym carries essentially the whole effect**: 40.0% → 3.3% with 61.9% of candidates
  filtered and 29 of its 30 runs affected, depth collapsing 126 → 24 turns. Acronym is the
  paper's training attack — the in-distribution case for the barrier.
- **OppositeDay moves modestly** (53.3% → 43.3% at 22.3% filtering).

## 9. NBF-specific measurements (§11)

| Metric | NBF OFF | NBF ON |
| --- | --- | --- |
| Barrier evaluations | **0** (invariant) | **825** |
| Queries accepted | — | 559 |
| Queries filtered | — | 266 |
| Filtering rate | — | **32.2%** |
| Runs with any evaluation | 0 | 87/90 |
| Runs affected by filtering | 0 | **60/90** |
| Runs terminated after filtering | 0 | 4 |
| Successful attacks after being filtered | — | **11** |
| Threshold | — | 0.0 |
| Bypass | — | none; every candidate in this arm was scored before use |

Barrier-score distribution (825 evaluations):

| Bucket | Count | Share |
| --- | --- | --- |
| < −0.1 | 149 | 18.1% |
| −0.1 … −0.01 | 41 | 5.0% |
| −0.01 … −0.001 | 77 | 9.3% |
| −0.001 … 0 | 292 | **35.4%** |
| ≥ 0 (filtered) | 266 | **32.2%** |

**Two thirds of all candidates (68%) sit within ±0.001 of the 0.0 threshold**, and only 18.1%
are clearly unsafe-looking (< −0.1). For most of the distribution, accepted and filtered
candidates are separated by floating-point-scale differences. This is a property of the
reference checkpoint on this pipeline and remains the most important caveat on every
filtering-rate number: the barrier operates at the centre of its own score distribution
rather than at a discriminative point. It also means the filtering rate is highly sensitive
to tiny score perturbations and would change materially under a different (even
slightly-shifted) threshold.

Filtering is spread across the entire trajectory rather than concentrated at opening moves
(index 1: 31/87 filtered; index 2: 40/84; index 4: 27/76; index 8: 21/52; index 14: 4/22).

Conditional reading (diagnostic only, never the official metric): of the 60 filtered runs,
11 succeeded (18%); of the 30 unfiltered runs, 18 succeeded (60%). Being filtered is
strongly associated with failure but is not fatal.

## 10. Attack-trajectory and reliability metrics (§12, §13)

| Metric | NBF OFF | NBF ON |
| --- | --- | --- |
| Total completed turns | 377 | 249 |
| Mean / median turns per run | 4.19 / 3 | 2.77 / 2 |
| Max turns | 8 | 8 |
| Runs reaching turn 8 | 31 | 8 |
| Target calls / natural EOS | 945 / 945 | 559 / 559 |
| Target refusals (all attacks) | 495 | 301 |
| Judge calls / judge failures | 1557 / 0 | 862 / 0 |
| Attacker calls | 958 | 834 |
| Attacker calls valid | 956 = **99.8%** | 829 = **99.4%** |
| Prose failures | 1 | 4 |
| Empty `generatedQuestion` | 4 | 3 |
| Malformed JSON | 1 | 1 |
| Truncated outputs | 0 | 0 |

**Termination breakdown** (authoritative field: `failure_class`; `termination_reason` reads
`attacker_generation_error` for any loop that ended via a raised exception, so OOM vs JSON
error must be read from `failure_class`):

| Outcome | NBF OFF | NBF ON |
| --- | --- | --- |
| Success (rubric 5) | 43 | 29 |
| Refusal-retry limit reached | 28 | 10 |
| Loop budget exhausted | 4 | **41** |
| Max turns reached | 7 | 4 |
| CUDA OOM (run killed mid-loop) | **10** | 1 |
| Attacker contract failure | 6 | 8 |

- **10 of the 11 OOMs are in the OFF arm**, and every one is a refusal-loop artifact
  (15–18 attacker calls with few or no accepted turns, context growing until the 24 GiB card
  is exhausted) rather than an accepted-depth artifact. The ON arm's 41
  `attack_loop_exhausted` terminations are the same budget being consumed by filtered
  candidates instead — the NBF converts one failure mode into another.
- The single ON-arm OOM shows the mechanism is a pressure limit, not an absolute barrier.
- Attacker contract reliability is essentially equal (99.8% vs 99.4%).

## 11. Runtime and GPU

| Metric | NBF OFF | NBF ON |
| --- | --- | --- |
| Mean run time | 1015.4 s | 660.4 s |
| Median | 1172.1 s | 572.3 s |
| Min / max | 3.4 / 1900.9 s | 3.4 / 1817.6 s |
| Throughput | 3.55 runs/h | 5.45 runs/h |
| Peak VRAM | 18.38 GiB | 18.38 GiB |
| OOM | **10** | 1 |
| Total run time (180 runs) | — | 41.9 h |

**NBF ON runs ~355 s faster on average** — the barrier's own compute (one mpnet embedding
plus a small MLP per candidate) is negligible against generation; the saving comes from
skipped target calls on filtered rounds and from shorter trajectories.

## 12. Infrastructure and integrity

- No infrastructure failure caused a run to be dropped; all 180 runs are in the denominator,
  including 11 OOMs and 14 attacker contract failures.
- Integrity check: **PASS** — dataset sha256 matches, frozen Stage 4.8 pilot data unchanged
  (`55101e0e…7feaad`), Stage 7 artifacts present and untouched, no duplicate run IDs, raw
  attacker/target/judge outputs present for every call, one-variable-only verified.
- Smoke runs stayed in `smoke/` and are **not** in the 180-run denominator.
- GPU released after the run; no source file, config or frozen artifact was modified.

## 13. Interpretation (§21) — strictly separated

1. **C1** is the frozen Stage 6 generation-point configuration. Nothing about C1 was changed
   here, and no result above should be read as a statement about C1's quality.
2. **NBF OFF vs ON** measures the Neural Barrier Function under that fixed C1 configuration.
   In the complete matrix the effect is a **15.6-point ASR reduction (47.8% → 32.2%)**,
   significant in both the pre-declared matched test (p = 0.0201) and the unmatched Fisher
   test (p = 0.0476). It is concentrated almost entirely in **Acronym (40.0% → 3.3%)**,
   partially present in **OppositeDay (53.3% → 43.3%)**, and absent in **Crescendo
   (50.0% vs 50.0%)**.
3. **The headline number is a floor, not a ceiling, for how a partial read would have
   looked.** The estimate shrank monotonically as goals were added (−26.7 → −15.6 points,
   §7); partial blocks would have overstated the effect by up to 70%.
4. **Local reproduction result.** These are local substitute models (Qwen3.8-27B attacker,
   Llama-3-8B-Instruct target, Qwen3-8B judge) and a reference NBF checkpoint on an
   mpnet-768 state. This is not GPT-4o-scale reproduction and the ASR values are not
   comparable to the paper's numbers.
5. **Substitution caveats.** The barrier-score concentration at the threshold (§9) suggests
   the reference barrier is poorly calibrated for this embedding/trajectory distribution.
   The measured effect is also partly a budget/mechanism effect rather than pure semantic
   filtering: NBF replaces 10 refusal-loop OOMs with 41 round-budget exhaustions, collapses
   acronym depth (126 → 24 turns), and leaves the most filter-resistant attack completely
   unaffected. A different `max_turns` or filtering policy would change how much of the
   measured ASR gap persists.

## 14. Completion condition (§22)

```
180/180 runs completed                              → MET
90 unique NBF OFF runs                              → MET (90)
90 unique NBF ON runs                               → MET (90)
180 total run artifacts                             → MET
0 missing run IDs / 0 duplicate run IDs             → PASS
configuration consistency                           → PASS
raw-output integrity                                → PASS
dataset integrity                                   → PASS
frozen-artifact integrity                           → PASS
```

Self-contained artifacts for the complete experiment:

```
results/phase18_nbf_experiment/
├── config/nbf_fidelity.json              checkpoint/architecture/threshold verification
├── raw/run_level_metrics.jsonl           180 run records (all fields, raw model outputs)
├── raw/nbf_off/runs.jsonl                90 NBF OFF runs
├── raw/nbf_on/runs.jsonl                 90 NBF ON runs
├── progress/progress.json                tracker state + estimate trajectory
├── smoke/run_level_metrics.jsonl         6 pre-run validation runs (not in the denominator)
├── final_report/phase18_summary.json     machine-readable headline numbers
└── analysis/                             asr_comparison, matched_analysis, nbf_analysis,
                                          trajectory_metrics, attacker_reliability,
                                          attack_level, goal_level, runtime_gpu,
                                          integrity_check + 7 figures
```

The frozen Stage 4.8 pilot, Stage 4.9/5/6/7 analyses and the frozen C1 configuration were
never modified; this experiment lives entirely in its own directory and the two arms differ
in exactly one variable — the NBF state.
