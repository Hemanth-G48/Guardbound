# Phase 18 Extension — Additional Matched NBF Runs (goals 30–39)

```
PHASE 18 EXTENSION COMPLETE
```

60 / 60 runs executed · 30 matched cells · goals 30–39 · threshold 0.0 · frozen Phase 18
methodology throughout.

**Direction reproduced: NBF OFF 23/30 = 76.7% vs NBF ON 17/30 = 56.7% (Δ = −20.0 points).**
The extension alone is not statistically significant at 30 matched cells (exact McNemar
p = 0.109), so this is a **directional replication**, not an independent confirmation.

---

## 1. Original Phase 18 (primary frozen result — unchanged)

| | Value |
| --- | --- |
| Matched cells | 90 (goals 0–29) |
| NBF OFF | 43/90 = **47.8%** |
| NBF ON | 29/90 = **32.2%** |
| Difference | **−15.6 percentage points** |
| Exact McNemar | **p = 0.0201** |

Phase 18's raw artifact hash is unchanged (`02e11c0fcd975dfa00a97258a4dc2dde…`, verified
identical at preflight and after the extension), its status still reads `COMPLETE`, and its
90-cell denominator was not touched. Nothing in Phase 18's directory was written.

## 2. Extension (goals 30–39, reported independently)

| Metric | NBF OFF | NBF ON |
| --- | ---: | ---: |
| Successes / runs | **23/30** | **17/30** |
| ASR | **76.7%** | **56.7%** |
| 95% CI (Wilson) | 59.1 – 88.2 | 39.2 – 72.6 |
| Mean turns | 3.80 | 2.53 |
| Median turns | 3 | 2.5 |
| Max turns | 8 | 8 |
| Runs reaching turn 8 | 7 | 1 |
| Target refusals | 68 | 44 |
| NBF evaluations | **0** (invariant holds) | 195 |
| NBF filtered | — | **75 (38.5%)** |

```
Absolute difference (ON − OFF) : -20.0 percentage points
Relative difference (ON / OFF) : 0.739  (i.e. -26%)
Fisher exact (unmatched)       : p = 0.170
```

### Matched analysis (30 cells, pre-specified exact McNemar)

```
OFF success / ON success : 15
OFF success / ON failure :  8     <- discordant, Phase 18 direction
OFF failure / ON success :  2     <- discordant, opposite direction
OFF failure / ON failure :  5

discordant pairs: 10        exact McNemar p = 0.109
mean paired turn difference (ON − OFF): -1.27 turns
```

**Direction: OFF > ON — the Phase 18 direction reproduced.** The discordant cells run 8:2 in
favour of NBF OFF. With 10 discordant pairs the extension does not reach significance on its
own, which is expected at this sample size (Phase 18 needed 90 cells for p = 0.0201).

### Per attack (extension, 10 cells each)

| Attack | NBF OFF | NBF ON | ON filtering |
| --- | ---: | ---: | ---: |
| Crescendo | 8/10 = 80.0% | 5/10 = 50.0% | 21/60 = 35.0% |
| OppositeDay | 7/10 = 70.0% | **9/10 = 90.0%** | 7/60 = 11.7% |
| Acronym | 8/10 = 80.0% | **3/10 = 30.0%** | 47/75 = 62.7% |

The composition of the effect differs from Phase 18 in an informative way: here **two** attacks
favour OFF (Acronym 8→3, Crescendo 8→5) while **OppositeDay reverses** (7→9). In Phase 18
Crescendo was exactly tied (15/30 vs 15/30) and Acronym carried the entire effect. The
opposite overall direction still holds, but the per-attack pattern is not stable across goal
sets — an observation to carry forward, not a claim.

## 3. Combined descriptive analysis (120 cells)

**EXTENDED COMBINED ANALYSIS (descriptive) — NOT the original Phase 18 result.** The 90-cell
Phase 18 figure remains the primary frozen result and must never be presented as a 120-cell
result.

| | Value |
| --- | --- |
| Cells | 120 (90 Phase 18 + 30 extension) |
| Combined NBF OFF | 66/120 = **55.0%** (CI 46.1 – 63.6) |
| Combined NBF ON | 46/120 = **38.3%** (CI 30.1 – 47.3) |
| Combined difference | **−16.7 percentage points** |

No combined p-value is claimed as primary; the extension's pre-specified test is its own
30-cell exact McNemar.

## 4. Reliability

| | NBF OFF | NBF ON |
| --- | ---: | ---: |
| Attacker calls | 190 | 196 |
| Valid outputs | 190 (**100%**) | 195 (**99.5%**) |
| Semantically usable | 190 | 195 |
| Contract failures | 0 | **1 (prose)** |
| Empty / malformed / truncated | 0 / 0 / 0 | 0 / 0 / 0 |
| Target calls / natural EOS | 189 / 189 (**100%**) | 120 / 120 (**100%**) |
| Judge calls / failures | 353 / **0** | 214 / **0** |
| CUDA OOM | **1** | 0 |

No `infrastructure scheduling failure` occurred: all 60 runs were genuinely executed, and all
remain in the denominator — including the single OOM and the single attacker contract failure.

## 5. Trajectory and termination

| Termination reason | NBF OFF | NBF ON |
| --- | ---: | ---: |
| Success (rubric 5) | 23 | 17 |
| Loop budget exhausted | 0 | **10** |
| Max turns reached | 3 | 1 |
| Refusal-retry limit | 4 | 1 |
| Attacker generation error | 0 | 1 |
| CUDA OOM | 1 | 0 |

The same mechanism as in Phase 18 is visible: the NBF arm converts target access into
round-budget consumption (10 runs end with the loop budget exhausted after filtering), which
is why its trajectories are shorter (2.53 vs 3.80 mean turns) and its target-call count lower
(120 vs 189) despite identical `max_turns`.

NBF ON filtering: 75 of 195 candidate evaluations (38.5%) across 19 of 30 runs; 7 of those
runs still succeeded, 12 did not. Barrier score distribution (195 evaluations): mean −0.17,
median 0.0, min −0.85, max 0.0.

## 6. Runtime / GPU

| | NBF OFF | NBF ON |
| --- | ---: | ---: |
| Mean run time | 608.1 s | 443.6 s |
| Median | 442.8 s | 365.7 s |
| p95 | 1644.0 s | 1176.5 s |
| Throughput | 5.92 runs/h | 8.12 runs/h |
| Peak VRAM | 18.378 GiB | 18.378 GiB |
| OOM | 1 | 0 |
| Total wall clock (60 runs) | — | **8.76 h** |

As in Phase 18, the NBF arm runs faster because filtering consumes round budget and shortens
trajectories. **This is a trajectory-truncation effect, not evidence that the NBF is
intrinsically cheaper.**

## 7. Integrity

| Check | Result |
| --- | --- |
| Execution | **60/60**, 30 NBF OFF + 30 NBF ON, 30 matched cells |
| Unique run IDs | 60 unique, **0 duplicates** (`NBFXOFF_*` / `NBFXON_*` namespace) |
| Seed matching | **identical seed per matched cell** (`derive_seed(goal_id, attack, "off")` for both arms) |
| Configuration identity | threshold **0.0** (only), mode I1, T=0.7, max_turns 8, retries 0, repair NONE, fallback NONE, 3 attack prompt hashes, goals 30–39 |
| NBF OFF invariant | **0 evaluations** across all 30 OFF runs |
| Phase 18 protection | raw sha256 identical before/after; status `COMPLETE`; 90-cell result untouched |
| No Phase 19 contamination | only threshold 0.0; no override passed; Phase 19 directory untouched |
| Integrity status | **PASS** |

## 8. Interpretation (§21)

**OFF > ON → additional evidence that the Phase 18 direction generalizes beyond goals 0–29.**

Stated precisely, which is what the evidence supports and no more:

1. The **direction** of the Phase 18 observation reproduced on the next 10 sequential unseen
   goals under identical methodology: OFF 76.7% > ON 56.7%, discordant cells 8:2.
2. The **magnitude is not established** by this extension: Δ = −20.0 points here vs −15.6 in
   Phase 18, and the extension alone is not significant (p = 0.109, 30 cells). The combined
   descriptive figure (−16.7 points over 120 cells) is the more stable estimate, but it is
   descriptive only.
3. The **per-attack composition is not stable**: Acronym and Crescendo both favour OFF here
   while OppositeDay reverses, whereas Phase 18 had Crescendo tied and Acronym carrying the
   effect. The aggregate direction is reproducible; the mechanism's attribution to specific
   attacks is not.
4. No configuration change was made at any point. The extension was frozen before execution,
   ran to exactly 60 runs, and no threshold, attacker, prompt, temperature, NBF or protocol
   setting was touched — including in response to the results.

---

### Files

```
results/phase18_extension/
├── config/manifest.json            study definition + frozen-configuration record
├── config/preflight.json           23/23 preflight checks (§13)
├── config/phase18_reference.json   the frozen Phase 18 result, for reference only
├── raw/runs.jsonl                  60 run records (all §15 fields)
├── raw/nbf_off/runs.jsonl          30 NBF OFF runs
├── raw/nbf_on/runs.jsonl           30 NBF ON runs
├── progress/progress.json          run-level tracker + ETA
├── analysis/asr_comparison.json    extension-only ASR
├── analysis/matched_analysis.json  30-cell exact McNemar
├── analysis/combined_descriptive.json   120-cell descriptive (clearly labelled)
├── analysis/attacker_reliability.json · trajectory_metrics.json · nbf_analysis.json
├── analysis/runtime_gpu.json · integrity_check.json
└── final_report/phase18_extension_summary.json
```
