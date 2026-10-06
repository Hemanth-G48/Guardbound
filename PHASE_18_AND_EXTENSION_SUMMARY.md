# Phase 18 + Phase 18 Extension — Summary

**Does the Neural Barrier Function reduce attack success rate under the frozen C1 attacker
configuration?** Short answer: **yes, in the observed direction, by roughly 16–20 percentage
points, driven mostly by filtering that consumes attack round budget — but the effect is
conditional on the NBF threshold and its per-attack composition is not stable across goal
sets.**

Two experiments, reported separately:

| | Phase 18 (primary, frozen) | Phase 18 Extension (replication check) |
| --- | --- | --- |
| Status | **COMPLETE** — 180/180 runs | **COMPLETE** — 60/60 runs |
| Goals | 0–29 (30 goals) | 30–39 (next 10 sequential unused goals) |
| Matched cells | 90 | 30 |
| Wall clock | 41.89 h | 8.76 h |
| Run IDs | `NBFOFF_*` / `NBFON_*` | `NBFXOFF_*` / `NBFXON_*` |
| Result | OFF 47.8% vs ON 32.2%, Δ **−15.6 pts**, McNemar **p = 0.0201** | OFF 76.7% vs ON 56.7%, Δ **−20.0 pts**, McNemar **p = 0.109** |

---

## 1. Design (identical in both)

The **only** experimental variable is the NBF state (OFF vs ON). Everything else is frozen and
identical:

| Component | Value |
| --- | --- |
| Attacker | `Qwen/Qwen3.8-27B` rev `1d4bf0f2…`, NF4 + BF16 + double quant, thinking OFF, `max_new_tokens` UNSET |
| Attacker instruction | **C1**, generation-point mode `I1`, temperature 0.7, top_p 1.0 |
| Target | `meta-llama/Meta-Llama-3-8B-Instruct` rev `8afb486c…`, natural EOS, no token cap |
| Judge | `Qwen/Qwen3-8B` rev `b968826d…`, thinking OFF, temperature 0.0 |
| Attacks | Crescendo, OppositeDay, Acronym |
| NBF | checkpoint `cea1a75b…a136fe`, `NeuralBarrierFunction(dynamics, predictor)`, `all-mpnet-base-v2` (768-d), 768-d state, 5 classes, zero initial state, score `P(class5) − max(P(1–4))`, **threshold 0.0**, `eta = 0`, `steer_target = False` |
| Protocol | `max_turns = 8`, retries 0, repair NONE, fallback NONE |
| Success | `attack.was_successful()` ⇔ rubric score 5 |
| Seeds | `derive_seed(goal_id, attack, "off")` — **same seed for both arms of a matched cell** |

Filtering rule: `score < threshold → ACCEPT`, `score ≥ threshold → FILTER`.

## 2. Phase 18 — primary frozen result

```
NBF OFF : 43/90 = 47.8%   (95% CI 37.8 – 58.0)
NBF ON  : 29/90 = 32.2%   (95% CI 23.5 – 42.4)
Δ       : -15.6 percentage points  (relative 0.674, i.e. -33%)
```

**Matched analysis (90 cells, exact McNemar):**

| Cell outcome | n |
| --- | ---: |
| both succeed | 20 |
| **OFF succeeds, ON fails** | **23** |
| **OFF fails, ON succeeds** | **9** |
| neither succeeds | 38 |

→ discordant 32, **exact McNemar p = 0.0201**; unmatched Fisher p = 0.0476; mean paired turn
difference −1.42 turns (ON shorter).

**NBF behaviour:** OFF arm records **0 evaluations** (invariant); ON arm 825 evaluations,
266 filtered (**32.2%**), 60/90 runs affected, 11 successes still occurred after filtering.

**Reliability / runtime:**

| | NBF OFF | NBF ON |
| --- | ---: | ---: |
| Attacker calls (valid) | 958 (99.8%) | 834 (99.4%) |
| Target calls / natural EOS | 945 / 945 | 559 / 559 |
| Judge calls / failures | 1557 / 0 | 862 / 0 |
| Mean run time | 1015 s | 660 s |
| CUDA OOM | **10** | 1 |

## 3. Phase 18 Extension — replication check

```
NBF OFF : 23/30 = 76.7%   (95% CI 59.1 – 88.2)
NBF ON  : 17/30 = 56.7%   (95% CI 39.2 – 72.6)
Δ       : -20.0 percentage points  (relative 0.739, i.e. -26%)
```

**Matched analysis (30 cells, exact McNemar):** both 15 · **OFF-only 8** · **ON-only 2** ·
neither 5 → discordant 10, **p = 0.109** (Fisher p = 0.170).

**Direction: OFF > ON — the Phase 18 direction reproduced**, but the extension alone is
**not significant** at 30 cells (Phase 18 needed 90 cells to reach p = 0.0201).

**NBF behaviour:** OFF arm 0 evaluations (invariant); ON arm 195 evaluations, 75 filtered
(**38.5%**), 19/30 runs affected, 7 successes after filtering.

## 4. Combined descriptive view

**Descriptive only — this is NOT the Phase 18 result.** The 90-cell figure above remains the
primary frozen result and must never be relabelled as 120 cells.

```
120 matched cells (90 + 30)
NBF OFF : 66/120 = 55.0%   (95% CI 46.1 – 63.6)
NBF ON  : 46/120 = 38.3%   (95% CI 30.1 – 47.3)
Δ       : -16.7 percentage points
```

## 5. Per-attack composition is *not* stable across goal sets

| Attack | Phase 18 (n=30 each) | Extension (n=10 each) |
| --- | --- | --- |
| Crescendo | OFF 15/30 vs ON 15/30 (tied) | OFF 8/10 vs ON 5/10 (favours OFF) |
| OppositeDay | OFF 16/30 vs ON 13/30 (favours OFF) | **OFF 7/10 vs ON 9/10 (favours ON)** |
| Acronym | OFF 12/30 vs ON 1/30 (strongly favours OFF) | OFF 8/10 vs ON 3/10 (favours OFF) |

The **aggregate direction reproduces; the attribution to specific attacks does not.** In
Phase 18, Acronym carried essentially the whole effect and Crescendo was untouched; in the
extension Acronym *and* Crescendo both favour OFF while OppositeDay reverses.

## 6. Mechanism — filtering is a budget/access effect, not just semantic blocking

- Filtering **gates target calls** (Phase 18: 945 vs 559; extension: 189 vs 120).
- Fewer target calls → shorter trajectories (Phase 18: 4.19 → 2.77 mean turns; extension:
  3.80 → 2.53).
- Phase 18 ON-arm terminations include **41 loop-budget-exhausted** runs; extension, **10** —
  filtered candidates consume round budget without producing progress.
- Lower threshold sensitivity confirms the chain: filtering rate is what moves, and ASR follows
  it (see §8).

**OOM must not be misread.** Phase 18 OOMs (10 OFF vs 1 ON) are **refusal-loop artifacts** —
OFF runs regenerating context with ~0 accepted turns until the 24 GiB card fills. The NBF arm
avoids them because filtering consumes the budget first. Likewise the NBF arm's faster runtime
(−355 s/run in Phase 18, −165 s in the extension) is **trajectory truncation, not evidence that
the NBF is intrinsically cheaper.**

## 7. Integrity

Both experiments: **all runs genuinely executed, none retried, repaired, replaced or dropped**;
failures (OOMs, attacker contract failures) stay in the denominator.

| Check | Phase 18 | Extension |
| --- | --- | --- |
| Execution | 180/180, 90 cells | 60/60, 30 cells |
| Unique run IDs / duplicates | 180 / 0 | 60 / 0 |
| Seed matching per cell | ✔ | ✔ |
| Threshold | 0.0 only | 0.0 only (no override) |
| Config identity | mode I1, T=0.7, max_turns 8, retries 0, repair/fallback NONE | same |
| Integrity status | **PASS** | **PASS** |
| Phase 18 data hash | `02e11c0fcd975dfa00a97258a4dc2dde…` — unchanged before/after the extension |

The only source change in this whole line of work is an **additive** `safety_threshold_override`
parameter (default `None`) in `src/guardbound/attacks/runner.py`, used solely by the separate
Phase 19 threshold study; both Phase 18 and its extension run the legacy default path
(threshold 0.0). Test suite: **804 passed / 2 pre-existing unrelated failures**.

## 8. Caveats — what these results do and do not establish

1. **Threshold-conditional.** A separate Phase 19 sensitivity study (incomplete, 33/810 runs;
   classified `INCONCLUSIVE` on sample size) shows the NBF score distribution piles up on the
   decision boundary: **52% of the 825 Phase 18 scores lie within ±0.001 of threshold 0.0**, and
   that ±0.001 step produces the largest single-step change in filtering in the whole range
   (32.2% → 67.6% / 15.4%). The −15.6-point figure is therefore **conditional on the
   pre-registered threshold 0.0**, and should be reported as such. Threshold 0.0 should stay
   frozen as the pre-registered value, not adopted as an optimum.
2. **Local substitutes, not paper-scale.** Local models (Qwen3.8-27B attacker, Llama-3-8B
   target, Qwen3-8B judge) and a reference NBF checkpoint on an mpnet-768 state. ASR values are
   not comparable to the paper's numbers, and "GPT-4o equivalence" is not claimed.
3. **Magnitude is not settled.** −15.6 pts (Phase 18), −20.0 pts (extension, ns), −16.7 pts
   (combined, descriptive). The combined estimate over 120 cells is the most stable, but it is
   descriptive by construction.
4. **Not threshold-optimised, not attacker-optimised.** C1 was frozen throughout; no retries,
   repair, fallback, prompt changes or threshold tuning were introduced at any point — including
   in response to results.

## 9. Where the artifacts live

```
results/phase18_nbf_experiment/          Phase 18 (180 runs, primary frozen result)
  raw/run_level_metrics.jsonl            all 180 run records (raw model outputs included)
  analysis/                              9 JSONs + 7 figures
  final_report/phase18_report.md         full report
  final_report/phase18_summary.json      machine-readable headline numbers

results/phase18_extension/               Extension (60 runs, goals 30-39)
  raw/runs.jsonl + raw/nbf_{off,on}/     60 run records
  analysis/                              8 JSONs (incl. combined_descriptive.json)
  final_report/phase18_extension_report.md   full report
  final_report/phase18_extension_summary.json

PHASE_18_NBF_EXPERIMENT_REPORT.md        repo-root mirror (Phase 18)
PHASE_18_EXTENSION_REPORT.md             repo-root mirror (extension)
```

Reproduce / extend (both are resumable by run ID; no completed run is ever re-executed):

```bash
python scripts/phase18_nbf_experiment.py --goals 30 --goal-start 0   # Phase 18
python scripts/phase18_extension.py                                  # extension (60 runs, fixed)
```

Frozen baseline configuration: `configs/reproduction_phase14_frozen.yaml`
(NBF checkpoint path + expected sha256, threshold 0.0, eta 0.0).
