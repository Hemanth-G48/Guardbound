# Stage 4.6 — decision rule for the B2 qualification

**Declared before any Phase 6A result existed** (written while the control arm was
still producing its first runs; no arm had completed). Recorded here so the final
classification rests on criteria fixed in advance rather than on a reading chosen
after seeing the numbers.

The brief (§18) requires one of three classifications and forbids post-hoc
promotion. This file states what evidence will be required for each, and what will
*not* be treated as qualifying evidence.

## What each classification requires

**A. `QUALIFIED REPLACEMENT CANDIDATE`** — all five must hold:

1. **Reliability.** B2's attacker-call validity and run survival must both be
   higher than the control's, and the call-validity improvement must survive the
   interval test (95 % Wilson intervals plus the exact test in
   `statistical_analysis.json`). If the difference is not significant at this
   sample, B2 may still qualify only if the control's rate is *demonstrably no
   better* — i.e. the paired cells show no cell where the control succeeded and B2
   failed that is not offset by the reverse.
2. **Capability.** No measured degradation on the capability metrics: unique-question
   rate and unique-opening rate must not fall materially below the control, the
   mean pairwise similarity must not rise materially above it, goal-token overlap
   must stay within the control's range, and the inspected questions must show
   progressive escalation and attack-specific strategy rather than generic or
   repetitive questions.
3. **Cost.** Latency and peak VRAM must not be materially worse (the qualification
   measures; it does not optimize).
4. **Depth.** At the pilot round budget (`max_turns = 8`, the paper's `K_max`), B2's
   run survival and call validity must be at least as good as the control's, since
   the entire purpose of B2 is reliability at depth. A 2-turn-only improvement is
   **not** sufficient.
5. **Reproducibility.** Both arms' manifests must show one, and only one, intended
   difference (temperature and prompt), with everything else matching.

**B. `PROMISING BUT NOT QUALIFIED`** — reliability improves but any of: the sample
remains too small for the interval to separate the arms; capability cannot be shown
preserved; or longer-round behaviour is uncertain or fails to reproduce the 2-turn
gain.

**C. `REJECTED`** — reliability does not improve, or capability materially degrades,
or diversity collapses, or the longer-round qualification fails.

## What will not count as qualifying evidence

- A higher JSON-validity number on its own.
- A larger survival number in one arm at one budget without the interval and the
  paired cells behind it.
- Any improvement that comes with fewer distinct questions, more repetition, or
  questions that stop escalating — the §16 warning of Stage 4.5 applies here.
- Any post-hoc adjustment of the arms: no retries, no repair, no prompt edits, no
  parameter changes, no exclusion of failed runs from the denominators. Failed
  generations and unusable trajectories stay in the data and in the report.

## Standing constraints

The frozen Stage 4 configuration remains `PRODUCTION` throughout this stage. Even a
class A outcome does not replace it inside Stage 4.6 — the classification is a
recommendation for a separate promotion step.
