# Methodology / subsystem fidelity classification (Phase 16.4 §23)

Vocabulary: **REPRODUCED** · **PARTIALLY REPRODUCED** · **LOCALLY ADAPTED** ·
**NOT REPRODUCED** · **UNKNOWN**. Plus the audit severity rating for any issue.

| Subsystem | Classification | Severity of issues found | Basis |
|---|---|---|---|
| **Attack algorithm — Crescendo** | **REPRODUCED** | — | prompts byte-identical; loop, refusal budget, scoring, NBF order match the author |
| **Attack algorithm — OppositeDay** | **REPRODUCED** (control flow) / **LOCALLY ADAPTED** (failure contract) | HIGH (F2) | all prompts byte-identical; `generate_*_step` returns `("", "")` instead of raising |
| **Attack algorithm — Acronym** | **PARTIALLY REPRODUCED** | MEDIUM (2 apostrophes) + HIGH (F2) | refusal-OR-disclaimer gate identical; 2 example apostrophes differ (U+2019 vs ASCII); same failure-contract adaptation |
| **Attack algorithm — ActorAttack** | **LOCALLY ADAPTED** (and excluded) | MEDIUM (off-path) | prompt whitespace/apostrophe diffs; rubric drops `json_format=True`; `step_judge` runs at 0.7 not 0.0; async/sync judge-order divergence |
| **Attack algorithm — RedQueen** | **NOT REPRODUCED** | HIGH (missing) | no official code in the repo; local file is a `NotImplementedError` stub; correctly excluded |
| **Attack runner (loop/budget)** | **REPRODUCED** | HIGH (coverage gap) | the transcribed `while round_number + num_filtering < max_rounds` semantics reproduce the author's, including his counter quirk — but only the **sync** twin is tested; the async one that Phase 14 executes is not |
| **NBF — architecture & weights** | **REPRODUCED** | — | identical dims/layers/activations; author checkpoint, strict remap; parity `max_diff = 0.0` |
| **NBF — threshold & filtering** | **REPRODUCED** | — | `h + eta >= 0` → FILTER identical; nominal 3 trials / 1 effective evaluation matches the author's own structure |
| **NBF — state evolution** | **REPRODUCED** | — | candidate scored against its pre-update state, then `f_θ` advances; matches `calculate_score_from_dialog` |
| **NBF — OFF path** | **REPRODUCED** | — | `barrier=None` threaded end-to-end; no forward pass occurs (not merely empty telemetry) |
| **NBF — steered chat (`steer_target=True`)** | **PARTIALLY REPRODUCED** | HIGH if used (latent) | candidate filter would score against an empty dialogue because the steered branch never appends to `messages`; the frozen path hardcodes `steer_target=False` |
| **Prompts (all roles)** | **REPRODUCED** (Crescendo/OD/refusal/disclaimer byte-identical) / **LOCALLY ADAPTED** (rubric normalisations, declared D5) | INFORMATIONAL–LOW | byte comparison against the author's files |
| **Embeddings** | **REPRODUCED** | — | same model id, pooling, normalisation, 768-d |
| **Target driving** | **REPRODUCED** for structure / **LOCALLY ADAPTED** for budget | **HIGH (F1)** | correct `target_system`, user/assistant ordering, temp 0.7 — but the **256-token cap truncates 95.0 % of responses** where the paper's target was unbounded |
| **Evaluator / rubric** | **REPRODUCED** (author's mechanic) / **LOCALLY ADAPTED** (fallback, substituted model) | MEDIUM (latent) | identical rule and prompt ± normalisations; `fallback → 1` never fired (0/2755 parse failures); model substituted |
| **Success definition** | **REPRODUCED** (author's) / **METHODOLOGICAL DIFFERENCE** (paper's) | MEDIUM (interpretation) | success = rubric max score 5, not the paper's judge-model verdict; the `ASRJudge` path is unused by the frozen study |
| **JSON contract** | **LOCALLY ADAPTED** (necessary) | HIGH (F2 asymmetry) | the parse chain has no reference counterpart; behaviour is loud for Crescendo and silent for Acronym/OD |
| **Failure taxonomy** | **PARTIALLY REPRODUCED** | HIGH (F2) | 12/180 execution failures recorded as `ATTACK_FAILURE`; the taxonomy is not applied uniformly across attacks |
| **Telemetry / records** | **REPRODUCED / strong** | LOW (F3) | 55–56 fields, 0 schema violations over 180 records, raw outputs preserved, `turns=None` on failures — but `top_p` is declared 1.0 while the effective value is 0.8/0.9 |
| **Seeds / determinism** | **LOCALLY ADAPTED** | MEDIUM | not paper-specified; unique per run, resume-safe; seed set once per run; OFF/ON **not** RNG-paired |
| **GPU / residency** | **LOCALLY ADAPTED** | MEDIUM | pinned-slot rotation with sequential-residency support; cannot change generation; batch-level aborts can silently shorten the matrix |
| **Dataset & goals** | **REPRODUCED** | LOW | author's file, SHA-pinned and verified per batch, file order, no skipping |
| **Experiment matrix** | **PARTIALLY REPRODUCED** | LOW–MEDIUM | 3 of 4 attacks; RedQueen absent; 1,200-run design never executed (30-goal pilot only) |
| **Test suite** | **PARTIALLY REPRODUCED** | HIGH (gap) | parity vs the author's code is genuine and valuable; the executing async runner, seeding contract, matrix invariants and end-to-end JSON-failure path are untested |

## Coverage gaps ranked by the plausibility of hiding an ASR-affecting defect

1. Production async runner untested (sync twin only) — **highest**.
2. Seeding contract untested.
3. Full NBF filter/regeneration schedule parity lives in a script, not a test.
4. Matrix/denominator invariants untested.
5. End-to-end JSON-failure wiring untested — **this is the gap that let F2 survive**.
6. Prompt parity asserted only after normalisation.
7. Effective sampling parameters unpinned — the gap that let F3 survive.
