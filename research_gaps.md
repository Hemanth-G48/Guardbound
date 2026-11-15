# Research Gaps, Methodology Faults, and Improvement Ideas

Scope: the NBF Q-filter defense (Hu, Robey, Liu, arXiv:2503.00187v3) as reproduced
in this repo. Each item is tagged:

- **[repo]** — evidence found in this repo's own reports/results (file named).
- **[paper]** — a limitation of the original paper's method or protocol, as
  implemented here.
- **[idea]** — my suggestion; not verified by any experiment here.

Priority: **P1** = could change a headline conclusion · **P2** = strengthens the
work · **P3** = nice to have.

Report files named below live in `docs/reports/`.

---

## A. Faults and weaknesses in the current methodology

### A1. The effect size rests on a poorly calibrated barrier — P1 [repo]
- `PHASE_19_NBF_THRESHOLD_REPORT.md`: on 825 frozen evaluations, **52% of NBF
  scores fall within ±0.001 of the decision boundary** (61% in a wider band).
  The headline η is 1e-3, so the filter decision is dominated by noise at the
  boundary, not by class separation.
- `PHASE_18_COMBINED_TOTAL_ASR.md`: ASR drops 55.0% → 38.3% with the NBF, but the
  report itself says the effect works largely **by consuming the attacker's
  round budget** (filtered candidates use up turns), and is concentrated in
  Acronym (50% → 10%) while Crescendo moves −7.5 pts and Opposite-day −2.5.
- **Gap:** it is not shown that the barrier recognizes *dangerous trajectories*
  rather than simply rejecting many candidates. **Fix:** add a control that
  filters the *same fraction* of candidates at random (or by a trivial
  heuristic, e.g. a harm-classifier score) and compare ASR. If random
  filtering at equal rate gives similar ASR, the barrier adds nothing beyond a
  rate limiter.

### A2. Turn-0 state degeneracy — P1 [repo]
- `NBF_ANALYSIS.md` (an earlier model): with `x_0 = 0` the barrier predicted
  unsafe (`h ≈ +0.99`) for **every** first query, blocking whole dialogues,
  because a filtered turn doesn't advance the state, so the state stays zero.
- That model was trained with binary labels and later models changed, but the
  *structural* issue remains: **a filter that doesn't update state can
  deadlock**. **Fix/idea:** study initial-state choice (learned `x_0`, context
  initialization), and report first-turn false-positive rate for every
  checkpoint on benign openers.

### A3. Statistical power and reporting — P1 [repo][paper]
- `HANDOFF.md` key facts: the paper gives **no seed, no repetitions, no CIs**.
  Local pilots here use 30–40 goals and 120 matched cells; Phase 19's threshold
  study finished **33 of 810 runs** (GPU-time bound).
- McNemar/Fisher p-values are labelled *descriptive* by the repo itself. Attacks
  are stochastic (temperature 0.7), and attacker/target/judge all vary.
- **Fix:** repeated seeds per goal, goal-level (clustered) bootstrap CIs,
  report variance from attacker sampling separately from variance across goals,
  and pre-register the comparison and η before looking at results.

### A4. Local-model results don't transfer to the paper's claims — P1 [repo]
- `PHASE_15_NBF_OFF_BASELINE_ASR_ANALYSIS.md`: the local NBF-OFF baseline is "a
  local substitute-model figure, **not comparable to the paper's GPT-4o ASR**".
  Baseline ASR also differs a lot between phases (6.67% in Phase 15 vs 55% in
  Phase 18) because the attacker setup changed.
- The barrier was trained on conversations from one attacker/target
  distribution; evaluation uses different (open) models. Attacker identity
  alone moved ASR massively across phases (the Phase 15–17 attacker
  qualification series), so **ASR is highly attacker-dependent** and the
  defense's measured benefit may be too.
- **Fix:** report ASR for ≥3 attacker models per defense; treat attacker as a
  factor in the analysis, not a fixed choice.

### A5. Judge validity and circularity — P1 [paper][idea]
- Labels, attack-success decisions and MTBench scores all come from LLM judges
  (GPT-4o; local Llama pseudo-labels as the free alternative). Training labels
  and the evaluation judge share a family and rubric, so the NBF may learn the
  **judge's** notion of unsafe rather than real harm.
- No inter-judge agreement or human audit is reported in the repo that I found.
  The repo's Phase 14.4 evaluator qualification suggests judge behavior was
  already a concern.
- **Fix:** human-label a stratified sample (~300 turns); report judge–human
  agreement (κ); evaluate with a *different* judge than the one used for
  training labels; test pseudo-label (Llama) vs GPT-4o label quality
  explicitly (`label_conversations_with_llama.py` exists, but the link from
  label noise to NBF quality isn't quantified).

### A6. Threshold sign/semantic bugs show fragility — P2 [repo]
- `PHASE_23_DISCREPANCY_REGISTER.md`: D-01 (η sign inverted at the attack
  level), D-02 (ActorAttack ignored `steer_target`), D-03 (failed goals counted
  as failed *attacks* in the ASR denominator; ≈5 pts of bias at a 5% failure
  rate). All were invisible because earlier runs used η = 0.
- Fixed in code, but **earlier result files predate the fixes**. **Fix:** a
  results provenance table (which run used which code/commit/η rule), and
  re-run or explicitly quarantine anything produced before Phase 24.

### A7. Qualification probes tested the wrong thing — P2 [repo]
- `PHASE39_R1_QUALIFICATION_ROOT_CAUSE_REPORT.md`: both gpt-4o qualification
  failures were mainly **probe-design defects**; the model wasn't shown
  incapable. The gate is Guardbound-specific (not in the paper), so it can
  block valid runs. **Fix:** validate probes against a known-good run before
  using them as a gate; log the false-block rate.

---

## B. Gaps in the original paper's method (open research questions)

### B1. Adaptive adversary is weak — P1 [paper]
E7 ("NBF-adaptive Crescendo, 3-sample worst case") is a mild adaptive attack.
Stronger adaptivity would include:
- **Score-guided query search**: query the barrier many times; it is a cheap
  5-class MLP on a public embedding.
- **Embedding-space attacks**: craft messages whose embeddings land in low-`h`
  regions while keeping the intent (the barrier sees only a 768-d sentence
  embedding).
- **State poisoning**: admitted turns update `x`, so use several benign turns
  to move `x` into a region where `h` is permissive, then ask.
- **Transfer attack**: attack a surrogate NBF, then transfer.
- **Idea:** report ASR vs attacker *query budget* against the filter, not a
  single number.

### B2. Embedding bottleneck — P1 [idea]
The barrier only sees a **single sentence embedding per turn**. Paraphrase,
translation, encoded/obfuscated text, long turns (embedding truncation), and
role-play framing can shift embeddings without changing harmfulness.
- **Test:** a paraphrase/translation/encoding robustness suite; measure `h`
  stability. **Improve:** ensemble of embedders (the repo already has mpnet and
  distilroberta), or a safety-tuned encoder.

### B3. Only *user* turns are scored — P2 [paper][idea]
The filter looks at `U_k` and the state, not the **model's reply**. Attacks that
extract harm from replies ("continue your previous answer", summarization
requests) can look innocent on the user side. The dynamics model `g_θ`
predicts the response embedding, but it is not used for filtering.
- **Idea:** add output-side filtering, or use the gap between predicted and
  actual response embedding as a signal.

### B4. Empirical barrier, no guarantee — P2 [paper]
The "barrier function" language implies forward invariance, but the safe set is
defined by a learned classifier, `f_θ` is a learned approximation, and
invariance is enforced only on training transitions via a hinge loss. There is
no verified bound on `f_θ` error or on invariance outside the training data.
- **Idea:** measure invariance violations on held-out transitions: the rate at
  which a turn admitted at `h < −η` is followed by an unsafe label. Better
  still: conformal/calibrated bounds on `h`.

### B5. A single "unsafe = 5" cut — P2 [paper]
5-level judge scores are collapsed into unsafe (5) vs safe (1–4). Turns scored 4
("almost harmful") count as fully safe, and barrier quality is likely sensitive
to this cut.
- **Ablation:** unsafe ∈ {4,5}; regression on the 1–5 score; an ordinal loss.

### B6. κ and loss-weight sensitivity unexplored at scale — P3 [repo]
E3/E4 define ablations (λ ∈ {10,100,1000}, κ ∈ {2,3,4}), but local runs haven't
completed them at a power that supports conclusions.
- **Idea:** do them on a cheap proxy (classification metrics + offline replay of
  recorded conversations) and run attacks only for the best 2–3 configs.

### B7. Generalization to unseen attack families — P1 [paper]
E6 uses leave-one-out and Red-Queen. Also test: PAIR/TAP-style attacks spread
over turns, many-shot/long-context attacks, task-decomposition attacks,
multilingual attacks, tool/agent contexts. And the **benign side**: customer
support, code, medical and long-document chats. Over-refusal is currently only
measured on XSTest-like sets.

### B8. Over-refusal measured on short benign prompts — P2 [paper]
XSTest / PHTest / JBB are short, single-turn. The paper's claim is about
*multi-turn* robustness, so the cost side should also be measured on
**multi-turn benign dialogues** (long coding or tutoring sessions), where the
state keeps advancing and drift could cause false blocks late in a chat.
- **Metric to add:** per-turn false-block rate as a function of turn index and
  conversation length.

### B9. Behavior beyond 8 turns — P2 [idea]
Training and evaluation cap at 8 turns. Real chats are longer. State saturation,
drift, and compounding false positives are untested.

### B10. Cost/latency — P3 [idea]
The filter adds an embedding call plus two small MLPs per turn, presumably
cheap, but this isn't measured. Report the added latency and compare it with
guard-model baselines (Llama Guard etc.), which cost a full LLM call.

---

## C. Evaluation-design improvements

| # | Improvement | Why | Pri |
|---|---|---|---|
| C1 | **Matched random/heuristic filter control** at equal filter rate | Separates "barrier knowledge" from "attacker budget burn" (A1) | P1 |
| C2 | **ASR vs η Pareto with CIs**, and ASR vs *helpfulness* in one plot (scaffolded in `analysis/pareto.py`) | Shows the safety/utility trade-off honestly | P1 |
| C3 | **Attacker-as-factor** analysis (≥3 attackers) | A4 | P1 |
| C4 | **Per-turn analysis**: at which turn the unsafe trajectory is first caught, and how many turns before harm | Tests the *proactive* claim, which ASR alone doesn't | P1 |
| C5 | **Early-warning lead time**: turns between the first `h > −η` and the first judged-unsafe reply | Quantifies "proactive" | P2 |
| C6 | Human audit of the judge (A5) | Validity | P1 |
| C7 | **Stronger baselines**: Llama Guard 3 / ShieldGemma applied to the *whole conversation so far*, and a simple "classify the cumulative transcript with an LLM" filter | The NBF must beat a straightforward conversation-level classifier to justify its complexity | P1 |
| C8 | Report **attack-level filtering** (candidate rejected, attacker retries) and **target-level steering** (refusal returned) separately | The repo shows these are different mechanisms (D-02); they answer different threat models | P2 |
| C9 | Provenance: tie every table to commit, config and dataset hash (manifests exist; add an automatic results index) | A6 | P2 |

---

## D. Extension ideas (new contributions)

1. **Calibrated barrier.** Temperature scaling / conformal calibration so that
   η has a meaning ("≤ x% of safe trajectories filtered") instead of being tuned
   by ASR. Addresses A1.
2. **Learned or context-conditioned initial state `x_0`.** Fixes the degenerate
   first turn (A2) and supports system-prompt-aware filtering.
3. **Bidirectional barrier** (user turn + model reply). Closes B3.
4. **Embedding-robust barrier**: train with paraphrase / translation /
   adversarial-embedding augmentation, or ensemble embedders. Closes B2.
5. **Adversarial training of the barrier** against the adaptive attacker
   (min-max: the attacker searches for low-`h` harmful trajectories, the barrier
   is retrained on them). Closes B1.
6. **Soft intervention instead of hard refusal**: near the boundary, inject a
   safety prompt, steer the reply, or ask a clarifying question. This reduces
   over-refusal where the barrier is least confident (A1: 52% of scores are near
   the boundary).
7. **Uncertainty-aware cascade**: use MC-dropout or ensembles; filter only when
   confidently unsafe, and escalate uncertain cases to a heavier judge (cheap
   NBF first, expensive guard second).
8. **Longer-horizon training**: more than 8 turns, variable length, with benign
   long conversations in the data (B8/B9).
9. **Cross-model transfer study**: train on target A's conversations, deploy on
   target B. Tests whether the barrier is model-specific (A4).
10. **Interpretability**: which directions in embedding/state space the barrier
    uses. The repo's E9 PCA is a start. Probing for topic vs *intent trajectory*
    would show whether it is truly dynamic or just a topic detector.

---

## E. A minimal plan that would most improve the work

1. **Equal-rate random filter control** (C1). Cheap, and it may reshape the story.
2. **Human audit of the judge** (C6) on ~300 turns.
3. **Offline replay evaluation**: score *recorded* conversations at many η
   without any LLM calls. This gives full ROC/PR curves for the barrier itself
   (is it a good classifier of trajectory risk?) at zero API cost, and makes A1
   measurable at scale.
4. **Stronger baseline** (C7): whole-transcript guard-model classification.
5. **Multiple attackers × multiple seeds** with clustered bootstrap CIs (A3, A4).
6. **Embedding-robustness suite** (B2). A paraphrase/translation attack on the
   barrier is probably the cheapest way to find a real weakness.
7. Then, and only then, a stronger adaptive attack (B1) as the headline
   robustness experiment.

---

## Caveats on this document

- Sections A–C cite what the repo's reports state; I did not re-run any
  experiment. Numbers (55.0% → 38.3%, 52% within ±0.001, 33/810 runs) are copied
  from those reports and refer to specific frozen configurations (C1 attacker,
  threshold 0.0, local models).
- Items tagged **[idea]** are hypotheses; some (embedding-space attacks, state
  poisoning) may turn out harder than described.
- I did not read the paper's full text for this document. **[paper]** items are
  inferred from the repo's description of the method and from `HANDOFF.md`.
