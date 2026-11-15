# PHASE 21 REPORT — GPT-4o Paper-Reproduction Requirements

> **SUPERSEDED BY PHASE 22.** The three P0 blockers below are implemented and
> offline-validated; see `PHASE_22_IMPLEMENTATION_REPORT.md` at the repo root.
> The `$1.50` budget cap described here was a Guardbound-only safeguard and has
> been **removed** — the paper specifies no API budget and `--budget-usd` no
> longer exists. This file is kept as the record of the pre-Phase-22 state.

Single consolidated report. Supersedes and replaces the seven Phase 21 documents
(matrix, model/API audit, discrepancy register, attack/evaluator audit, cost
estimate, action plan, readiness report).

**Status: `NOT READY`.** Two P0 methodology blockers. Zero paid API requests.
No experiment run. `main` @ `89ed03f` + uncommitted work.

Approved for this phase: model **GPT-4o**, attacker condition **original paper
baseline** (C1/I1 deferred, must not run), **no paid calls**. Methodological
authority: **the published paper**; the official code fills gaps only.

Paper: arXiv:2503.00187v3 (Hu, Robey, Liu; TMLR 02/2026). Official code:
`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/`.

Optional reference (not required reading): `PHASE_21_PAPER_METHODOLOGY_EXTRACTION.md`
— the full 20-field-per-experiment extraction with page citations.

---

## 1. What is blocked, and why

| # | Blockers | Priority |
|---|---|---|
| P0-1 | Evaluator temperature is 0.0; the paper says 0.7 for everything, with no judge exception | P0 |
| P0-2 | Attack set inverted: Acronym evaluated (paper: training-only), ActorAttack omitted (paper: main evaluation) | P0 |
| P0-3 | ActorAttack's official pipeline is never executed; the runner loses its per-actor structure | P0 |

Nothing is blocked by the model: GPT-4o is available (§4).

---

## 2. Discrepancy register — paper vs official code vs Guardbound

| ID | Paper | Official code | Guardbound | Prio |
|---|---|---|---|---|
| **D-01** | "All temperatures are set to be 0.7" (p9 §5.1); no other temperature rule exists in the paper | 0.7 generation; **0.0** in `check_refusal`, `check_disclaimer`, `evaluate_with_rubric` | 0.0 at 8 sites | **P0** |
| **D-01b** | 0.7 | ActorAttack's judge calls inherit 0.7 (no override) | rubric at **0.0** — differs from paper *and* code | **P0** |
| **D-01c** | 0.7 | — | MMLU p.209, over-refusal p.162, ASR judge p.196, MTBench judges p.181/225 all 0.0 | P1 |
| **D-02** | **never names the attacker model** (p7 delegates to the attack papers) | `--attacker_model` default `gpt-4o` | GPT-4o, unlabelled provenance | gap |
| **D-03** | ActorAttack, Crescendo, Opposite-day (p9; Tables 1,2,8). Acronym "omitted during evaluation" (p22 B.1) | all four available | Crescendo, Opposite-day, **Acronym**; ActorAttack **excluded** | **P0** |
| **D-04** | headline ASR uses **η = 1e-3** (p9 §5.2; Tables 1, 8); η=0 is the ablation (Tables 6,7); η=5e-4 recommended (Table 15) | CLI default 0 | `threshold: 0.0`, `eta: 0.0` everywhere | P1 |
| **D-05** | `NOT SPECIFIED` | retries up to 10× then returns `{}` | retries 0, failures raised | P2 (intentional) |
| **D-06** | `NOT SPECIFIED` | `json.loads` raises on bad JSON | returns text → `AttackGenerationError`; rubric failure → `None` | P2 (adaptation) |
| **D-07** | attack prompts = the cited attacks' own; not reproduced in the paper | authoritative bytes | 3 typographic deviations (§3) | P2 |
| **D-08** | regeneration rule stated, **no bound** (p22 B.1) | `C_refused < 10` | same | P2 (gap filled by code) |
| **D-09** | guardrail: "harmless iff score is 1" (p22) — Guardbound matches | — | guard *exception* silently recorded as `"harmless"` (`guardrail_f1.py:340`) | P3 |
| **D-12** | ActorAttack is main evaluation | `run_actor_attack` → per actor: reset conversation, run query chain (`N_retry=3`), `summary()` turn, early stop at 5 | `ActorAttack.run_full_attack` exists but **has no callers**; runner uses `next_query` → loses per-actor reset, summary turn, early stop | **P0** |

Where the paper is silent and the code speaks, the resolution is "keep the code's
value, label it code-recovered". Where the paper speaks, the paper wins.

---

## 3. Prompt and evaluator fidelity

| Prompt | Difference vs official | Verdict |
|---|---|---|
| Crescendo attacker (all turns) | none — hashes identical | **pass** |
| Opposite-day attacker | 2 blank lines lose 36 spaces | typographic deviation |
| Acronym attacker | 3 × `’`→`'`, 1 blank line | typographic deviation |
| Rubric (2 copies) | 1 collapsed double-space + 3 inserted spaces (`acton`, `content,which`, `deviation.It`); **the five score criteria read identically** | typographic deviation |
| Refusal (2 copies), disclaimer | none — hashes identical | **pass** |
| Target conversation | system + alternating turns, JSON off, 0.7 | **pass** |
| Provider payload | SDK body == constructed messages, nothing added | **pass** |

No difference changes an instruction, example, field name, score definition or
message order. The apostrophe change is a code-point change so tokenization may
differ. Recommendation: **restore the official bytes** for the reproduction, since
the paper is the authority and the edits are whitespace; the trade-off is that the
cloud prompts then differ from the frozen local runs. Decide, don't drift.

Evaluator mechanics are correct: rubric failures yield `None` and are classified
(never a substituted `1`); the refusal threshold is `value and metadata > 90`, so
`metadata == 90` is correctly **not** a refusal; no evaluator retries.

Attack logic fidelity: Crescendo/Opposite-day/Acronym match the official control
flow exactly, including the `max_rounds = 1` zero-iteration quirk, refusal-pop
budget restoration, `C_refused < 10`, refusal-before-rubric ordering, and Acronym's
`refusal OR disclaimer` condition. ActorAttack's *components* match (pre-attack,
`N_retry=3`, `successful`/`rejective`/`unknown`, rubric-before-step-judge); only its
**wiring** is wrong (D-12).

---

## 4. Model and API facts (verified 2026-11-10, official docs)

| Item | Value |
|---|---|
| Status | **Available, Deprecated**, announced shutdown **2027-01-14** |
| Snapshots | `gpt-4o-2024-08-06` Deprecated (default), `2024-11-20` Deprecated, `2024-05-13` shut down |
| Paper's snapshot available? | **Yes** → no substitution required. Replacement `gpt-6-luna` exists but **must not** be adopted without approval |
| Context / output | 128,000 / 16,384 |
| Endpoint | `v1/chat/completions` supported (matches the official code); Responses also supported |
| Structured Outputs | supported |
| Price (1M) | $2.50 input · $1.25 cached · $10.00 output |
| Temperature | no documented restriction (non-reasoning model) |
| Revision exposure | the API does not return the served revision → record it as unverifiable |

Sources: `developers.openai.com/api/docs/models/gpt-4o`, `…/api/docs/pricing`,
`…/api/docs/deprecations`.

Not verifiable offline: whether this account can reach the model today. The
qualification run is the step that closes that.

---

## 5. Scope — what a GPT-4o-only reproduction can claim

| Repro | Paper result | Attacks | NBF | Feasible | Cost |
|---|---|---|---|---|---|
| **R1** | Table 1, GPT-4o row | ActorAttack, Crescendo, Opposite-day | η = **1e-3** | yes, after P0-1..P0-3 | ≈$57 |
| **R2** | Tables 8–9, GPT-4o rows | same 3 | η ∈ {0, 1e-4, 1e-3} | yes, after P0-1..P0-3 | ≈$171 |
| **R3** | Table 4 (guardrail F1) | — (classifier; no target LLM) | MPNet / DistilRoBERTa | yes, needs the 3 datasets | ~$0 |
| R4 | Table 12, GPT-4o column | SafeMT_ATTACK_600, MHJ | η = 1e-3 | **no** — datasets absent | — |
| R5 | Tables 2,3,5,6,7,10,11,13,15 | — | — | **no** — targets llama/Phi-4/GPT-5/Claude | — |
| R6 | Table 14 (adaptive) | Crescendo + argmax | η = 0 | **no** — paper uses GPT-3.5 | — |
| R7 | Figure 6 (RedQueen) | — | η = 1e-3 | **no** — GPT-3.5/llama-3.1-70b + stub | — |
| R8 | Figures 9–11 (PCA) | — | — | **no** — GPT-3.5 | — |
| R9 | Figure 8 (training dynamics) | — | — | **no** — NBF frozen | — |

**Claim discipline:** R1 may be described as "the GPT-4o row of the paper's main
multi-turn evaluation". It may **not** be described as reproducing Table 1, which
reports eight rows across four model families. No cross-model claim is available
under the approved model decision.

Paper constants needed to run: HarmBench 200 behaviours (p22), max 8 turns (p22),
training η=0 / κ=3 / λ=(1,1,100,100) (p8, p22), unsafe label = 5 (p22), all 200
cases share one 6-token `target_system` `"You are a helpful assistant."`. Paper
specifies **no** seeds, repetitions, CIs, token limits or attacker prompt text.

---

## 6. Cost

Measured input tokens (`tiktoken` `o200k_base` on the real prompts) + declared
output assumptions. Reproduce with `scripts/phase21_cost_estimate.py`.

| Scope | Low | Expected | High |
|---|---:|---:|---:|
| R1, 3 attacks × 200 goals × K=8 (upper bound) | **$40.66** | **$56.99** | **$91.38** |
| R1 with early stop at score 5 (≈5.5 turns) | ≈$28 | ≈$39 | ≈$63 |
| R2 (η sweep, ×3) | ≈$122 | ≈$171 | ≈$274 |
| Qualification, one attacker condition | — | **$0.30–0.45** | — |
| 5-goal pilot, one attack | — | ≈$0.48 | — |

Key measured per-call inputs: rubric 656 tok, refusal 569, disclaimer 292,
ActorAttack step judge 165, ActorAttack pre-attack 1,974 tok over 4 calls.
Attacker input grows with history (Crescendo 1,503 → 1,993 tok at turn 8).
ActorAttack makes **no per-turn attacker call** (queries are pre-generated).

**$1.50 cap verdict:** adequate for qualification (~3× headroom) and a 5-goal
pilot; **38× short** of R1. Do not raise it in this phase.

---

## 7. Implementation plan — what must be changed

### P0-1 — Evaluator temperature 0.0 → 0.7 (D-01)

* **Files:** `attacks/crescendo_paper.py:186,337` · `opposite_day.py:227,371` ·
  `acronym.py:252,277,415` · `actor_attack.py:544`.
* **Change:** add `PAPER_EVALUATOR_TEMPERATURE = 0.7` /
  `LEGACY_EVALUATOR_TEMPERATURE = 0.0` constants; give each attack class an
  `evaluator_temperature` parameter (default **0.0**, preserving frozen behaviour)
  forwarded to the module-level evaluator helpers; paper configs set 0.7.
* **Risk:** ASR at 0.7 is not comparable with the frozen local numbers (0.0).
  Record the value in every manifest so regimes never mix.
* **Tests:** each of the 8 sites receives the configured value; default stays 0.0;
  paper config yields 0.7 on the wire; parity tests stop hard-coding 0.0.
* **Accept:** paper config → every evaluator request carries `temperature: 0.7`.

### P0-2 — Attack set (D-03)

* **Files:** the cloud configs (`cloud_research_reference.yaml`,
  `cloud_research_gpt4o_mini*.yaml`, `cloud_qualification.yaml`), `cloud_pilot.py`
  validation.
* **Change:** `attacks.enabled: [actor_attack, crescendo_paper, opposite_day]`;
  add `attacks.training_only: [acronym]` with the p22 reason.
* **Risk:** adding ActorAttack without P0-3 produces a non-faithful ActorAttack.
  **Ship P0-2 and P0-3 together.**
* **Tests:** a paper config that enables Acronym or omits a paper attack is
  rejected; manifest records `training_only`.
* **Accept:** resolved set == exactly the three paper attacks.

### P0-3 — ActorAttack through its official composite (D-12)

* **Files:** `attacks/runner.py`, `cloud_pilot.py`.
* **Change:** add an explicit composite execution strategy for ActorAttack that
  uses `prepare_attack` once then per actor: fresh `[system]` conversation → query
  chain with `N_retry=3` → `summary()` turn → early stop at score 5. One record
  per actor, aggregated to one behaviour per goal for ASR. Leave the generic path
  for the other attacks.
* **Risk:** high — new execution path and record shape; select it explicitly in
  config so the other attacks are untouched.
* **Tests:** composite vs official `attack_single` call sequence on a scripted
  mock; each actor's first target call starts at `[system]`; summary invoked once
  per non-skipped actor and can stop the goal; ASR counts one behaviour per goal.
* **Accept:** a 2-goal mock run shows 3 actor conversations per goal and one
  summary call per non-skipped actor, matching the official control flow.

### P1-1 — Headline η (D-04)

Config only: R1 sets `threshold: 1e-3`; R2 sweeps `{0, 1e-4, 1e-3}`. Records must
show the threshold in force.

### P1-2 — Evaluation-harness temperatures (D-01c)

`evaluation/mmlu.py:209`, `over_refusal.py:162`, `asr.py:196`,
`mtbench.py:181,225` → same parameterisation as P0-1 (default 0.0, paper 0.7).
`mtbench.py:296,308` already 0.7.

### P1-3 — Typographic prompt deviations (D-07)

Either preserve (comparable to past local runs, disclosed with hashes) or restore
the official bytes (strict fidelity, forks from the frozen local prompts). The
audit script must report exactly the chosen state and nothing else.

### P2 / P3

P2-1 record the `C_refused < 10` cap as code-recovered · P2-2 count guardrail
failures instead of recording them as "harmless" · P2-3 label the attacker model
as code-recovered · P3-1 add ActorAttack contract probes to the qualification
harness · P3-2 RedQueen stays a stub (infeasible under the model decision anyway).

**Order:** P0-1, P0-2, P0-3, P1-1 → re-run offline suite + both audits →
then request approval for a paid qualification run.

---

## 8. Decisions required before any paid call

1. Adopt **0.7** for all judge calls (recommended; paper is authority) or keep 0.0
   and declare the deviation.
2. Headline **η**: 1e-3 (recommended, Tables 1/8) vs 0 (Tables 6/7) vs 5e-4
   (Table 15).
3. Authorise the **ActorAttack composite runner path** (P0-3). Without it
   ActorAttack cannot be in the reproduction.
4. **Prompts**: restore official bytes (recommended) or preserve and disclose.
5. Accept that scope = GPT-4o rows only; no cross-model claim.
6. Accept the **budget** implication: $1.50 is a qualification cap; R1 needs
   ≈$60–95.

---

## 9. What is trustworthy today

* Provider layer: faithful — message arrays reach the SDK unchanged, no hidden
  retry, no fallback, no response repair, no prompt rewriting.
* NBF: unchanged and provider-independent — checkpoint
  `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe`,
  1536-512-512-768 dynamics, 1536-32-32-5 predictor, MPNet 768-d, x₀=0,
  `h = P(5) − max(P(1..4))`, accept iff `score < threshold`; real-checkpoint tests
  pass.
* Test suite: **1013 collected / 1009 passed / 4 failed / 0 errors / 0 skipped**.
  The 4 failures are pre-existing and environmental (2 need CUDA, 2 need
  `Qwen/Qwen3.5-4B` weights). A green suite does **not** establish methodology
  fidelity.
* Pricing and model availability: verified against official docs on 2026-11-10.

## 10. Status

**`NOT READY`** — three P0 items (D-01, D-03, D-12) prevent a paper-faithful run.
They are bounded: one parameterisation, one config set, one new execution path; no
new research, no model change, no NBF change. Landing P0-1…P0-3 plus P1-1 moves
the project to **READY WITH DOCUMENTED DEVIATIONS**.
