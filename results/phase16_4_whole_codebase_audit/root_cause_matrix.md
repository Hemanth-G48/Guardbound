# Root-cause matrix (Phase 16.4 §18 and §20)

Impact classification: **HIGH / MEDIUM / LOW / UNKNOWN**. Diagnostic only — no
component is ranked and no ASR was computed.

## A. Causal-style diagnostic table

| Potential cause | Evidence FOR | Evidence AGAINST | Remains unknown | Status | Confidence |
|---|---|---|---|---|---|
| **Local attacker capability** | All three papered roles were substituted; attacker parse failures 0.91 % per call (18/1978); Acronym 1.97 %; instruction-following is imperfect on a minority of calls | 150/162 `ATTACK_FAILURE` runs had **zero** parse failures — they failed on content, not format | The attacker's *content* quality vs GPT-4o is unmeasured (no ablation) | **CONTRIBUTOR, share UNKNOWN** | MEDIUM |
| **Local target capability** | Target refuses frequently (mean 5.31 refusals/OFF run) and full compliance is rare (6/90 at score 5) | Target engages in essentially every run and yields partial compliance (best score 3 in 53/90) — not blanket refusal | Whether partial compliance is a capability limit or an artifact of truncation | **CONTRIBUTOR, share UNKNOWN** | MEDIUM |
| **Local evaluator behaviour** | Substituted model; rubric semantics differ from GPT-4o calibration | **0 parse failures in 2,755 evaluator JSON calls**; the `fallback → 1` path never fired; the evaluator sees the correct `(query, response)` pair and no wrong turn; score distribution is smooth and plausible (3 dominant) | Whether Llama-3.2-3B's *severity calibration* matches GPT-4o's | **LOW (operationally sound)** | HIGH |
| **Attack implementation** | — | Line-level comparison vs the author's code: prompts byte-identical (Crescendo/OD), loops/scoring/refusal/budget identical; parity tests + a numerical NBF parity test | Nothing material found | **RULED OUT** | HIGH |
| **NBF implementation** | — | Architecture, checkpoint (0.0 diff), embedding, `h` formula, `h+eta>=0` rule, pre-update state order, OFF invariant all verified identical; no cross-run state leakage | — | **RULED OUT** | HIGH |
| **Prompt mismatch** | Acronym 2 apostrophes; ActorAttack whitespace (excluded attack); rubric 4 typo normalisations | All semantically inert; no JSON/reasoning/role text added or removed | — | **RULED OUT (informational only)** | HIGH |
| **Conversation construction** | — | Roles/order verified in live traces (system/user/assistant/user…), goal present every turn, no duplication, no truncation, evaluator gets the raw pair | — | **RULED OUT** | HIGH |
| **JSON interface** | The local parse chain is an addition vs the paper (server-side JSON mode) | Every malformed reply fails loudly for Crescendo; raw output preserved; no repair/retry/fallback anywhere | **Acronym/OppositeDay absorb malformed replies as empty queries (F2)** | **ACTIVE DEFECT (taxonomy), MEDIUM–HIGH** | HIGH |
| **Context truncation** | `truncation=True, max_length=131072` exists in code | Worst-case prompt ≈7.5 K of 131 K tokens — never reached; `max_turns_context` never passed | — | **RULED OUT (input)** | HIGH |
| **Generation parameters** | **Target output capped at 256 tokens: 95.0 % of 929 responses hit the cap and end mid-word (F1)**; `top_p` declared 1.0 but effective 0.8/0.9 (F3) | Temperature 0.7/0.7/0.0 all correct; no quantization; `do_sample` correct | The share of the success gap caused by the cap | **ACTIVE FACTOR (F1 HIGH, F3 MEDIUM)** | HIGH (fact) / MEDIUM (impact) |
| **Seed methodology** | OFF and ON use **different seeds** (+10000) and diverge further on filtered-candidate regeneration; attacker and target share one RNG stream | Both arms get identical goals, attacks, budget, threshold, stack; seeds are unique and resume-safe | — | **LIMITS INFERENCE, not validity** | HIGH |
| **GPU / runtime constraints** | Batch-level VRAM guards sit **outside** the per-run try → an abort leaves no failure rows (demonstrated at 108/180) | No OOM, no CUDA errors, 0 load refusals, peak 20.281 GB of 25.763; eviction cannot change generation (full re-encode per call, same dtype/attention) | — | **SURVIVORSHIP RISK (mitigated by resume)** | HIGH |
| **Dataset / goal distribution** | 30-goal pilot slice only; the paper's 200-goal set was not completed | Dataset SHA-pinned and verified; file order preserved; no skipping | Whether goals 30–199 behave differently | **UNKNOWN (scope limitation)** | MEDIUM |
| **Evaluation methodology** | The frozen study's "ASR" is the rubric's max-score-5, not the paper's judge-model verdict; the `ASRJudge` path is unused | The rubric mechanic *is* the author's success definition; failures are retained in the denominator; per-conversation denominator matches the paper | Whether a judge-model ASR would differ from the rubric verdict | **METHODOLOGICAL DIFFERENCE (declared)** | HIGH |

## B. Attribution matrix (paper vs local)

| Component | Paper | Local | Fidelity | Potential impact on low ASR | Evidence |
|---|---|---|---|---|---|
| **Attacker** | GPT-4o | Qwen3-4B-Instruct-2507 | substituted | **MEDIUM–UNKNOWN** | 0.91 % parse failures; content quality unmeasured |
| **Target** | GPT-4o | Phi-4-mini-instruct | substituted | **MEDIUM–UNKNOWN** | heavy refusals (5.31/run) + partial compliance (3 dominant) |
| **Evaluator** | GPT-4o | Llama-3.2-3B-Instruct | substituted | **LOW** | 0/2755 parse failures; plausible score distribution |
| **Attack algorithms** | author code | ported | **faithful** | **LOW / RULED OUT** | line-level + parity tests |
| **NBF** | author model | same checkpoint, remapped | **faithful (0.0 diff)** | **LOW / RULED OUT** | parity numerics; OFF invariant proven on path |
| **Prompts** | author constants | same ± inert whitespace/apostrophes | **high** | **LOW** | byte comparison |
| **Generation** | unbounded completions | **256-token cap on every role** | **deviating** | **HIGH** (F1) + **MEDIUM** (F3) | tokenizer proof; generation_config |
| **Context** | full | full (no truncation) | **faithful** | **LOW** | ≤7.5 K of 131 K |
| **Evaluation** | rubric + judge model | rubric max==5 | **faithful to the author's mechanic**, differs from the paper's judge | **MEDIUM (interpretation)** | success = `5 in _scores`; failures retained |

## C. Are we comparing the right things? (§19)

The paper's result is a property of the **joint system**. This audit can state
precisely which components differ:

* **Faithful**: attack algorithms, NBF (weights, architecture, threshold, state
  order, OFF invariant), prompts, conversation construction, success definition,
  dataset and its ordering, temperatures.
* **Substituted**: all three model roles.
* **Deviating for implementation reasons** (independent of model capability):
  the 256-token output cap (F1), the undeclared effective `top_p` (F3), and the
  failure-class asymmetry (F2).
* **Methodology-level**: non-RNG-paired OFF/ON arms; rubric-based rather than
  judge-model-based verdict; 30-goal scope so far.

Therefore the statement "ASR is low because the local attacker is weaker" is **not
supported as a complete explanation**: at least one deviating implementation
parameter (F1) plausibly depresses the numerator independently of model capability,
and it has never been varied. Equally, "the code is fine because tests pass" is not
supported: the async runner — the code that actually executes — is untested, which
is how F2 survived.
