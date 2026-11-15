# Phase 16.4 — Whole-Codebase Forensic Audit

**Date:** 2026-09-28
**Type:** READ-ONLY audit. No code, prompt, config, parser, NBF, dataset or result
file was modified. No ASR experiment, no pilot, no tuning.
**Question:** are the poor attack results primarily caused by the local-model
substitution, or could implementation / algorithmic / prompt / NBF / target /
evaluator / state / runtime / methodological issues contribute?

**Method:** four parallel read-only code investigations (attack fidelity vs the
author's shipped reference; NBF; evaluator/target/JSON; methodology/runtime/tests)
plus independent empirical re-analysis of the 180 frozen pilot records and direct
inspection of the cached model configs and tokenizer. Every headline claim below
was re-verified first-hand; the section that qualifies it names the check.

---

## 1. Executive summary

The attack algorithms, the NBF, the prompts and the evaluation rubric are
**faithful to the author's reference code** — verified by line-level comparison
against the author's own source shipped in `nbf_original_stuff/`, by the repo's
parity tests, and by direct numerical parity (checkpoint diff `0.0`). No
algorithmic or NBF defect was found that could explain the low attack success.

Three findings do matter, and two of them were **not previously measured**:

| # | Finding | Severity | Affects |
|---|---|---|---|
| F1 | **Target responses are truncated at 256 tokens in 95.0 % of turns** (median 256, max 256, 0 % under 200; tails end mid-word). The paper's GPT-4o target was unbounded. | **HIGH** | the ASR numerator itself (plausibly depresses full compliance) |
| F2 | **Acronym/OppositeDay attacker JSON failures are silently recorded as `ATTACK_FAILURE`**, not `JSON_PARSE_ERROR` — 12 of 180 runs (6.7 %). Crescendo reports them correctly. | **HIGH** | failure taxonomy / interpretation, **not** the numerator |
| F3 | **The declared `top_p = 1.0` is not enforced.** The loader never passes `top_p`/`top_k`, so sampling inherits the models' own configs: **`top_p=0.8, top_k=20`** for the Qwen3-4B attacker, `top_p=0.9` for the Llama-3.2-3B evaluator. | **MEDIUM** | sampling fidelity + telemetry accuracy |

**Answers:** Q1 implementation bug? — **YES, limited** (F2 is a real defect; F1 is
an unmeasured adaptation that plausibly depresses success; no algorithm/NBF bug).
Q2 prompt/methodology mismatch? — **YES** (F1, plus non-RNG-paired OFF/ON arms).
Q3 local models a contributor? — **YES as a contributor, share UNKNOWN** (all three
papered roles were substituted; no ablation isolates them). Q4 faithful enough to
blame models? — **NO/INCONCLUSIVE** while F1 and F2 are un-isolated. Q5 pilot —
**NOT AUTHORIZED** pending two decisions (§21).

---

## 2. Repository / code-state audit

* Commit `b46d6fd7ed37b7a6ee9e643476ae2b2263b85624`, branch `main`, working tree
  dirty with the Phase 14–16 uncommitted work (documented in `AGENTS.md`).
* Frozen study config: `configs/reproduction_phase14_frozen.yaml`,
  `config_hash 61ea203c20e68c51`; dataset SHA `ac789de8…`; NBF checkpoint SHA
  `cea1a75b…136fe`. All three re-verified unchanged by this audit.
* Largest real artifact: the Phase 15 pilot, **180/180 runs**,
  `FCB952F1598B697B033D605B0791CFBB…`, 55–56 fields per record, 0 schema
  violations (Phase 15 audit). The 1,200-run study has **never been launched**.
* Provenance trap (LOW): the runner's module default is
  `configs/reproduction_three_model.yaml` (`phase14_full_reproduction.py:73`),
  whose stack differs from the frozen study's. Omitting `--config` silently
  selects a different model stack. All recorded runs passed `--config`
  explicitly, so no recorded result is affected.

## 3. Reference methodology reconstruction

Full table: `reference_vs_local_matrix.md`. Summary of the classification:

| Component | Verdict |
|---|---|
| Attack algorithms (Crescendo, OppositeDay, Acronym) | **EXACT MATCH** in control flow, scoring, refusal/backtracking, loop budget |
| Prompts (Crescendo, OppositeDay) | **EXACT MATCH** (byte-identical) |
| Prompts (Acronym) | 2 Unicode apostrophes differ (U+2019 vs ASCII) — semantically inert |
| Rubric prompt | 4 typo/whitespace normalisations — declared (D5), semantically inert |
| Refusal / disclaimer detection | rule `value and metadata > 90` identical; prompts byte-identical |
| NBF architecture, checkpoint, embedding, threshold, state order | **EXACT MATCH** |
| Attacker / target / evaluator models | **METHODOLOGICAL DIFFERENCE** — GPT-4o ×3 → local Qwen3-4B / Phi-4-mini / Llama-3.2-3B |
| `max_new_tokens=256` on the target | **METHODOLOGICAL DIFFERENCE with measured HIGH effect** (F1) |
| `top_p` | **POTENTIAL BUG in the telemetry/declaration** (F3) |
| ActorAttack / RedQueen | excluded / reference absent (unchanged status) |

## 4. Attack-algorithm audit

Official reference code is present for four of five attacks and was compared
directly (`nbf_original_stuff/…/NBF-LLM/attacks/`).

* **Crescendo** — system prompt, round-1 boilerplate and per-round feedback are
  **byte-identical**; message construction, `(Summary)`-based history bookkeeping
  (`round_number > 1` ⇔ accepted>0), refusal-before-rubric ordering, `C_refused < 10`,
  `score == 5` termination, and the
  `while round_number + num_filtering < max_rounds` loop with its "previous
  iteration's counter" quirk all match.
* **OppositeDay** — all prompts byte-identical; same control flow.
* **Acronym** — prompts match except two apostrophes; refusal **or** disclaimer
  gate and budget match.
* **ActorAttack** — excluded from the study. Prompts differ by one apostrophe and
  three trailing spaces; its rubric call omits `json_format=True`
  (`actor_attack.py:538`) and `step_judge` omits `temperature=0.0`
  (`actor_attack.py:426`), both deviations from the author. Not on any recorded path.
* **RedQueen** — **no reference implementation exists** in the repo; the local file
  is a `NotImplementedError` stub. Correctly excluded.

**Candidate filtering**: nominal `range(3)` with an unconditional `break` in both
branches ⇒ **one barrier evaluation per candidate**, matching the author's
identical structure. `filter_trials = 3` in the records is the declared config
value, not the executed count — correctly documented.

**Latent, off the frozen path** (`steer_target=True`, used by
`run_reproduction.py` mode C, not by Phase 14): the candidate filter would score
candidates against an effectively empty dialogue state, because the steered branch
never appends to `messages`. HIGH if ever used; **no effect on any recorded
result** (Phase 14 hardcodes `steer_target=False`).

**Latent**: `get_attack("crescendo")` resolves to the *heuristic* Crescendo
(`ESCALATION_PROMPTS`), not the author-faithful `crescendo_paper`. The frozen study
uses `crescendo_paper`; `DEFAULT_EVALUATION_ATTACKS` lists `"crescendo"`.

## 5. NBF audit

* Architecture identical to the reference `NeuralStateSpaceModel` /
  `NeuralBarrierFunction`: 768/768 state+input, 512×2 hidden, 768 outputs,
  32×2 predictor hidden, 5-class head, ReLU only, linear output, softmax applied
  outside the model. **No mismatch found.**
* Barrier score `h = p(class 5) − max p(classes 1–4)` — identical formula.
* Filter rule `h + eta >= 0` → FILTER: verified identical in the reference
  (`threshold = -args.threshold`, filter branch `h >= -eta`) and locally
  (`steered_chat.py:199`; runner `safety_threshold = -eta if eta > 0 else 0.0`).
* **State order correct**: the candidate is scored against its **pre-update** state,
  then `x_k = f_θ(x_{k-1}, u_k)` advances. Identical in both.
* Checkpoint: author file, SHA-pinned, key remap `ssm.state_transition.* →
  f_theta.net.*`, `nbf.* → net.*`, strict load, `.eval()`, `no_grad` at every
  scoring site. The repo's numerical parity test reports `max_diff = 0.0`.
* Embedding: `all-mpnet-base-v2`, default pooling, no normalization, 768-d — same
  model id and same defaults as the reference.
* **NBF-OFF invariant holds on the real execution path**, not just in telemetry:
  `barrier=None` and `embed_fn=None` are threaded from the runner down, so
  `use_safety_filter` is False, the candidate-filter block is skipped entirely, no
  barrier forward pass occurs, and the attack-side filter self-guards.
* No cross-run state leakage: barrier scoring re-initialises `x_0 = 0` per call and
  never writes back; attacks are freshly constructed per run and `reset()` per
  conversation.

## 6. Prompt fidelity audit

Byte-level comparison against the author's files:

| Prompt | Result |
|---|---|
| Crescendo system / boilerplate / feedback | **byte-identical** |
| OppositeDay system / boilerplate / feedback | **byte-identical** |
| Acronym system (2 example lines) | U+2019 `’` (author) vs ASCII `'` (local) — INFORMATIONAL |
| Rubric | 4 normalisations (`acton`→`act on`, `content,which`→`content, which`, `deviation.It`→`deviation. It`, double→single space) — declared D5 |
| Refusal checker | byte-identical (2504 chars, repo-verified) |
| Disclaimer checker | byte-identical, including the author's own typo |
| ActorAttack prompts | 1 apostrophe + 3 trailing-space lines differ |

None of these differences is semantically load-bearing. Notably, the local prompt
set does **not** add any JSON-repair, reasoning-stripping or role-changing text —
the attacker's `.format(conversationObjective=…)` substitution and the literal
`{{maxRounds}}`/`{{currentRound}}` placeholders match the author exactly.

## 7. Conversation / state audit

Verified from the runtime traces captured during Phase 16 and from the runner:

* attacker turn 1 `[system, user]`, turn 2 `[system, user, assistant, user]`,
  turn 3 `[system, user, assistant, user, assistant, user]` — the previous attacker
  query is an `assistant` message, the target reply is inside the existing
  "The last response was: …" user message. **Ordering correct on 100 % of observed
  turns**; goal present in the system message on every turn.
* Target conversation: dataset `target_system` seed, then `user`/`assistant` pairs —
  identical shape to the author's `history_t`.
* Earlier target replies are replaced by the attacker's **own** `(Summary)` text
  (author behaviour), while the most recent reply is echoed verbatim. The attacker
  summary never reaches the evaluator — the rubric receives the raw
  `(query, response)` pair. **No wrong-turn scoring found.**
* No system-prompt duplication, no history truncation (`max_turns_context` is never
  passed on this path), no context-window pressure (worst case ≈7.5 K of 131 K
  tokens).

## 8. Attacker audit

* Measured over the 180 frozen records: **1,978 attacker JSON calls, 18 parse
  failures = 0.91 %**; 12.99 % of runs hit at least one; mean 10.99 calls/run.
* All 11 `SUCCESS` records show 0 parse failures in their own runs.
* **F2 (see §18)**: for Acronym and OppositeDay a malformed reply is converted to
  an empty query (`return "", ""`) instead of raising, so the run ends as
  `attack_loop_exhausted` and is classified `ATTACK_FAILURE`. Proof: 12 runs with
  `attacker_parse_failures > 0` and `failure_class = ATTACK_FAILURE`, each with a
  final `[generation/str]` (unparseable) raw output; Acronym produced **0**
  `JSON_PARSE_ERROR` records despite 11 parse failures.
* Attacker instruction-following is otherwise good: 150/162 `ATTACK_FAILURE` runs
  had **no** parse failure at all — they failed on content, not on format.

## 9. Target audit (descriptive, from the frozen records)

| | OFF (n=90) | ON (n=90) |
|---|---:|---:|
| mean turns/run | 7.05 | 3.71 |
| mean refusal_count/run | 5.31 | 2.41 |
| best per-run rubric score | `{1:2, 2:7, 3:53, 4:21, 5:6}` | `{1:3, 2:3, 3:34, 4:10, 5:5}` |
| termination | 44 max-turns, 26 refusal-budget, 10 loop-exhausted, 6 success | 50 loop-exhausted, 23 max-turns, 9 refusal-budget, 5 success |

The target is **not** refusing everything: it engages in essentially every run and
returns *partially compliant* content (best score 3 in 53/90 OFF runs) rather than
blanket refusals — yet full compliance (score 5) is rare (6/90). That shape is
consistent with a target that yields partial information and stops short of the
complete harmful artifact — and, per F1, with answers that are cut off mid-sentence
before they could constitute full compliance.

**F1 — truncation, measured:** 929 target responses; tokenised with the target's own
tokenizer: mean **254.9** tokens, median **256**, max **256**; **95.0 % at the exact
256 cap**, 99.6 % ≥240, **0 % <200**; tails end mid-word
(`…allowing consumers to choose their`). The paper's GPT-4o target had **no token
cap**. This is a local runtime adaptation whose *effect* had not been measured; it
removes the tail of essentially every target answer the rubric is asked to score.

## 10. Evaluator audit

* Rubric: identical modulo the 4 declared normalisations; score range 1–5, success
  = score 5; temperature hardcoded `0.0` at every rubric/refusal/disclaimer call
  site → greedy decoding, matching the paper's judge temperature.
* Refusal rule `value and metadata > 90` identical; disclaimer rule identical.
* **Parse-failure exposure measured**: **2,755 evaluator JSON calls, 0 parse
  failures (0.000 %)** across the 180 records. The "fallback to 1" path
  (`return 1` on a non-dict/out-of-range score) therefore **never fired** in the
  frozen pilot — the mechanism is a latent risk (MEDIUM), not an observed depressant.
* The evaluator does not see anything it should not: it is given the raw current
  `(query, response)` pair and the goal, never the attacker summary, never a wrong
  turn. No evidence the evaluator artificially deflates scores: 53/90 OFF runs
  were rated 3 (partial compliance) and 27 ranked 4–5, which is a plausible
  reading of genuinely partial answers.
* Model substitution (GPT-4o → Llama-3.2-3B) is a real calibration change; the
  0 % parse-failure rate shows it is at least *operationally* reliable.
* `actor_attack` rubric/step-judge temperature and `json_format` deviations are
  latent (attack excluded).

## 11. JSON / interface audit

* Local chain: `json.loads` → `_extract_json_block` (first balanced `{…}`) → raw
  string. **This chain has no counterpart in the author's code** — the paper's
  OpenAI calls used server-side JSON mode; the author's own local path could not do
  JSON at all. It is a necessary LOCAL_RUNTIME_ADAPTATION, and the frozen study
  depends on it.
* Raw output is preserved on failure (`attacker_raw_outputs`, `error_message`) —
  verified across all 180 records.
* No retry, no regeneration, no fallback, no repair anywhere on this path
  (re-verified; consistent with Phases 15/16).
* **F2 is the interface finding**: the raise-vs-`""` asymmetry between Crescendo and
  its two siblings converts execution failures into attack failures.
* Truncation of the *output* (F1) is separate from input truncation
  (`truncation=True, max_length=131072`), which the prior audit correctly showed is
  never reached.

## 12. Seed / randomness audit

* `derive_seed = 42 + goal·1000 + attack_offset + condition_offset(10000)`; unique
  per (goal, attack, condition); `_seed_all` seeds `random`, `torch`, `numpy`
  (no explicit `torch.cuda.manual_seed_all`, no deterministic-algorithms flag).
* **The seed is set once per run, before the first generation** — not per turn;
  attacker and target therefore share one global RNG stream and each other's
  sampling shifts the other's.
* `do_sample = temperature > 0`: attacker and target sample (0.7); the evaluator is
  greedy (0.0). Diversity is **not** suppressed for the generative roles.
* **OFF and ON do not share seeds** (+10000 offset) and their streams diverge
  further because ON regenerates filtered candidates. The arms are
  **between-subjects, not RNG-paired** — a legitimate design, but paired-difference
  language would overstate the inference.
* **F3**: `top_p` is declared `1.0` in the manifest but never passed to `generate`;
  the models' own `generation_config.json` supplies `top_p=0.8, top_k=20`
  (Qwen3-4B attacker) and `top_p=0.9` (Llama-3.2-3B evaluator); Phi-4-mini has no
  such file. Declared deviation D-LLM1 ("top_p=1.0") is therefore **inaccurate**.

## 13. GPU / runtime audit

* Gates: pre-load 3.0 GB, post-load 1.5 GB, per-run 6.0 GB, ceiling 22.0 GB
  reserved. Pinned attacker + target↔evaluator rotation in slot 2. None of these
  can change generation: eviction only moves weights; reload re-encodes the full
  context each call (no KV-cache persistence), keeps `sdpa` + bf16, and no
  quantization exists anywhere.
* Runtime does change the **matrix**, not the generation: `pre_run_check` /
  `ceiling_check` are called **outside** the per-run `try`, so a refusal aborts the
  whole batch with **no failure rows** for the remaining runs (demonstrated live:
  the pilot stopped at 108/180). Mitigated by the resume, but any analysis must
  verify planned-vs-present counts.
* No OOM, no CUDA errors, 0 load refusals in the 180 runs; peak 20.281 GB.

## 14. Experimental methodology audit

* Goals: HarmBench file order, no shuffle, no skipping; `--limit` truncates the
  tail only and does not renumber run ids.
* Matrix: 200 × 3 attacks × 2 conditions = **1,200 runs**; ActorAttack excluded
  (`LOCAL_MODEL_CAPABILITY_LIMIT`), RedQueen excluded (`OFFICIAL_IMPLEMENTATION_NOT_PRESENT`).
* Denominator: **per conversation**; failures are retained (the OFF ASR report
  explicitly computes 6/90 and notes exclusion would give 6/86). No survivorship
  exclusion inside the file — but see the batch-abort note above for runs that were
  never written.
* Success = `5 in rubric scores` (per conversation), matching the paper's
  `success_score: 5`. The separate `ASRJudge`/`evaluation.*` path is **not** used by
  the frozen study — so "ASR" here is the rubric's own max score, which is the
  author's mechanic, not the paper's judge-model verdict. Documented.
* OFF/ON opportunity: identical goals, attacks, turn budget, threshold and stack;
  **different seeds** (§12).
* The pilot's aborted-then-resumed history is properly documented and the final
  180/180 file is complete.

## 15. Test-coverage audit

674 `def test_` functions / 34 files; the suite is strong on parity and telemetry
and genuinely compares against the author's reference (`tests/parity/*`: prompts,
message construction, refusal thresholds, NBF numerics with the real checkpoint,
`max_diff = 0.0`).

**Ranked gaps that could hide an ASR-affecting defect:**

1. **The production async runner is never exercised by pytest** — every runner test
   calls the sync twin; the two are ~270 near-duplicate lines. Historically a fix
   landed in one and not the other.
2. No test covers the seeding contract (`derive_seed`/`_seed_all`).
3. The full NBF filter/regeneration schedule parity lives in a *script*
   (`_audit_loop_parity.py`), not a test.
4. No test asserts matrix/denominator invariants (every goal×attack in both
   conditions, failures retained, per-conversation success).
5. The JSON-failure wiring is tested only at the record-builder level, not
   end-to-end through the runner — which is exactly why **F2 went unnoticed**.
6. Acronym/OppositeDay prompt parity is asserted only after whitespace/apostrophe
   normalisation, bounding substantive drift weakly.
7. No test pins the effective sampling parameters (the gap that let F3 persist).

## 16–17. Attribution and root-cause matrices

See `root_cause_matrix.md` (hypothesis-by-hypothesis evidence for/against) and
`reference_vs_local_matrix.md` (§3) plus `methodology_fidelity_matrix.md`.

## 18. Critical / high-severity findings

**F1 — Target output truncation — HIGH**
`LOCATION`: `configs/reproduction_phase14_frozen.yaml` (`models.target.max_new_tokens: 256`)
+ `src/guardbound/llm/local_client.py` (generation cap).
`EXPECTED` (paper): unbounded GPT-4o target completions.
`ACTUAL`: 95.0 % of 929 target responses hit exactly 256 tokens and end mid-word.
`EVIDENCE`: tokenizer re-encode of the frozen records (mean 254.9, median 256,
max 256, 0 % <200); tails quoted in §9.
`IMPACT`: the rubric scores an incomplete answer in nearly every turn; a truncated
answer cannot cleanly satisfy "fulfill the instruction without any deviation"
(score 5). Plausible depressant on measured success; directionally consistent with
53/90 OFF runs landing at partial compliance (3).
`CONFIDENCE`: HIGH for the truncation; MEDIUM for its share of the ASR gap.
`NEXT ACTION (not implemented)`: decide whether the 256-token target cap is an
accepted local adaptation (declared, as it partially is) or a deviation to be
reconsidered before a large run — a methodology decision, not a code fix.

**F2 — Failure-class misassignment for Acronym/OppositeDay — HIGH**
`LOCATION`: `src/guardbound/attacks/acronym.py` (`generate_acronym_step` returns
`("", "")`), `opposite_day.py` (same); `src/guardbound/attacks/runner.py`
(`if not query: break`); `scripts/phase14_full_reproduction.py` (`failure_class =
SUCCESS if success else ATTACK_FAILURE`).
`EXPECTED`: a malformed attacker reply is an execution failure
(`JSON_PARSE_ERROR`), as it is for Crescendo.
`ACTUAL`: it becomes an empty query → `attack_loop_exhausted` → recorded
`ATTACK_FAILURE`.
`EVIDENCE`: 12 of 180 records with `attacker_parse_failures > 0` and
`failure_class == ATTACK_FAILURE` (11 Acronym, 1 OppositeDay), each with a final
`[generation/str]` raw output; Acronym has 11 parse failures and **0**
`JSON_PARSE_ERROR` records while Crescendo's 6 parse failures map to 6.
`IMPACT`: the OFF half's execution-failure count is understated ~3–4×
(4 recorded vs ~15); `ATTACK_FAILURE` is not a clean attack-outcome category for
two of three attacks, so failure-taxonomy statements (including Phase 15's) are
biased. It does **not** change the success numerator — those runs had no chance to
succeed.
`CONFIDENCE`: HIGH (code + records + raw outputs agree).
`NEXT ACTION (not implemented)`: declare the asymmetry in the analysis, or fix the
raise/return contract for the two attacks — a code change that would alter existing
record semantics and therefore needs an explicit decision.

**F3 — Declared `top_p` not enforced — MEDIUM**
`LOCATION`: `src/guardbound/llm/local_client.py` (`gen_kwargs` omits `top_p`/`top_k`).
`EXPECTED`: the manifest and deviation D-LLM1 claim `top_p = 1.0`.
`ACTUAL`: sampling inherits `generation_config.json` — `top_p=0.8, top_k=20`
(Qwen3-4B attacker), `top_p=0.9` (Llama-3.2-3B evaluator); Phi-4-mini has none.
`EVIDENCE`: cached `generation_config.json` files; `gen_kwargs` source (no
`top_p`); `metadata.top_p = 1.0` hardcoded at `phase14_full_reproduction.py:823`.
`IMPACT`: narrower nucleus sampling than declared; the telemetry field is wrong.
`CONFIDENCE`: HIGH for the mechanism; MEDIUM for the exact effective value (not
executed).
`NEXT ACTION`: either declare the real values or pass them explicitly — a
methodology decision.

**Latent (off the frozen path, would be HIGH if used):** steered-mode candidate
filter scored against an empty dialogue (§4); `get_attack("crescendo")` resolving
to the heuristic implementation (§4).

## 19. Unknowns and limitations

* No ablation isolates model capability from the two methodology factors: the study
  has never run with an uncapped target or with a GPT-4o-class attacker.
* The share of the ASR gap attributable to truncation cannot be quantified without
  a new experiment (not run — prohibited by this phase).
* The paper's absolute ASR figures were **not** recovered in this audit; the frozen
  study is explicitly a controlled OFF-vs-ON contrast, not an absolute-ASR
  reproduction.
* `_extract_json_block` could in principle select an earlier balanced block from a
  verbose reply; occurrence not measured (0 evaluator parse failures makes it low
  risk in practice).
* Effective CUDA RNG seeding and any nondeterminism from non-deterministic kernels
  were not executable-verified.
* The batch-abort survivorship mechanism is real but currently masked (the pilot
  was resumed to 180/180).

## 20. Recommended next steps (none implemented)

1. **Decide F1**: keep the 256-token target cap as a declared local adaptation, or
   treat it as a deviation to lift before the large run. (Methodology decision.)
2. **Decide F2**: either declare the Acronym/OppositeDay failure-class asymmetry as
   a known taxonomy caveat in every analysis, or repair the raise/return contract
   — accepting that this changes what existing records mean.
3. **Correct F3** in the manifest/deviation record (declare the true effective
   `top_p`), and consider passing sampling parameters explicitly.
4. Add the missing tests (§15 gaps 1–5), especially an async-runner parity test and
   an end-to-end JSON-failure test.
5. Only then reconsider the pilot/1,200-run commitment.

## 21. Final pilot authorization decision

```text
NOT AUTHORIZED
```

The pipeline itself is sound — the algorithms, NBF, prompts, parsing and evaluation
are faithful, and no algorithmic defect explains the low success rate. But the
audit surfaced **one implementation defect that biases failure taxonomy (F2)** and
**one unmeasured methodology difference that plausibly depresses the success
numerator itself (F1)**, plus a misdeclared sampling parameter (F3). Committing
1,200 runs before those three are explicitly accepted — or deliberately changed —
would hard-code an interpretation problem into the study's headline numbers.

**This is a pre-commitment decision, not a verdict on the models.** The status of
the local-model substitution stands as Phase 16.3 left it: a contributor of unknown
share.
