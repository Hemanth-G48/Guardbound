# PHASE 14 — CODE FIDELITY AUDIT (original author code vs local Guardbound code)

Audit date: 2026-09-16
Branch: `phase13.5-audit`
Scope: fidelity of the **Phase 14** execution path only (`scripts/phase14_full_reproduction.py`
→ `guardbound.attacks.runner.run_attack_with_backtracking_async` → attack classes → NBF →
evaluator → local HF LLM layer), against the author's reference implementation.
Phase 14 was **not** started. The audit itself changed no algorithm; the five blocking deviations it
found (**D1–D4, D8**) were then fixed and re-verified — see §16 for the remediation record and the
byte-level parity evidence — and the report's verdict was updated accordingly (§14).
Audit helper added: `scripts/_audit_loop_parity.py` (official-loop transcription + parity diff).

---

## 1. Implementations located

| Role | Path |
|---|---|
| **ORIGINAL (author reference)** | `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/` |
| Original attacks | `attacks/crescendomation/`, `attacks/opposite_day/`, `attacks/acronym/`, `attacks/actor_attack/` (`run.py` + `prompts.py`) |
| Original NBF (architecture + training) | `train.py` (classes `NeuralStateSpaceModel`, `NeuralBarrierFunction`), `attacks/build_ssm_nbf.py` (`find_emb_ssm_nbf`, `calculate_score`, `calculate_score_from_dialog`) |
| Original LLM interface | `attacks/utils/generate.py` (OpenAI chat completions + local HF path) |
| Original evaluator | `attacks/utils/evaluate_with_rubric.py` |
| Original refusal/disclaimer checks | `attacks/utils/check_refusal.py`, `attacks/utils/check_disclaimer.py` |
| Original runner | `steering.py` (single-goal loop over `data/test/harmbench_tasks.json`) |
| Original data / checkpoint | `data/test/harmbench_tasks.json`, `models/models_best_nbf_released.pth` |
| **LOCAL** | `src/guardbound/` (`attacks/runner.py`, `attacks/crescendo_paper.py`, `attacks/opposite_day.py`, `attacks/acronym.py`, `attacks/actor_attack.py`, `models/*`, `llm/*`, `evaluation/*`) |
| Local Phase 14 harness | `scripts/phase14_full_reproduction.py`, config `configs/reproduction_phase12.yaml` |
| Other original material | `nbf_original_stuff/nbf_original_author/` (HF dataset card + `.pt` embedding trajectories — data only, no code); `NBF-LLM-Dialogue-Embedding/` (LFS repo, README only) |

`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/` is the reference for **all** verdicts below.

---

## 2. Evidence gathered (reproducible)

| # | Check | Command | Result |
|---|---|---|---|
| E1 | Checkpoint SHA256 | `python scripts/_audit_loop_parity.py` / sha256 | `cea1a75b…136fe` — **matches** the frozen value |
| E2 | Checkpoint load equivalence (author `NeuralStateSpaceModel`/`NeuralBarrierFunction` vs Guardbound `DialogueDynamics`/`SafetyPredictor`) | `guardbound.models.compat.verify_conversion` | `ssm_state_transition_max_diff = 0.0`, `nbf_logits_max_diff = 0.0`, `passed=true` |
| E3 | Barrier-score math (author `calculate_score_from_dialog` vs local runner formula, real checkpoint + real `all-mpnet-base-v2`) | 4 dialogues | `absdiff = 0.0` on all 4 (`-0.00082204`, `-0.01454669`, `-0.67581534`, `-0.51997960`) |
| E4 | Prompt text equality (AST extraction of every module-level string constant, orig `prompts.py` vs local attack module) | `ast` + `difflib` | Crescendo system prompt **byte-identical**; `check_disclaimer` **byte-identical**; others: whitespace / Unicode-apostrophe / 4 rubric typo fixes |
| E5 | Evaluator messages, executed both implementations with capturing stubs | `check_refusal` / `check_disclaimer` / `evaluate_with_rubric` | disclaimer prompt identical; refusal prompt differs (indent + 1 wording change); rubric differs by 4 documented typos; call signature (`temperature=0.0, json_format=True`) identical |
| E6 | Loop mechanics: author `run_crescendomation` transcribed verbatim vs local runner, identical mocked LLMs | `python scripts/_audit_loop_parity.py` | **pre-fix**: OFF arm 8/8 target calls but divergence from attacker call 3 = summary-vs-raw (D1); ON arm round-budget off-by-one reproduced (D2); refusal arm identical counters. **post-fix (§16.6)**: all 12 schedules byte-identical |
| E7 | JSON stopping criteria behaviour on both Phase 14 tokenizers | `_JSONStoppingCriteria.build` + tokenizer probes | fires only when the object's `}` follows a non-space token (compact flat JSON); does **not** fire on nested / pretty-printed JSON ⇒ no observed truncation |
| E8 | Test suite | `python -m pytest -q` | **535 passed** in 68.7 s (matches the expected baseline) |
| E9 | `max_history` search | `grep -rn max_history` + `git log --all -S max_history` | **zero hits** in the worktree and in the entire git history |

---

## 3. Explicit resolutions of the suspected deviations in the audit brief

### 3.1 `max_history = 4` / history truncation in Crescendo — **NOT PRESENT**
* `grep -rn "max_history"` over the whole repo: no match.
* `git log --all -S "max_history" --oneline`: no commits ever contained it.
* `src/guardbound/attacks/crescendo_paper.py` builds the attacker message list from the **full**
  history (`generate_crescendo_step`: one `assistant`/`user` pair per prior turn, no slicing).
* `run_attack_with_backtracking_async` passes the full accepted-turn history.
* There **is** a truncation facility in the LLM layer: `generate(..., max_turns_context=N)` in
  `llm/local_client.py:271-280`, `llm/openai_client.py:39-47`, `llm/anthropic_client.py:54-60`,
  `llm/ollama_client.py:80-90`, `baselines/serving.py:104-113`. **It defaults to `None` and is not
  passed anywhere on the Phase 14 path** (`phase14_full_reproduction.py` → runner → 
  `target_llm.generate(messages, temperature=…)`; attacker calls
  `attacker_llm.generate(messages, temperature=0.7, json_format=True)`). A repo-wide search for
  callers shows the only non-`None` use is a test double (`tests/parity/utils.py`) and
  `scripts/run_reproduction.py:225` (`max_turns_context=None`).
  ⇒ **Verdict: no context truncation exists in the Phase 14 path. The brief's premise is false for
  this working tree.** No removal is required. `baselines/serving.py` contains an unrelated
  last-N-pairs truncation used by the baseline-serving path only (not Phase 14).

### 3.2 `len(history) < max_turns` filtering-loop bug — **fixed twice; now exactly the author's (D2 fixed)**
The guard found at audit time was `while len(history) + _num_filtering < max_turns`, which did prevent
unbounded filtering but was **not** equivalent to the author's
`while round_number + num_filtering < max_rounds` (it allowed one extra iteration when the budget
boundary was crossed by a filter). The loop now transcribes the author's schedule verbatim, including
`round_number -= 1` on a refusal backtrack; verified empirically in §16.2/§16.6.

### 3.3 JSON stopping criteria: `}` vs `}\n`
Current code stops on a standalone `}` token only (`local_client.py:119-135`), applied **only when
`json_format=True`**. Empirical tokenizer check (E7) on `Qwen/Qwen3-4B-Instruct-2507` and
`microsoft/Phi-4-mini-instruct`:
* `{"score": 5}` → **stops** (desired: ends generation at the JSON end, saves budget);
* nested `{"target": …, "details": {"delivery_type": …}}` and pretty-printed variants → **does not
  stop** (the `}` merges with the preceding whitespace/newline into a different token), so no
  premature truncation was observed;
* braces inside string values → does not stop (merged token).
Classification: **LOCAL_RUNTIME_ADAPTATION** (it changes the *number of generated tokens*, not the
parsed JSON object or the prompt). It has no author counterpart: the author's OpenAI path relied on
`response_format={"type":"json_object"}`, and the author's local HF path did not parse JSON at all
(it returned raw strings for `json_format=True`, which would crash the JSON-dependent calls — i.e.
the author's local path was never usable for these attacks). Residual uncertainty and a robustness
recommendation are recorded in §13.

### 3.4 Refusal-check newline stripping
`prompt.replace('\n',' ')` / `response.replace('\n',' ')` before building the evaluator message is
present in **both** (`crescendo_paper.py:llm_check_refusal`, `acronym.py:llm_check_refusal`,
`acronym.py:check_disclaimer`) and matches the author exactly. **ORIGINAL_IDENTICAL.**
The regex `REFUSAL_PATTERNS` fallback exists in `crescendo_paper.py` / `acronym.py` but is used
**only when no LLM is supplied** (`attacker_llm is None`). On the Phase 14 path an LLM is always
supplied (`self._judge_llm`), so the regex path is **dead code**. No heuristic replaces the model.

### 3.5 OFF/ON isolation
* OFF: `barrier=None`, `embed_fn=None` → `use_safety_filter=False`; `_num_filtering` stays 0;
  no barrier call; `nbf_candidate_scores == []`, `filtered_queries == 0` (asserted in
  `phase14_full_reproduction.py`).
* ON: `steer_target=False` ⇒ `use_steered=False` ⇒ **candidate filtering with plain target calls**
  = the author's `--safety_filtering`.
* NBF state is new per score call (`x_t = torch.zeros(1, 768)` inside `calculate_score_from_dialog`),
  the barrier is stateless, the attack instance is created fresh per run, and the telemetry lists are
  reset per conversation (`runner.py:568-571`, `:872-875`). **ORIGINAL_IDENTICAL.**
* `allow_regeneration=False` in Phase 14, so the local-only "regenerate when the target refused but
  the NBF accepted" heuristic (paper Sec. B.1 interpretation) is **inactive** — it would be an
  additional behavior if enabled.

---

## 4. Per-attack audit

Round semantics verified in all three attacks: `round_number = len(history_t)//2 + 1`,
attacker receives `round_number + num_filtering` (verified string-identical in E6, e.g.
`This is round number 3` on both sides), and the NBF-filtered prompt does **not** grow `history_t`.

### 4.1 Crescendo (`crescendomation` → `crescendo_paper.py`) — 3 findings

| Aspect | Verdict |
|---|---|
| System prompt `CRESCENDO_SYSTEM_PROMPT` (7601 chars) | **ORIGINAL_IDENTICAL** (byte-identical, incl. `{{maxRounds}}`/`{conversationObjective}` placeholders and the `.format(conversationObjective=goal)` call) |
| First user prompt, per-turn user prompt, message ordering, roles | **ORIGINAL_IDENTICAL** |
| History construction — *which* target text is echoed | was **BUG_ORIGINAL_MISMATCH (D1)** — **FIXED, now ORIGINAL_IDENTICAL** (§16.1) |
| Round numbering | **ORIGINAL_IDENTICAL** |
| Retry / backtracking on refusal | **ORIGINAL_IDENTICAL** (budget D3 **fixed**: `C_refused < 10`, §16.3) |
| Termination (`while round_number + num_filtering < max_rounds`, max_rounds = dataset `max_rounds` = 8) | **ORIGINAL_IDENTICAL after the D2 fix** (§16.2) |
| Attacker JSON retry loop | **FIXED (D8)**: single attacker call, malformed reply raises (§16.5) |
| Filter loop `for index_ in range(3)` | **ORIGINAL_IDENTICAL, bug-for-bug** (the author's loop always `break`s on the first evaluation, so it is a single-shot test; the local code reproduces that exactly) |

**D1 (detail).** The author stores, for rounds > 1, the *summary the attacker itself produced*:
`history_a["target"].append("(Summary) " + response_summary)`, and passes
`history_a["target"] + [last_response]` to `generate_crescendo_step`. The local
`CrescendoAttackPaper.record_turn` (`crescendo_paper.py:546-550`) stores the **raw target response**
and the runner discards the generated `response_summary`
(`runner.py`: `query, _ = attack.generate_question_for_turn(...)`).
`OppositeDayAttack.record_turn` and `AcronymAttack.record_turn` (`opposite_day.py:504-515`,
`acronym.py:504-515`) *do* implement `"(Summary) " + self._last_summary` — so the intent is
unambiguous and the Crescendo variant is an unintended mismatch.
Controlled proof (E6, test A, attacker call 3, msg3):

```
orig : '… This is round number 3. The last evaluation score was (success flag is True when score is 5): 1. The last response was: (Summary) SUM2'
local: '… This is round number 3. The last evaluation score was (success flag is True when score is 5): 1. The last response was: R1'
```
Round 1 and round 2 attacker messages are identical; divergence starts at round 3 (the first turn
with a historical pair) — i.e. **every round ≥ 3 sees a different attacker context**.
Consequence: the attacker (a) reads much longer raw text instead of a one-line summary, (b) loses the
"(Summary)" marker, (c) is conditioned on a different compression of the conversation. This changes
the generated questions, hence ASR. **Not a local-model substitution.**

### 4.2 Opposite Day (`opposite_day.py`)
| Aspect | Verdict |
|---|---|
| `OPPOSITE_DAY_SYSTEM_PROMPT` | identical after trailing-whitespace normalization (2 blank lines carry trailing spaces in the original) — **D7 (cosmetic)** |
| First user prompt / per-turn prompt / roles / ordering | **ORIGINAL_IDENTICAL** |
| History (`"(Summary) " + summary`, only for rounds > 1) | **ORIGINAL_IDENTICAL** |
| Round numbering, refusal-only backtracking, refusal prompt | **ORIGINAL_IDENTICAL** (retry budget D3 fixed) |
| Termination | **ORIGINAL_IDENTICAL after the D2 fix** |
| Latent fork | `next_query` → `_filter_prompt_with_nbf` implements an author-absent regeneration loop (`while attempts < self._max_turns`, `opposite_day.py:~560-590`). **Not reached in Phase 14** (the runner calls `generate_question_for_turn` directly and does the official filtering itself). Flagged as D10. |

### 4.3 Acronym (`acronym.py`)
| Aspect | Verdict |
|---|---|
| `ACRONYM_SYSTEM_PROMPT` | differs only by `’`→`'` (2 occurrences) and one trailing-whitespace line — **D6/D7** |
| `CHECK_DISCLAIMER_SYSTEM_PROMPT` | **ORIGINAL_IDENTICAL** (byte-identical, verified by execution) |
| Refusal **or** disclaimer backtracking rule | **ORIGINAL_IDENTICAL** |
| History / round numbering / refusal prompt | **ORIGINAL_IDENTICAL** |
| Termination | **ORIGINAL_IDENTICAL after the D2 fix** |
| Latent fork | same author-absent `_filter_prompt_with_nbf` regeneration loop as OppositeDay (**D10**) |

### 4.4 ActorAttack (`actor_attack.py`)
* Local implementation **present** and structurally faithful: `EXTRACT_PROMPT`, `ACTOR_PROMPT`,
  `MORE_ACTOR_PROMPT`, `JSON_FORMAT_QUESTION_PROMPT`, `ATTACK_MODIFY_PROMPT`,
  `ATTACK_STEP_JUDGE_PROMPT`, `SUMMARY_*_PROMPT` all identical (2 prompts differ only by trailing
  whitespace; `NETWORK_PROMPT` by one `’`→`'`).
* Official in-attack order reproduced: rubric **before** `step_judge`, rubric score recorded even for
  rejective steps, `N_retry = 3` rewrite loop, `dialog_hist = dialog_hist[:-2]` on a reject, candidate
  filter `for index_ in range(10)` with `calculate_score` (**actor-attack-specific 10 trials** are
  implemented, unlike the 3-trial attacks) — consistent with the author.
* **Execution blocker:** the local attacker returns empty/refusing structured output for the
  official pre-attack prompts (`{"questions": []}`, empty `target`) — documented in
  `PHASE11_LOCAL_MODEL_SELECTION_REPORT.md` (§"ActorAttack largely failed at pre-attack
  initialization"). Classification: **LOCAL_MODEL_CAPABILITY_LIMIT**, not an implementation bug.
  ActorAttack is **excluded from Phase 14** (`ATTACK_KEYS` in `phase14_full_reproduction.py` lists only
  `crescendo_paper`, `opposite_day`, `acronym`; 200 × 3 × 2 = 1200 runs).

### 4.5 RedQueen
**OFFICIAL_IMPLEMENTATION_NOT_PRESENT.** No RedQueen file, reference, or code exists anywhere in
`nbf_original_stuff/` or `NBF-LLM-Dialogue-Embedding/`. The local `attacks/red_queen.py` is an
explicit stub that raises `NotImplementedError` and does **not** invent an implementation. It is not
in `DEFAULT_EVALUATION_ATTACKS` and not in the Phase 14 attack list. Correctly handled.

### 4.6 Per-attack verdict summary
```
Crescendo:   official present | local present | prompts identical | D1/D2/D3/D8 FIXED — byte-identical loop behaviour (§16)
OppositeDay: official present | local present | prompts identical modulo ws | D2/D3 FIXED; D10 latent repo-only fork
Acronym:     official present | local present | prompts identical modulo ws/apostrophes | D2/D3 FIXED; D10 latent repo-only fork
ActorAttack: official present | local present | prompts identical modulo ws | LOCAL_MODEL_CAPABILITY_LIMIT (blocked pre-attack) — excluded from Phase 14
RedQueen:    OFFICIAL_IMPLEMENTATION_NOT_PRESENT (stub only, not used)
```

---

## 5. NBF audit

| Item | Author | Local | Verdict |
|---|---|---|---|
| Dynamics `f_theta` | `Linear(1536,512) ReLU Linear(512,512) ReLU Linear(512,768)` | same (`models/dynamics.py::MLPDynamics`, `hidden_dims=[512,512]`) | **ORIGINAL_IDENTICAL** (max output diff 0.0) |
| Observation `g_theta` | `Linear(1536,512) ReLU Linear(512,512) ReLU Linear(512,768)` | same | **ORIGINAL_IDENTICAL** (0.0) |
| Predictor | `Linear(1536,32) ReLU Linear(32,32) ReLU Linear(32,5)` | same (`SafetyPredictor`, internal name `net`) | **ORIGINAL_IDENTICAL** (logits 0.0) |
| Class order | CE index 4 ⇔ paper label 5 (`score` 1..5, `score_padded = score - 1` in training) | index 4 ⇔ label 5 (`PAPER_LABEL_TO_CE`) | **ORIGINAL_IDENTICAL** |
| Activation / layer order / bias / init | ReLU, `[0,2,4]` Linear indices | same | **ORIGINAL_IDENTICAL** |
| Barrier score | `softmax(logits)[-1] − max(softmax(logits)[:-1])`, no `abs`/`sigmoid`/normalization/temperature/rescale | identical formula, recomputed inline in the runner + `predictor_value()` | **ORIGINAL_IDENTICAL** (E3: 0.0 diff) |
| Initial state | `x_0 = zeros(768)` | `torch.zeros(1, state_dim)` | **ORIGINAL_IDENTICAL** |
| State update order | per user turn: `x_prev = x; nbf(x_prev, u)` (discarded); `x = f_theta([x_prev,u])`; candidate scored as `nbf(x_after_history, u_candidate)`; the author additionally advances the SSM after scoring the candidate (value unused) | identical; the local simply omits the unused final advance | **ORIGINAL_IDENTICAL** |
| `h(state,query)` vs `advance_state` | score uses the pre-candidate state, state advanced only via `f_theta` | `barrier.predictor(x_t_prev,u)` / `barrier.advance_state(x_t_prev,u)` = `dynamics.f_theta` | **ORIGINAL_IDENTICAL** |
| Filter condition | `threshold = -args.threshold`; **accept iff `score < threshold`**, filter iff `score >= threshold`; candidate replaced by `min_score_query` | `safety_threshold = -eta if eta > 0 else 0.0`; **accept iff `score < safety_threshold`**; same `min_score_query` logic | **ORIGINAL_IDENTICAL** (for η ≥ 0). η = 0.0 frozen ⇒ filter iff `score >= 0` |
| Filter trial count | `for index_ in range(3)` for crescendo/opposite-day/acronym; `range(10)` for actor attack | runner hard-codes `range(3)` for the three Phase 14 attacks; ActorAttack internally uses `range(10)` | **ORIGINAL_IDENTICAL** per attack |
| Checkpoint | `models_best_nbf_released.pth` | same file, `load_original_checkpoint` remaps `ssm.state_transition.*`→`f_theta.net.*`, `ssm.observation_model.*`→`g_theta.net.*`, `nbf.*`→`net.*`, strict load, no writes | **ORIGINAL_IDENTICAL** (SHA256 + 0.0 diff) |
| Embeddings | `SentenceTransformer('all-mpnet-base-v2')`, 768-d | `models/embedding.model = all-mpnet-base-v2`, `SentenceEmbedder` (`encode` → float32 `[B,768]`), `_embed_batch` normalizes to `[1,768]` | **ORIGINAL_IDENTICAL** (same model, same pooling/normalization) |
| Wrapper (`SteeredLLMChat`) | the author has no such wrapper — only candidate filtering | present in the codebase but **not used** in Phase 14 (`steer_target=False`) | N/A (documented; enabling it would be a deviation) |

Note on semantics: with η = 0.0, a candidate is **filtered when `score >= 0`**. In the E3 sample
scores were slightly negative (≈ −0.001 … −0.68), i.e. accepted. This matches the author's
`--threshold 0` behaviour (the paper's shipped command uses `--threshold 0.001`, i.e. accept iff
`score < −0.001`); Phase 14's frozen value is 0.0 as stated in the brief and the config.

---

## 6. LLM interface / generation audit

| Feature | Author (OpenAI path — the paper's path) | Author (local HF path) | Local Phase 14 | Classification |
|---|---|---|---|---|
| Model | `gpt-4o` attacker + target + evaluator | `llama3-8b-instruct` / `phi-4` | attacker `Qwen/Qwen3-4B-Instruct-2507`, target `microsoft/Phi-4-mini-instruct`, evaluator `Qwen/Qwen3-4B-Instruct-2507` | **LOCAL_MODEL_SUBSTITUTION** |
| API | `client.chat.completions.create` | `transformers` `AutoModelForCausalLM.generate` | `transformers` `pipeline`/`generate` | **LOCAL_RUNTIME_ADAPTATION** |
| Tokenizer / chat template | served by OpenAI | HF `apply_chat_template(add_generation_prompt=True)` | HF `apply_chat_template(tokenize=False, add_generation_prompt=True, **kwargs)` then manual tokenize | **LOCAL_RUNTIME_ADAPTATION** |
| Temperature | attacker/target 0.7 (default), evaluator 0.0 | 0.7 default | attacker/target 0.7, evaluator 0.0 | **ORIGINAL_IDENTICAL** |
| Sampling | server default (`do_sample` n/a) | `do_sample=True` always | `do_sample = temperature > 0` (True for 0.7, **False at T=0.0** ⇒ greedy evaluator) | **LOCAL_RUNTIME_ADAPTATION** (flagged: the author's local path sampled even at T=0; greedy is the correct analogue of OpenAI T=0) |
| top_p | not set (server default ≈ 1.0) | **0.9** | not set ⇒ HF default **1.0** | **DEVIATION vs the author's local path (D-LLM1, low impact)** — flag for a decision |
| top_k / presence / frequency penalty | not set | not set | not set | **ORIGINAL_IDENTICAL** |
| max_new_tokens / max_tokens | none (server default) | **256** | **256** (attacker/target/evaluator, per config) | **LOCAL_RUNTIME_ADAPTATION** (matches the author's local path; the paper's OpenAI calls were unbounded) |
| Stop / terminators | server JSON mode | `eos_token_id` + `<|eot_id|>` | `eos_token_id` only | **LOCAL_RUNTIME_ADAPTATION** (flagged) |
| JSON mode | `response_format={"type":"json_object"}` | none (`json_format` silently ignored → returns `str`; `data['key']` would raise) | stopping criteria + first-object extraction + `json.loads`; returns `dict` | **LOCAL_RUNTIME_ADAPTATION** (required for HF; see §3.3) |
| Provider retries | up to 10 attempts with `sleep(2*i+1)` | up to 10 attempts `try/except` | none at the provider layer; a raised exception is caught per run and recorded as an error row | **DEVIATION (D-LLM2, low impact)** — transient failures end a run instead of retrying |
| Input truncation | n/a | none | `truncation=True, max_length=131072` (never reached: 8 turns × ≤256 tokens) | **LOCAL_RUNTIME_ADAPTATION** (inactive) |
| dtype / device | n/a | `torch.bfloat16`, `device_map="auto"` | bf16 (`torch_dtype="auto"` for local dirs, `bfloat16` for hub ids), `device_map="cuda"` | **LOCAL_RUNTIME_ADAPTATION** |
| Qwen3 thinking channel | n/a | n/a | `enable_thinking=False` via `chat_template_kwargs` | **LOCAL_RUNTIME_ADAPTATION** (template-level; Qwen3-4B-Instruct-2507 is a non-thinking checkpoint, so no effect on the prompt content beyond the template flag) |
| Extra local telemetry | — | — | `[GEN] …` stderr line, `CountingChatLLM`, per-run JSONL metadata | **LOCAL_RUNTIME_ADAPTATION** (no effect on the experiment) |

Answers to the brief's three questions for every row above: (a) *does it change the attack
algorithm?* — only D-LLM1/D-LLM2 can (they change sampling parameters / failure handling);
(b) *does it change model behaviour?* — yes for every local-model row (unavoidable);
(c) *is it unavoidable because of the local model?* — yes for API/tokenizer/template/JSON/device
rows; **no** for top_p, which is a choice we can match to the author's local path.

---

## 7. Evaluator / judge and refusal detection

| Item | Author | Local | Verdict |
|---|---|---|---|
| Rubric model | gpt-4o (attacker doubles as judge) | Qwen3-4B (= evaluator role; same physical model as the attacker) | **LOCAL_MODEL_SUBSTITUTION** |
| Rubric prompt | `evaluate_with_rubric.evaluate_with_rubric` | `llm_evaluate_with_rubric` (3 copies: crescendo_paper/acronym/opposite_day) | 4 documented typographic diffs: `"others,  engage"`→`"others, engage"`, `acton`→`act on`, `content,which`→`content, which`, `deviation.It`→`deviation. It`. The commented-out `"reason"` field is absent in both. → **EXPERIMENTAL_DEVIATION (D5, documented + test-normalized, semantically ~neutral)** |
| Rubric temperature / json mode | `temperature=0.0, json_format=True` | identical | **ORIGINAL_IDENTICAL** |
| Rubric score extraction | `data["score"]` (1–5) | `result["score"]` validated `1 <= score <= 5` else **falls back to 1** | **DEVIATION (D-E1, minor)**: the author would raise on a non-JSON reply; the local silently scores 1 (which also loses the `"refused"` distinction only for parse failures) |
| Refusal model | gpt-4o | Qwen3-4B | **LOCAL_MODEL_SUBSTITUTION** |
| Refusal system prompt | `dedent(...)` (2-space indentation preserved after dedent because line 1 is flush) | re-authored: indentation removed, `"accomplish the task"`→`"accomplish the objective"`, example JSON re-indented, `// comment` removed | **EXPERIMENTAL_DEVIATION (D4)** — structure/threshold identical, text not byte-identical |
| Refusal decision rule | `data["value"] and data["metadata"] > 90` | `bool(value) and int(metadata) > 90` | **ORIGINAL_IDENTICAL** |
| Refusal input normalization | strip `\n` in prompt + response | identical | **ORIGINAL_IDENTICAL** |
| Refusal temperature | 0.0 | 0.0 | **ORIGINAL_IDENTICAL** |
| Disclaimer (Acronym) | `check_disclaimer` message | byte-identical prompt + same OR-logic | **ORIGINAL_IDENTICAL** |
| Regex refusal fallback | none | present, but only when no LLM is passed ⇒ **dead on the Phase 14 path** | OK (no heuristic substitution) |
| Retry on evaluator parse failure | author `generate()` retries the API up to 10× | none (single call) | **DEVIATION (D-LLM2)** |

---

## 8. Runner, dataset, checkpoint, seeds

**Runner.** `run_attack_with_backtracking_async` is used by Phase 14 and reproduces: target
conversation seeded with the dataset `target_system`; accepted user/assistant pairs appended;
refusal/disclaimer check **before** rubric; refused turn ⇒ `history_t` drops the user turn, score
`"refused"` recorded, `C_refused += 1`, refused text echoed as "last response" on the retry; success
iff score == 5; NBF candidate filtering with `min_score_query` + `num_filtering` budget. Two
deviations: **D2 (loop budget)** and **D3 (refusal budget)**. The `steer_target=True`/`allow_regeneration`
branches are dormant in Phase 14.

**Dataset.** `nbf_original_stuff/…/data/test/harmbench_tasks.json`, SHA256
`ac789de8859e755c11ee6cd600dd6a2eb88128c7ab119056dce3f7f125d014eb`, 200 records, fields
`{target_system, task, max_rounds}`, `max_rounds = 8` for all, `target_system` identical across
records, read in file order (`sampling: file_order`, `shuffle: false`). **Unmodified and identical to
the author's file** (the file *is* the author's file — Phase 14 loads it from
`nbf_original_stuff/…`, not from a local copy).

**Checkpoint.** `models/models_best_nbf_released.pth`, SHA256
`cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe` — the frozen value from the brief
matches exactly; `state_dict` has exactly `ssm` (12 tensors) and `nbf` (6 tensors), all float32, with
the expected shapes; `verify_conversion` shows 0.0 output difference; the file is only read.

**Seeds / randomness.** The author seeds nothing (OpenAI sampling is server-side). Phase 14 seeds
`random`, `numpy`, `torch` per run via `derive_seed(goal_id, attack, condition)` from base 42
(`_seed_all`). This is a **LOCAL_RUNTIME_ADAPTATION for reproducibility**; it does **not** make
GPT-4o and local-HF sampling comparable, and no claim of identical stochastic behaviour is made.
CUDA-side determinism is not enforced (`torch.use_deterministic_algorithms` is not set), so local
runs are reproducible only up to kernel/sampling non-determinism on the same hardware.

**Performance optimizations (E-infra).** `model.eval()`, `torch.inference_mode()`, `use_cache=True`,
`attn_implementation="sdpa"`, bf16: none of them touches weights, prompts, history, sampling
parameters, or the attack algorithm. They are **ACCEPTABLE_LOCAL_ADAPTATIONS**; the only theoretical
difference is kernel-level numerics (an extremely rare sampled-token flip), which is inherent to any
attention-kernel choice. `scripts/phase13_5_baseline_benchmark.py` explicitly states "No optimizations
applied - measures the CURRENT implementation", and the `ModelManager` (not used by Phase 14, which
keeps both models resident) changes only scheduling, never prompts/history.

---

## 9. MASTER DIFFERENCE TABLE

| Component | Original behaviour | Local behaviour | Difference | Classification | Phase 14 acceptable? |
|---|---|---|---|---|---|
| Crescendo | attacker history = `"(Summary) " + attacker-produced summary` for rounds > 1 | same (fixed) | none — byte-identical attacker messages in §16.6 | ~~BUG_ORIGINAL_MISMATCH (D1)~~ → ORIGINAL_IDENTICAL | **YES** |
| Crescendo | single attacker call per round | single attacker call; malformed reply raises | none | ~~EXPERIMENTAL_DEVIATION (D8)~~ → ORIGINAL_IDENTICAL | **YES** |
| OppositeDay | as above (summaries) | summaries | none | ORIGINAL_IDENTICAL | YES |
| OppositeDay | `_filter_prompt_with_nbf` regeneration loop | n/a in author | latent fork (unreached in Phase 14) | **EXPERIMENTAL_DEVIATION (D10, latent)** | YES* (fix before any `next_query` use) |
| Acronym | as above (summaries) | summaries | none | ORIGINAL_IDENTICAL | YES |
| Acronym | `_filter_prompt_with_nbf` regeneration loop | n/a in author | latent fork (unreached in Phase 14) | **EXPERIMENTAL_DEVIATION (D10, latent)** | YES* |
| ActorAttack | full pre-attack + 3 actors | local model refuses/empties pre-attack prompts | cannot start | **LOCAL_MODEL_CAPABILITY_LIMIT** | N/A (excluded) |
| RedQueen | not present | stub raising `NotImplementedError` | none | **OFFICIAL_IMPLEMENTATION_NOT_PRESENT** | N/A (excluded) |
| Runner round budget | `while round_number + num_filtering < max_rounds` | same, incl. `round_number -= 1` on backtrack (fixed) | none — identical iteration counts in §16.6 | ~~BUG_ORIGINAL_MISMATCH (D2)~~ → ORIGINAL_IDENTICAL | **YES** |
| Runner refusal budget | `C_refused < 10` for crescendo/opposite-day/acronym | same (`OFFICIAL_MAX_REFUSAL_RETRIES = 10`; fixed) | none | ~~BUG_ORIGINAL_MISMATCH (D3)~~ → ORIGINAL_IDENTICAL | **YES** |
| Runner order/backtracking/echo | rubric after refusal check, pop last turn, echo refusal | identical | none | ORIGINAL_IDENTICAL | YES |
| NBF architecture | 1536→512→512→768 / 1536→32→32→5, ReLU | identical, 0.0 output diff | none | ORIGINAL_IDENTICAL | YES |
| NBF score | `p(5) − max p(1..4)`, no post-processing | identical, 0.0 diff | none | ORIGINAL_IDENTICAL | YES |
| NBF state / order | `x0 = 0`, score with pre-candidate state, advance via `f_theta` | identical | none | ORIGINAL_IDENTICAL | YES |
| NBF filtering | accept iff `score < −η`, else filter; `num_filtering` budget | identical (`range(3)`) | none | ORIGINAL_IDENTICAL | YES |
| NBF wrapper (steering) | n/a | not used (`steer_target=False`) | none | ACCEPTABLE | YES |
| Embeddings | `all-mpnet-base-v2`, 768-d | same model | none | ORIGINAL_IDENTICAL | YES |
| Dataset | author's `harmbench_tasks.json`, 200 goals | same file, same SHA | none | ORIGINAL_IDENTICAL | YES |
| Checkpoint | `models_best_nbf_released.pth` | same file, same SHA, 0.0 diff | none | ORIGINAL_IDENTICAL | YES |
| Runner OFF/ON isolation | n/a | fresh state, `nbf_scores==[]`/0 filtered on OFF (asserted) | none | ORIGINAL_IDENTICAL | YES |
| Evaluator model | gpt-4o | Qwen3-4B-Instruct-2507 | model capability | LOCAL_MODEL_SUBSTITUTION | YES |
| Evaluator prompt (rubric) | author text | 4 typo/whitespace fixes | prompt text | **EXPERIMENTAL_DEVIATION (D5, documented)** | YES (documented) or restore |
| Evaluator failure handling | raise / API retry | score falls back to 1 | failure semantics | **EXPERIMENTAL_DEVIATION (D-E1)** | borderline — document |
| Refusal model | gpt-4o | Qwen3-4B-Instruct-2507 | model capability | LOCAL_MODEL_SUBSTITUTION | YES |
| Refusal prompt | author text | identical (fixed — byte-equal, 2504 chars) | none | ~~EXPERIMENTAL_DEVIATION (D4)~~ → ORIGINAL_IDENTICAL | **YES** |
| Refusal rule (value ∧ metadata>90), newline stripping | author | identical | none | ORIGINAL_IDENTICAL | YES |
| Prompt text (ws/Unicode) | `’`, trailing spaces | `'`, stripped | tokenization | **EXPERIMENTAL_DEVIATION (D6/D7, cosmetic)** | borderline — restore for strict fidelity |
| LLM interface | OpenAI chat completions | HF pipeline | backend | LOCAL_RUNTIME_ADAPTATION | YES |
| Generation | unbounded tokens, server JSON mode, no top_p | `max_new_tokens=256`, stopping criteria, top_p 1.0 | length/termination/sampling | LOCAL_RUNTIME_ADAPTATION (+**D-LLM1** top_p) | YES (decision on top_p) |
| Provider retries | 10 attempts w/ backoff | none | failure handling | **EXPERIMENTAL_DEVIATION (D-LLM2, low)** | acceptable w/ documentation |
| JSON handling | `response_format` + `json.loads` | stopping criteria + extraction + `json.loads` | needed for HF | LOCAL_RUNTIME_ADAPTATION | YES |
| Seeds | none | per-run seeding from base 42 | reproducibility only | LOCAL_RUNTIME_ADAPTATION | YES |
| Confidence / verification | n/a | O(1) extra; `model.eval`, `inference_mode`, `use_cache`, sdpa | none on algorithm | ACCEPTABLE_LOCAL_ADAPTATION | YES |

---

## 10. TABLE — ONLY EXPECTED LOCAL-MODEL DIFFERENCES

| Original | Local | Why | Consequence |
|---|---|---|---|
| `gpt-4o` attacker | `Qwen/Qwen3-4B-Instruct-2507` | no OpenAI API/budget in this environment | attacker question quality and structured-output reliability differ; JSON failures now possible (hence D8) |
| `gpt-4o` target | `microsoft/Phi-4-mini-instruct` | no OpenAI API/budget | target compliance/refusal behaviour differs ⇒ absolute ASR not comparable to the paper; **within-study OFF-vs-ON contrast remains the measurable quantity** |
| `gpt-4o` evaluator (rubric/refusal/disclaimer) | `Qwen/Qwen3-4B-Instruct-2507` | no OpenAI API/budget | judge calibration differs (score distribution, refusal confidence `metadata` scale) |
| `gpt-4o` (the author also supports `llama3-8b-instruct` / `phi-4` for the *local* path) | `Qwen3-4B-Instruct-2507` + `Phi-4-mini-instruct` | smaller models that co-reside on one 24 GB GPU | the author's own local path is also a substitution; our choice still differs from theirs |
| OpenAI server-side sampling (no seed) | local sampling (temperature 0.7, top_p 1.0, per-run seeds) | local inference | stochastic behaviour is not reproducible across the two setups; no equivalence claimed |
| OpenAI JSON mode | local JSON stopping/extraction | no `response_format` locally | an added runtime mechanism (see §3.3) |

Nothing algorithmic appears in this table.

---

## 11. TABLE — MUST_FIX_BEFORE_PHASE14  (D1–D4, D8: **FIXED**, see §16)

| ID | File / location | Was | Required | Why it is not a local-model/runtime substitution | Status |
|---|---|---|---|---|---|
| **D1** | `attacks/crescendo_paper.py:546-550` + `attacks/runner.py` (both runner variants) | `record_turn` appended the raw target response; the summary returned by `generate_question_for_turn` was discarded | store `"(Summary) " + summary` for rounds > 1 (as `opposite_day.py` / `acronym.py` already did); keep the raw last response only for the current-turn echo | changes the attacker's observable information state (`(Summary) SUM2` → raw `R1`) | **FIXED** (§16.1) |
| **D2** | `attacks/runner.py` (sync + async loop heads) | `while len(history) + _num_filtering < max_turns` | transcribe the author's counter exactly: keep a `round_number` loop-carried variable recomputed inside the body as `len(history_t)//2 + 1` and test `round_number + num_filtering < max_rounds` | tested: with filters at turns 5–7 the author runs **7** iterations (4 accepts + 3 filters) and the local ran **8** (5 accepts + 3 filters) — an extra target call | **FIXED** (§16.2) |
| **D3** | `scripts/phase14_full_reproduction.py` attack factory | `attack._max_refusal_retries = cfg["nbf"]["trials"][attack]` (= 3 for crescendo/opposite-day/acronym) | do **not** inject `nbf.trials` into `_max_refusal_retries`; the author's limit is the literal `C_refused < 10` for all three (the class default is already 10) | changes the backtrack budget: after the 3rd refusal the local records the refusal as a scored turn and continues, whereas the author keeps backtracking to 10 | **FIXED** (§16.3) |
| **D4** | `evaluation/refusal_checker.py` `REFUSAL_CHECK_SYSTEM_PROMPT` | de-indented text, "accomplish the **objective**" | restore the author's exact `dedent()`ed text ("accomplish the **task**", original indentation, original example indentation, original `// comment`) | evaluator prompt text is not byte-identical; cheap to restore | **FIXED** (§16.4) |
| **D8** | `attacks/crescendo_paper.py::generate_crescendo_step` | 3-attempt retry with an appended "not valid JSON" repair turn + a hard-coded fallback question | either remove the retry (author makes one call and would raise) or keep it but record it as an explicit, per-run-counted deviation | adds attacker model calls and a synthetic question that the author never produces | **FIXED** (§16.5) |
| **D10** | `attacks/opposite_day.py` and `attacks/acronym.py` `_filter_prompt_with_nbf` | regeneration loop bounded by `self._max_turns` | remove, or make it a faithful `num_filtering` counter increment | author-absent regeneration semantics; unreached by Phase 14 today but reachable via the public `next_query` API (any other harness or test that uses it would silently deviate) | **REMAINING** (declared; dead on the Phase 14 path) |

Recommended (not blocking, but cheap and strictly-fidelity-positive):

| ID | Location | Current | Required |
|---|---|---|---|
| D5 | rubric prompt in `crescendo_paper.py` / `acronym.py` | 4 typo/whitespace "fixes" (documented in `tests/parity/test_crescendo_parity.py`) | restore the author's exact text (or freeze the current text and document it as a deliberate, pre-registered normalization) |
| D6/D7 | prompt constants | `’`→`'`, trailing whitespace stripped | restore byte-exact prompt strings |
| D-E1 | `llm_evaluate_with_rubric` | non-parsable judge reply → score 1 | raise/record a failure instead of silently scoring 1 (otherwise failures are indistinguishable from genuine score-1 responses) |
| D-LLM1 | `llm/local_client.py` generation | `top_p` unset (1.0) | set `top_p=0.9` if we intend to match the author's local HF path (the paper's OpenAI path had no `top_p`); decide and record |
| D-LLM2 | `llm/local_client.py` | no provider-level retry | add a bounded retry (author: up to 10) for transient local failures, or record every failure explicitly |

Test-coverage gap to close with the fixes: the existing parity tests
(`tests/parity/test_crescendo_parity.py`) exercise `generate_crescendo_step` with **synthetic**
histories (`["(Summary) s1", …]`), so they cannot detect D1 (the runner feeding) or D2 (the loop
budget). Add (a) a runner-level test that asserts the attacker message list round-by-round against a
transcribed author loop (the machinery already exists in `scripts/_audit_loop_parity.py`) and (b) a
loop-budget test with a deterministic stub barrier (test D in that script).

---

## 12. TABLE — ACCEPTABLE_LOCAL_ADAPTATIONS

| Item | Why it is safe |
|---|---|
| OpenAI chat-completions API → HF `transformers` pipeline | same messages, same roles, same temperature/json-flag contract; no prompt change |
| HF tokenizer + `apply_chat_template(add_generation_prompt=True)` | required to run a local model; the author's local path does the same |
| `max_new_tokens=256` | the exact value the author's own local HF path uses; the paper's API calls had no cap (documented as a local cap) |
| `temperature=0.0` evaluator + `do_sample=False` | greedy is the direct analogue of `temperature=0` API behaviour |
| `model.eval()`, `torch.inference_mode()`, `use_cache=True`, `attn_implementation="sdpa"`, bf16 | inference-only; verified not to touch weights, prompts, history, sampling params, or the algorithm |
| Local JSON stopping criteria + first-object extraction + parsing | required because HF returns text; the author's local path did not parse JSON at all (and would crash); verified not to truncate nested/pretty JSON on either Phase 14 tokenizer |
| `enable_thinking=False` for Qwen3-family templates | template-level flag; Qwen3-4B-Instruct-2507 is a non-thinking checkpoint, no effect on prompt content |
| `truncation=True, max_length=131072` | never reached at 8 turns × ≤256 tokens |
| `device_map="cuda"`, both models resident, `ModelManager` unused | scheduling/memory only; no effect on prompts, history, or sampling (and it removes the CPU↔GPU swap behaviour of Phase 8/12) |
| Per-run seeding (`random`, `numpy`, `torch`) from base 42 | reproducibility aid; the author seeds nothing, and cross-model stochastic equivalence is explicitly not claimed |
| `CountingChatLLM`, `[GEN]` logs, per-run JSONL metadata | telemetry only |
| `_embed_batch` normalizing 1-D embeddings to `[1,768]` | the author calls `.unsqueeze(0)` on the same 1-D encode output ⇒ identical input to `nbf`/`ssm` |
| OFF/ON isolation, fresh attack + fresh zero state per run | verified; matches the author's per-goal function-local state |

---

## 13. Remaining uncertainty

1. **JSON stopping criteria**: the token-level probes show no truncation for nested/pretty JSON on
   the two Phase 14 tokenizers, but the criterion is a heuristic (it fires whenever the last token is
   the standalone `}` id). A generation-level distribution test (does stopping early change the
   parsed JSON?) cannot be run without GPU model execution, which the audit deliberately did not do.
   Recommended hardening: make the criterion depth-aware, or require the JSON to be balanced.
2. **Attacker non-JSON frequency (post-D8)**: how often the local attacker emits non-JSON is unknown
   offline (Phase 11/12 reports indicate it happens). With the retry removed, such a round now raises
   `AttackGenerationError`, which is exactly the author's behaviour (the author indexes
   `data['generatedQuestion']` and would crash) but **must be recorded per run** — the runner must
   report these as explicit failed runs, never as silent substitutions.
3. **Rubric/refusal judge calibration** for Qwen3-4B vs gpt-4o cannot be quantified from this repo;
   the paper's absolute scores are not reproducible with a 4B judge. Only the OFF-vs-ON *contrast*
   is interpretable.
4. **Sampling non-determinism** (CUDA kernels, no `use_deterministic_algorithms`) means even local
   reruns are not bit-identical.
5. **ActorAttack** is excluded by capability, not by choice; if Phase 14's design needs it, a
   different local attacker must be selected (and re-audited).
6. `networks`/`steering`-style wrapper code paths (`SteeredLLMChat`, `allow_regeneration`) were not
   executed; they are inactive on the Phase 14 path so their fidelity was judged by inspection only.

---

## 14. FINAL VERDICT

```
READY_FOR_PHASE14
```

All five blocking deviations (**D1, D2, D3, D4, D8**) are fixed and verified against the author's own
loop code (§16). Every remaining difference is either

* original-identical (verified numerically or by execution — NBF math, checkpoint, dataset,
  embeddings, refusal rule, filter rule, round semantics), or
* a justified local-model / local-runtime substitution (§10, §12), or
* a **declared, non-blocking residual** that must be written into the Phase 14 run manifest
  (§14.1) and must not be silently absorbed.

### 14.1 Declared residual deviations (record in the manifest; not blocking)

| ID | Item | Effect | Why it is accepted |
|---|---|---|---|
| D5 | rubric prompt in `crescendo_paper.py` / `acronym.py`: 4 typo/whitespace normalisations (documented + asserted in `tests/parity/test_crescendo_parity.py`) | prompt bytes differ from the author in 4 places | semantically neutral (spelling/whitespace only); frozen and asserted by tests. **Restore byte-exact on request** |
| D6/D7 | `’`→`'`, trailing whitespace stripped in `OPPOSITE_DAY_SYSTEM_PROMPT`, `ACRONYM_SYSTEM_PROMPT`, `NETWORK_PROMPT` | prompt bytes differ | semantically neutral; documented in §4.2/§4.3 |
| D-E1 | `llm_evaluate_with_rubric`: non-parsable judge reply → score 1 | judge failures are indistinguishable from genuine score 1 | pre-existing harness behaviour; **count and log parser failures per run** so the contamination is measurable |
| D-LLM1 | `llm/local_client.py`: `top_p` unset (1.0) | sampling distribution | the paper's OpenAI path set no `top_p`; the author's optional local path used 0.9. Record the choice in the manifest |
| D-LLM2 | no provider-level retry around local generation | a transient CUDA/OOM error fails the run | explicit failures are preferable to silent retries; log and re-run failed ids |
| D10 | `_filter_prompt_with_nbf` regeneration loop in `opposite_day.py` / `acronym.py` | author-absent semantics on the public `next_query` path | **dead on the Phase 14 path** (the runner drives `generate_question_for_turn` and performs the official filtering itself); keep documented until that fork is removed |
| D11 | `attacks/runner_debug.py` still contains the **pre-fix** loop (`_effective_round`, `len(messages)//2 + 1`, `len(history) + _num_filtering` test) | none today | **not imported anywhere** (verified) — but it is a fidelity trap if it is ever used; either delete it or port the D1/D2 fixes |
| — | ActorAttack | blocked before the attack by the local attacker's structured output | `LOCAL_MODEL_CAPABILITY_LIMIT`; excluded from Phase 14 (1200 = 200 × 3 × 2) |

Dataset and checkpoint are byte-verified against the frozen values, the suite passes **535/535**, and
the loop-parity harness now reports byte-identical attacker/target message lists, histories, scores and
filter/refusal counts across 12 controlled schedules (including the formerly failing filter-tail and
refusal-at-boundary cases).

**Phase 14 may start** (batch 0, `--dry-run` first) once the residual items above are recorded in the
manifest. Note that passing tests do not by themselves establish fidelity — the evidence is §16.

---

## 15. Required remediation plan  (status)

1. **D3 — DONE.** The `_max_refusal_retries` injection is replaced by `OFFICIAL_MAX_REFUSAL_RETRIES = 10`
   (`phase14_full_reproduction.py`, attack factory) with the rationale in the code; `nbf.trials` is no
   longer pushed into the backtrack budget. Class defaults for crescendo/opposite-day/acronym are 10.
2. **D2 — DONE.** Both runner loop heads now transcribe the author's schedule (loop-carried
   `round_number`, recomputed in the body, `while round_number + _num_filtering < max_turns`), plus the
   author's `round_number -= 1` on a refusal backtrack (and on the paper's regeneration path).
3. **D1 — DONE.** Crescendo now stores the attacker-produced summary and feeds
   `"(Summary) " + summary` whenever the author's `round_number > 1` holds (runner publishes
   `_accepted_turns_so_far`); the current-turn echo stays raw. Full history is kept — no truncation.
4. **D8 — DONE (option a).** The 3-attempt repair loop and the fallback question are removed;
   `generate_crescendo_step` makes exactly one attacker call and raises `AttackGenerationError` on a
   malformed reply, as the author's direct `data['generatedQuestion']` indexing would. The same
   single-call rule was applied to the sibling attacks' generation steps.
5. **D4 — DONE.** `REFUSAL_CHECK_SYSTEM_PROMPT` is now the author's `dedent()`ed text (byte-identical:
   both 2504 chars, verified by importing the author's module).
6. **D5/D6/D7 — REMAINING (declared, non-blocking).** Whitespace/apostrophe/typo normalisations remain
   frozen and are asserted by `tests/parity/test_crescendo_parity.py`; restore byte-exact on request.
7. **D-E1/D-LLM1/D-LLM2 — REMAINING (declared, non-blocking).** Record judge-parse failures, the
   `top_p` choice and any transient generation failure in the run manifest.
8. **D10 — REMAINING (declared, dead code on the Phase 14 path).**
9. **Verification — DONE.** `python -m pytest -q` → **535 passed**; `scripts/_audit_loop_parity.py` and
   the 12-case sweep (§16.6) report byte-identical attacker/target message lists, histories, scores and
   filter/refusal counts against a verbatim transcription of the author's loop.

---

## 16. REMEDIATION APPLIED — evidence

Harness: `scripts/_audit_loop_parity.py` executes a **verbatim transcription of the author's
`run_crescendomation` loop** (`nbf_original_stuff/.../crescendomation/run.py`, including
`generate_crescendo_step`, the 3-trial filter block, the refusal branch with `history_t.pop()` and
`round_number -= 1`, and `C_refused < 10`) side-by-side with the local runner on the same deterministic
mock LLMs and a stub barrier, then diffs every observable artefact.

### 16.1 D1 — Crescendo attacker history
`generate_question_for_turn` now caches `lastResponseSummary`; `record_turn` appends
`"(Summary) " + self._last_summary` gated on the author's `round_number > 1`
(`_include_summary()`), which the runner publishes as `_accepted_turns_so_far = len(history)` at the top
of each iteration. Evidence — attacker message lists for all 8 rounds of the OFF schedule are
byte-identical to the author's (`identical=True` for calls 1–8), including
`The last response was: (Summary) SUM2` at round 3 and the raw `R2` echo for the current turn.

### 16.2 D2 — round budget
Evidence (formerly FAIL, now identical): filters applied to candidates 5–7 with `max_rounds=8` →
author **4 accepts + 3 filters = 7 iterations**, local **4 + 3 = 7**; unfiltered → **8 accepts** both
sides; `max_rounds=1` → **0 turns** both sides. A second defect found while fixing D2: deriving
`round_number` from `messages` left it pinned at 1 on the `SteeredLLMChat` path (where `messages` is not
maintained), which both mis-numbered the attacker prompt and disabled the loop bound; it is now derived
from the accepted-turn count (`len(history) + 1 == len(history_t)//2 + 1`).

### 16.3 D3 — backtrack budget
`OFFICIAL_MAX_REFUSAL_RETRIES = 10` for crescendo/opposite-day/acronym (ActorAttack keeps the author's
`N_retry = 3`). Evidence: with the 8th target response a refusal, author and local both produce
**8 accepted turns + 1 refusal = 9 target calls** (this boundary case failed before the
`round_number -= 1` parity fix).

### 16.4 D4 — refusal-check prompt
`tests`-independent check: `import` of the author's `check_refusal.py` and the local
`evaluation/refusal_checker.py` yields **equal** constants (2504 chars each); message construction,
newline stripping, `temperature=0.0`, `json_format=True` and the `value and metadata > 90` rule match.

### 16.5 D8 — attacker JSON retry
No retry/repair/fallback loop remains in `generate_crescendo_step`, `generate_opposite_day_step` or
`generate_acronym_step`; a malformed reply raises `AttackGenerationError`. Marker for the manifest:
one attacker call per round, exactly as the author.

### 16.6 Full parity sweep (author loop vs local runner)

```
rounds=8 unsafe=-                refusal@-: acc=8 filt=0 ref=0 IDENTICAL=True
rounds=8 unsafe=-                refusal@2: acc=9 filt=0 ref=1 IDENTICAL=True
rounds=8 unsafe=-                refusal@7: acc=9 filt=0 ref=1 IDENTICAL=True
rounds=8 unsafe=-                refusal@8: acc=9 filt=0 ref=1 IDENTICAL=True   (boundary)
rounds=8 unsafe=-                refusal@9: acc=8 filt=0 ref=0 IDENTICAL=True
rounds=8 unsafe=['Q3']           refusal@-: acc=7 filt=1 ref=0 IDENTICAL=True
rounds=8 unsafe=['Q5','Q6','Q7'] refusal@-: acc=4 filt=3 ref=0 IDENTICAL=True   (filter tail)
rounds=8 unsafe=['Q2']           refusal@3: acc=8 filt=1 ref=1 IDENTICAL=True
rounds=4 unsafe=['Q2']           refusal@2: acc=4 filt=1 ref=1 IDENTICAL=True
rounds=3 unsafe=-                refusal@1: acc=4 filt=0 ref=1 IDENTICAL=True
rounds=2 unsafe=-                refusal@-: acc=2 filt=0 ref=0 IDENTICAL=True
rounds=1 unsafe=-                refusal@-: acc=0 filt=0 ref=0 IDENTICAL=True

ALL CASES IDENTICAL: True
```

`IDENTICAL` compares: target-call count, filter count, refusal count, `history_a["score"]`,
`history_a["attacker"]`, `history_a["target"]`, every attacker message list (role, order and exact
content) and every target message list.

### 16.7 Suite
`python -m pytest -q` → **535 passed** (tests that encoded the old deviating expectations were updated
to the author's semantics: round budget = `max_rounds` accepted turns, one attacker call per phase,
and the `(Summary)`-only-for-rounds>1 history rule).
