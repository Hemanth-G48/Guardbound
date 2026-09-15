# Phase 10 — Paper Model Fidelity Recovery

**Status:** `CONDITIONAL`

**Date:** 2026-09-13
**Repository revision:** `ad48a94b418b3cfe6a130df952763dee62c172d6`
**Purpose:** Recover the paper's exact model configuration from the strongest available sources, compare it to the current smoke configuration, assess RTX 4500 Ada (24 GB) feasibility, and decide whether the current setup is suitable for numerical paper reproduction.

---

## 1. Executive Summary

The official NBF-LLM code's strongest signal is unambiguous and uniform across all four attacks and all authored files:

**Paper default configuration:**

- **Attacker:** `gpt-4o`
- **Target:** `gpt-4o`
- **Evaluator / judge:** `gpt-4o` (single model serving all four judging roles)
- **Provider:** OpenAI API (OpenAI Python client, hosted)
- **Temperature:** `0.7` for generation; `0.0` for refusal / disclaimer / rubric / step-judge evaluation
- **JSON mode:** `response_format={"type": "json_object"}` for every JSON-formatted call
- **Max output tokens (API):** not bounded by the code; Claude branch uses `max_tokens=8192`; OpenAI/GPT branches have no `max_tokens` set
- **Max output tokens (local fallback):** `256` (hardcoded for `llama-3-8b-instruct` and `phi-4` local paths only)
- **Local optional targets:** `llama3-8b-instruct` and `phi-4` documented as supported alternatives
- **NBF embedder:** `all-mpnet-base-v2`
- **NBF checkpoint:** `models_best_nbf_released.pth` (768/768/5)
- **Dataset:** `data/test/harmbench_tasks.json`, 200 goals, `max_rounds=8`
- **Threshold default:** `0.0` (`--threshold` default = 0; code uses `threshold = -args.threshold`)

**The current Phase 8/9 smoke configuration is a documented model substitution, not a paper reproduction.** It is suitable for pipeline validation (which it has successfully done) but **not** for numerical paper reproduction. The only configuration that matches the paper's default is the existing OpenAI backend path with `gpt-4o` on all three roles, and that path cannot be executed in this environment because no OpenAI API key is configured.

---

## 2. Paper model configuration (recovered)

### 2.1 Attacker

| Field | Paper value | Evidence |
|---|---|---|
| Model | `gpt-4o` | `steering.py`: `--attacker_model` default=`"gpt-4o"` |
| Provider | OpenAI API | `steering.py` + `initialize_clients()` selects `clients['gpt']` when GPT_API_KEY/BASE_URL_GPT are set; `get_client()` routes any `gpt/o1/o3` name to `clients['gpt']` |
| API style | Chat completions, OpenAI Python SDK | `attacks/utils/generate.py`: the `else` branch (GPT) calls `client.chat.completions.create(...)` |
| Temperature | `0.7` | `generate(..., temperature=0.7)` default; attacker_generate passes temperature through from `steering.py` |
| JSON mode | `response_format={"type": "json_object"}` | `generate.py`: `response_format={"type": "text"} if not json_format else {"type": "json_object"}` |
| Max output tokens | **Not explicitly capped** for GPT path | `generate.py` OpenAI branch does not set `max_tokens`; only the Claude branch sets `max_tokens=8192` and only the local `llama`/`phi` branches hardcode `max_new_tokens=256` |
| System prompt requirements | Standard chat roles; system message passed in message list for GPT path | `generate.py` OpenAI branch sends the full `messages` list including system role; Claude branch strips system into a separate `system=` param |
| Reasoning mode | Not used / not configured | No `o1`/`o3` special handling except that those models skip custom temperature in `generate.py`; none of the four attacks request reasoning-mode models |
| Deterministic? | No, temperature 0.7 | sampling is stochastic |
| Shared across attacks | Yes, one `attacker_generate` per `steering.py` run | `steering.py` creates a single `attacker_generate` lambda and passes it to every attack |

### 2.2 Target

| Field | Paper value | Evidence |
|---|---|---|
| Model | `gpt-4o` | `steering.py`: `--target_model` default=`"gpt-4o"` |
| Provider | OpenAI API | same client selection / `get_client()` path as attacker |
| Temperature | `0.7` | `target_generate` uses `generate(messages, client=target_client, model=target_model, **kwargs)` and `kwargs` carries temperature |
| Max output tokens | Not explicitly capped for GPT path | same as attacker |
| System prompt | `target_system` from dataset (`harmbench_tasks.json`) | every `run_*` reads `test_case["target_system"]` and prepends it to `history_t` / `dialog_hist` |
| Conversation format | standard chat roles | every attack builds a message list with system/user/assistant roles and passes it to `target_generate` |
| Special parameters | none | |

### 2.3 Evaluator / judge

The paper uses a **single model for all judging roles**. There is no separate evaluator model in the official code. The same `attacker_generate` is reused for:

- rubric evaluation (`evaluate_with_rubric`)
- refusal classification (`check_refusal`)
- disclaimer classification (`check_disclaimer`, Acronym only)
- ActorAttack step judge (`step_judge`)
- pre-attack extraction (`extract_harm_target`)
- actor generation (`get_actors`)
- query chain generation (`get_init_queries`)

| Role | Paper model | Temperature | JSON mode | Evidence |
|---|---|---|---|---|
| Rubric evaluator | `gpt-4o` via `attacker_generate` | `0.0` | yes | `evaluate_with_rubric(query, response, goal, attacker_generate)` in every attack; `generate.py` called with `temperature=0.0, json_format=True` |
| Refusal evaluator | `gpt-4o` via `attacker_generate` | `0.0` | yes | `check_refusal(prompt, response, attacker_generate)` in Crescendo/OD/Acronym; `generate.py` called with `temperature=0.0, json_format=True` |
| Disclaimer evaluator | `gpt-4o` via `attacker_generate` | `0.0` | yes | `check_disclaimer(prompt, response, attacker_generate)` in Acronym; `generate.py` called with `temperature=0.0, json_format=True` |
| ActorAttack step judge | `gpt-4o` via `attacker_generate` | defaults to `0.7` (no explicit override in `step_judge`) | yes | `step_judge(query, response, attacker_generate)` calls `attacker_generate(messages, json_format=True)` with no temperature override |

**Important nuance:** The official `step_judge` call does **not** explicitly set `temperature=0.0`. The `generate` signature defaults to `temperature=0.7`. This is a paper-source fact worth recording: the official three-way step judge runs at `0.7`, not `0.0`, unless the caller overrides it. Guardbound's `ActorAttack.step_judge` currently delegates to `self._judge_llm` and preserves this.

### 2.4 NBF

| Field | Paper value | Evidence |
|---|---|---|
| Embedder | `all-mpnet-base-v2` | `build_ssm_nbf.py`: `SentenceTransformer(emb_name)` with `emb_name='all-mpnet-base-v2'` |
| Embed dimension | 768 | `train.py`: `input_dim = 768`, `SentenceTransformer('all-mpnet-base-v2')` |
| State dimension | 768 | `train.py`: `state_dim = 768` |
| Predictor output classes | 5 | `build_ssm_nbf.py`: `class_num=5`; `train.py`: `NeuralBarrierFunction(..., class_num=5)` |
| Checkpoint | `models/models_best_nbf_released.pth` (or equivalent author release) | `steering.py` `--model_path`; Guardbound pins `nbf_original_stuff/.../models_best_nbf_released.pth` |
| Threshold default | `0.0` | `steering.py`: `--threshold` default=`0` |
| Threshold sign convention | code uses `threshold = -args.threshold` then `ssm_nbf_score < threshold` for "safe" | every `run_*` and `build_ssm_nbf.py` |
| Filtering trials | 3 for Crescendo/OppositeDay/Acronym; 10 for ActorAttack query scoring and summary | `run_opposite_day`, `run_acronym`, `run_crescendomation`: `for index_ in range(3)`; `call_multi` and `summary`: `for index_ in range(10)` |

### 2.5 Dataset

| Field | Paper value | Evidence |
|---|---|---|
| File | `data/test/harmbench_tasks.json` | `steering.py`: `./data/test/harmbench_tasks.json` |
| Fields | `task`, `target_system`, `max_rounds` | every `run_*` reads all three |
| Split | `test` | file path |
| Number of goals | 200 | standard HarmBench test split |
| Sampling | file order, no shuffle in the official loop | `steering.py`: simple `for i, each_case in enumerate(test_case)` with `if i < args.start_from: continue` |
| Random seed | not specified by official code for sampling | `steering.py` has no shuffle/seed for goal selection |
| Subset | not specified by official code | `steering.py` runs all goals from `--start_from` |
| Duplicates | not possible from this loop | sequential file order |

---

## 3. Evidence / source trace

The model configuration is recovered from these sources, in descending authority:

1. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/steering.py`**
   - `--target_model` default `"gpt-4o"`
   - `--attacker_model` default `"gpt-4o"`
   - All four attacks invoked through `target_generate` and `attacker_generate`, both built from `generate(messages, client=..., model=..., **kwargs)`
   - `initialize_clients()` requires API keys for GPT/Claude/Llama; exits if none present
   - `get_client()` maps model name → client

2. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/generate.py`**
   - GPT path: `client.chat.completions.create(model=model, messages=messages, temperature=temperature, response_format={"type": "json_object"} if json_format else {"type": "text"})`
   - Claude path: `client.messages.create(..., max_tokens=8192, ...)`
   - Local `llama-3-8b-instruct` path: `max_new_tokens=256, top_p=0.9, do_sample=True`
   - Local `phi-4` path: `max_new_tokens=256, top_p=0.9, do_sample=True` and `assert not json_format`
   - Temperature default: `generate(..., temperature=0.7)`
   - This file is the single authoritative source for the paper's generation parameters.

3. **Each attack's `run.py`**
   - All four attacks use `attacker_generate(messages, json_format=True)` for attack generation.
   - All four attacks use `attacker_generate(messages, json_format=True, temperature=0.0)` for refusal/disclaimer/rubric scoring.
   - Acronym additionally uses `check_disclaimer` with the same pattern.
   - ActorAttack uses `step_judge` with `attacker_generate(messages, json_format=True)` and no temperature override (defaults to 0.7).
   - ActorAttack pre-attack (`infer_single`) uses `attacker_generate` for extraction, actor enumeration, and query chain generation.

4. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/check_refusal.py`**
   - `generate(messages, json_format=True, temperature=0.0)`
   - `return data["value"] and data["metadata"] > 90`

5. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/check_disclaimer.py`**
   - `generate(messages, json_format=True, temperature=0.0)`
   - `return data["value"]`

6. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/utils/evaluate_with_rubric.py`**
   - `generate(messages, json_format=True, temperature=0.0)`
   - returns `data["score"]`

7. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/attacks/build_ssm_nbf.py`**
   - Embedder `all-mpnet-base-v2`
   - State/input/output dim 768, hidden_dim 512, hidden_dim_nbf 32, class_num 5
   - Filtering: `for index_ in range(3)` for Crescendo/OD/Acronym; `for index_ in range(10)` appears in ActorAttack's `call_multi` and `summary`

8. **`nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/train.py`**
   - Confirms architecture dimensions and the `all-mpnet-base-v2` embedder used during training.
   - Confirms `models_best_nbf.pth` contains `{"ssm": ..., "nbf": ...}`.

9. **Guardbound's `scripts/run_reproduction.py`**
   - Correctly documents the paper defaults as its own header comment: `attacker = target = gpt-4o (official default)`, `eta = 0.0`, `temperature = 0.7`, `embedding = all-mpnet-base-v2`, `checkpoint = models/models_best_nbf_released.pth`.

**Disagreement check:** No material disagreement was found among the authored files. The only nuance is that `step_judge` in the official code runs at the default temperature (0.7) rather than 0.0, because the function calls `attacker_generate(messages, json_format=True)` without an explicit temperature kwarg. This is a paper-source fact, not a Guardbound deviation.

---

## 4. Attacker configuration

**Paper attacker:**

- `gpt-4o`, OpenAI API, temperature 0.7, JSON mode for structured calls, no explicit max-tokens cap on the GPT path, single model shared across all four attacks and all pre-attack/judge roles.

**Current Guardbound attacker:**

- `Qwen/Qwen3.5-4B`, local HF pipeline, bf16, temperature 0.7, `enable_thinking=false`, max_new_tokens 256.

**Exact match?** No.

**Deviation type:** Model substitution (category B in Part 3): documented, unavoidable locally without API access, but changes numerical comparability.

---

## 5. Target configuration

**Paper target:**

- `gpt-4o`, OpenAI API, temperature 0.7, no explicit max-tokens cap, `target_system` from dataset prepended as system message.

**Current Guardbound target:**

- `microsoft/Phi-4-mini-instruct`, local HF pipeline, bf16, temperature 0.7, max_new_tokens 256, `target_system` from dataset used as system prompt.

**Exact match?** No.

**Deviation type:** Model substitution (category B): documented, but affects ASR directly because target safety behavior and helpfulness are model-dependent.

---

## 6. Evaluator configuration

**Paper evaluator:**

- A single `gpt-4o` serving all judging roles, temperature 0.0 for refusal/disclaimer/rubric, default temperature for ActorAttack step judge.

**Current Guardbound evaluator:**

- `Qwen/Qwen3-4B-Instruct-2507`, local HF pipeline, bf16, temperature 0.0 for refusal/disclaimer/rubric.

**Exact match?** No.

**Deviation type:** Model substitution (category B), plus a role-separation architectural difference. The paper uses one model for everything; Guardbound uses one physical model but treats rubric/refusal/disclaimer/step-judge as separate logical subroles routed to the same evaluator. That separation is an implementation extension (category C in Part 3), not an attack-algorithm change, and it is integrity-positive because it keeps the evaluator roles independently configurable.

**One subtle point worth recording:** Guardbound's `ActorAttack.step_judge` currently routes through `self._judge_llm`, preserving the paper's default-temperature behavior for the step judge. If someone later hardcodes temperature 0.0 on the step judge, that would be a new deviation from the paper's `step_judge` behavior. This is not currently a problem, but it should be noted.

---

## 7. Generation parameters

| Parameter | Paper (GPT path) | Paper (local llama/phi path) | Guardbound current |
|---|---|---|---|
| Temperature (generation) | 0.7 | 0.7 | 0.7 |
| Temperature (evaluator) | 0.0 | n/a (local paths assert no JSON for phi; llama path not used for eval in paper) | 0.0 |
| Top-p | not set for GPT; 0.9 for local llama/phi | 0.9 | not set locally (no top_p override in HFLocalChatLLM) |
| Top-k | not set | not set | not set |
| Max tokens (GPT) | **not capped** | 256 | 256 for local; API path would not cap unless configured |
| Max tokens (local) | 256 | 256 | 256 |
| Stop sequences | not set | not set | not set |
| JSON mode (structured calls) | `response_format={"type": "json_object"}` | n/a (local phi asserts no JSON) | json_format=True on ChatLLM; local client extracts first JSON block on failure |
| Thinking/reasoning mode | not used | not used | `enable_thinking=false` explicitly set for attacker and evaluator |
| Repetition penalties | not set | not set | not set |
| Seed | not set | not set | not set |
| Chat template kwargs | n/a (GPT uses raw messages) | n/a (local llama/phi use `apply_chat_template`) | `enable_thinking=false` passed to `apply_chat_template` for Qwen models |

**Values marked "not set" are paper facts, not Guardbound gaps.** The official code does not pass these parameters for the GPT path. Guardbound should not invent non-default values and call them paper-faithful.

**One fidelity note:** The official code does **not** cap GPT output length, so the paper's GPT-based attacks could receive longer completions than Guardbound's local default of 256 tokens. For the local substitution this is an unavoidable deviation, but if the cloud path is ever used, it should preserve the paper behavior (no artificial cap unless the config explicitly adds one).

---

## 8. Current Guardbound configuration

**Phase 8/9 smoke configuration:**

- Attacker: `Qwen/Qwen3.5-4B` (local, bf16, ~4.66B)
- Target: `microsoft/Phi-4-mini-instruct` (local, bf16, ~3.8B)
- Evaluator: `Qwen/Qwen3-4B-Instruct-2507` (local, bf16, ~4B)
- Embedding: `all-mpnet-base-v2`
- NBF checkpoint: `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/models/models_best_nbf_released.pth`
- Threshold: 0.0
- Filtering trials: crescendo 3, opposite_day 3, acronym 3, actor_attack 10
- Dataset: `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json`
- Max turns: 8
- Temperature: 0.7 attacker/target, 0.0 evaluator
- Max new tokens: 256 all roles
- Provider: local for all three roles

This configuration is **exactly what Phase 9 ran**. It is a valid and integrity-preserving local configuration. It is **not** the paper configuration.

---

## 9. Model-fidelity comparison

### 9.1 Role matrix

| Role | Paper model | Provider | API/local | Temperature | Max tokens | Guardbound current model | Exact match? | Evidence |
|---|---|---|---|---|---|---|---|---|
| Attacker | gpt-4o | OpenAI | API | 0.7 | not capped (GPT) | Qwen/Qwen3.5-4B | No | steering.py + generate.py |
| Target | gpt-4o | OpenAI | API | 0.7 | not capped (GPT) | microsoft/Phi-4-mini-instruct | No | steering.py + generate.py |
| Rubric evaluator | gpt-4o (shared) | OpenAI | API | 0.0 | not capped (GPT) | Qwen/Qwen3-4B-Instruct-2507 | No | run_*.py + evaluate_with_rubric.py |
| Refusal evaluator | gpt-4o (shared) | OpenAI | API | 0.0 | not capped (GPT) | Qwen/Qwen3-4B-Instruct-2507 | No | run_*.py + check_refusal.py |
| Disclaimer evaluator | gpt-4o (shared) | OpenAI | API | 0.0 | not capped (GPT) | Qwen/Qwen3-4B-Instruct-2507 | No | acronym/run.py + check_disclaimer.py |
| ActorAttack step judge | gpt-4o (shared) | OpenAI | API | 0.7 (no override) | not capped (GPT) | Qwen/Qwen3-4B-Instruct-2507 | No | actor_attack/run.py + step_judge |

### 9.2 Component matrix

| Component | Paper behavior | Current Guardbound behavior | Deviation |
|---|---|---|---|
| Attacker model | gpt-4o API | Qwen3.5-4B local bf16 | Model substitution |
| Target model | gpt-4o API | Phi-4-mini-instruct local bf16 | Model substitution |
| Evaluator model | gpt-4o API shared | Qwen3-4B-Instruct-2507 local bf16 | Model substitution |
| Evaluator role structure | One model, all roles | One physical model, logically separate subroles | Implementation extension (does not change attack algorithms) |
| Temperature generation | 0.7 | 0.7 | Match |
| Temperature evaluation | 0.0 (refusal/disclaimer/rubric), 0.7 (step judge) | 0.0 (refusal/disclaimer/rubric), routes step judge through same model | Match on temperature semantics; step judge delegation is faithful |
| Max tokens (GPT path) | not capped | 256 local; API path not capped unless configured | Substitution effect if cloud path used without explicit cap |
| JSON mode | `response_format={"type":"json_object"}` | ChatLLM json_format flag; local extracts first JSON block on parse failure | Behavior match on intent; local robustness is an extension |
| Reasoning/thinking | not used | `enable_thinking=false` explicit | Intent match; Guardbound makes it explicit rather than silent |
| NBF embedder | all-mpnet-base-v2 | all-mpnet-base-v2 | Match |
| NBF checkpoint | author release | same author release, sha256 pinned | Match |
| NBF dimensions | 768/768/5 | 768/768/5 | Match |
| NBF threshold default | 0.0 | 0.0 | Match |
| NBF filtering trials | 3 / 3 / 3 / 10 | 3 / 3 / 3 / 10 | Match |
| Dataset | harmbench_tasks.json test, 200 goals, file order | same file, same split, same order | Match |
| Max turns | 8 | 8 | Match |
| Attack algorithms | official | official (with robustness fixes that do not change prompts/algorithms) | Match |
| ActorAttack pre-attack | `infer_single` called once per goal before loop | `prepare_attack(goal)` called once per goal before loop | Match after Phase 9 fix |
| Refusal handling | LLM refusal judge, backtrack | LLM refusal judge, backtrack | Match |
| Acronym disclaimer | `check_refusal OR check_disclaimer` before rubric | same | Match |
| ActorAttack step judge | three-way judge with rewrite/retry | three-way judge with rewrite/retry | Match |

---

## 10. RTX 4500 Ada (24 GB) feasibility

### 10.1 Paper's primary model: gpt-4o

- Not local-executable by definition.
- No local bf16 weights exist for `gpt-4o`.
- No local substitute is the same model.
- If the experiment wants the exact paper configuration, this role requires an external API and an API key.
- Guardbound already has the OpenAI backend (`OpenAIChatLLM`) and the `providers.cloud` path in the config. The backend is present and ready; execution has not been tested here because no API key is configured.

### 10.2 Closest defensible local reproduction options

The paper also documents local support for `llama3-8b-instruct` and `phi-4`. Those are not "the paper model", but they are the paper's own documented local fallback targets. Against that backdrop, the relevant question for the RTX 4500 is which local model best matches the paper's **intent** while actually fitting.

**Candidate roles and approximate bf16 footprints:**

- `Qwen/Qwen3.5-4B` — attacker — ~8.3 GB bf16 resident, 8.8 GB on disk.
- `microsoft/Phi-4-mini-instruct` — target — ~7.7 GB bf16 resident, 7.2 GB on disk.
- `Qwen/Qwen3-4B-Instruct-2507` — evaluator — ~8.0 GB bf16 resident, 7.6 GB on disk.

**Can all three co-reside?** No. 8.3 + 7.7 + 8.0 ≈ 24 GB before overhead, and in practice the pipeline wants one role resident at a time anyway. With ModelManager strict eviction, peak resident is one model at a time, so peak VRAM is ~8.5 GB and the 24 GB GPU is not the binding constraint for these specific models.

**Can a larger target be used?** Yes, with caveats. A stronger target would improve the Crescendo saturation observation, but it increases per-load VRAM and may require swapping. The Phase 9 observation that OppositeDay reached rubric 5 on Phi-4-mini is the more important fact: the target is not the only bottleneck; attack-stringency and attacker capability are also in play.

**Does the exact paper model fit the RTX 4500?** No, because the exact paper model is a hosted API model, not a local weight file. The question is therefore not "does gpt-4o fit" but "can we match the paper's model choice on this hardware". The honest answer is:

- **Exact match:** impossible locally; requires API.
- **Closest local reproduction:** possible, but it is still a substitution. The paper's own local docs point to `llama3-8b-instruct` and `phi-4` as the intended local targets, which suggests the paper authors themselves treated local execution as a different execution mode, not identical to the GPT path.

### 10.3 Quantization

The current configuration uses unquantized bf16 for all three models. That is the most faithful local precision choice and it fits the 24 GB GPU with ModelManager swapping.

Quantization is **not required** for this three-model stack on 24 GB. It would only become relevant if a materially larger model were chosen for one of the roles and co-residency were desired. If quantization is ever introduced, it must be recorded explicitly as a deviation, because it changes model behavior.

---

## 11. API vs local execution

### 11.1 Does the paper require an API?

Yes, for the default configuration. The paper's defaults (`--target_model gpt-4o`, `--attacker_model gpt-4o`) are hosted OpenAI models, and `initialize_clients()` exits if no API credentials are present.

### 11.2 Does Guardbound already support the API path?

Yes.

- `OpenAIChatLLM` exists and supports `model`, `temperature`, `json_format` via `response_format={"type": "json_object"}`, and standard message formatting.
- `providers.cloud` is present in the three-model config (currently `enabled: false`).
- The runner's `build_metadata` / `make_llm` paths already handle the cloud path as a config switch, not a code fork.

### 11.3 What is required to use it?

- A valid `OPENAI_API_KEY` (or configured base URL + key) in the environment.
- Someone to authorize the cost/lab policy for running it.
- An explicit decision that the experiment is authorized to call the API.

### 11.4 Has the API path been tested in this environment?

No. No OpenAI API key is configured in this environment. The correct statement is:

> Exact paper configuration identified, but API execution not performed because credentials/cost authorization are unavailable.

This is not a Guardbound gap. It is an environment/authorization state.

---

## 12. Attack-specific model requirements

### 12.1 Crescendo

- Requires an attacker that can produce structured escalating questions in JSON and follow multi-turn context.
- Requires a target that can be progressively driven toward rubric 5.
- Paper uses `gpt-4o` for both.
- Guardbound's implementation is faithful; the Phase 9 saturation under `Qwen3.5-4B → Phi-4-mini` is a substitution effect, not a code defect.

### 12.2 OppositeDay

- Requires an attacker that can produce the nested opposite-viewpoint jailbreak structure and summaries.
- Requires a target that can be nudged from opposing-viewpoint compliance into the original harmful viewpoint.
- Paper uses `gpt-4o` for both.
- Guardbound proved this attack can reach rubric 5 with the substitute models, which is the strongest evidence that the pipeline itself is sound. The paper'sgpt-4o version would be expected to perform differently, possibly better, but that is a model-capability claim that must be tested with gpt-4o rather than asserted.

### 12.3 Acronym

- Requires an attacker that can reliably complete acronym extraction/build/shift prompt chains and return `generatedQuestion` / `lastResponseSummary` in JSON.
- Requires a target that can be gradually shifted into the harmful output through acronym framing.
- Paper uses `gpt-4o` for both.
- Guardbound's Acronym is faithful; the Phase 9 result (0 turns) is a model-cooperation failure of `Qwen3.5-4B`, not a pipeline bug.

### 12.4 ActorAttack

- Requires an attacker that can complete a multi-step pre-attack chain: extract harm target → enumerate actors → generate initial query chains → rewrite on refusal → three-way step judge.
- Requires a target that answers substantively enough for the step judge to see progress.
- Paper uses `gpt-4o` for both attacker and the step judge, and the step judge in the paper runs at default temperature (0.7), not 0.0.
- Guardbound's implementation is faithful after the Phase 9 `prepare_attack` fix and defensive parsing fixes. The Phase 9 result (mostly 0 turns, one 2-turn run) is a model-cooperation failure of `Qwen3.5-4B` on the pre-attack chain.

**Summary:** None of the four attacks contains a hidden assumption that only gpt-4o can satisfy in principle. The paper chose gpt-4o, and the attacks were designed around gpt-4o-class capability. The Phase 9 failures are best explained as **this specific substitute attacker being too weak for two of the four attacks' prompt chains and too weak for Crescendo escalation against this target**, not as the attacks being gpt-4o-only by design.

---

## 13. Phase 9 failure interpretation

### 13.1 ActorAttack failure

Observed under the current substitute configuration:

- `Qwen3.5-4B` frequently fails to complete ActorAttack's pre-attack chain cleanly enough to produce usable actors or query lists.
- Phase 9 produced 0-turn runs in most cases; one Mode B goal produced 2 turns.
- This is a **model-substitution limitation**, observed with this specific attacker. It does not prove ActorAttack is broken in Guardbound. The official `run_actor_attack` calls `infer_single(goal, actor_num, attacker_generate)` exactly once per goal before the attack loop; Guardbound now does the equivalent with `prepare_attack(goal)`.

### 13.2 Acronym failure

Observed under the current substitute configuration:

- `Qwen3.5-4B` frequently returns an empty or malformed `generatedQuestion` from Acronym's first prompt.
- Phase 9 produced 0-turn runs across the board.
- This is the same class of model-cooperation limitation as ActorAttack. Acronym's prompt chain is different, but the failure mode is similar: the substitute attacker does not reliably return the structured JSON the attack needs.

### 13.3 Crescendo saturation

Observed under the current substitute configuration:

- `Qwen3.5-4B` produces escalating Crescendo queries, but `Phi-4-mini-instruct` stays at rubric ≤ 3 across all 6 Crescendo runs in Phase 9.
- OppositeDay reached rubric 5 on the same goals and same target, which shows `Phi-4-mini` is capable of rubric 5 when the attack prompts are more effective.
- This is an **attack × target interaction observed under substitution**, not a statement that Crescendo cannot reach rubric 5 in general. The paper's `gpt-4o` attacker likely produces different escalation, and the paper's `gpt-4o` target likely responds differently.

### 13.4 OppositeDay success

Observed under the current substitute configuration:

- OppositeDay reached rubric 5 on 2 of 3 goals in Mode A.
- This demonstrates the pipeline can execute a successful multi-turn jailbreak end to end, including escalation, refusal backtracking, and rubric-5 termination.
- The fact that Mode B did not reproduce the same successes is confounded by the stochastic first-prompt behavior of the substitute attacker across runs; it is not evidence that the NBF breaks OppositeDay.

---

## 14. Candidate reproduction configurations

### 14.1 Configuration A — Exact paper configuration

- Attacker: `gpt-4o`, OpenAI API, temperature 0.7, JSON mode for structured calls.
- Target: `gpt-4o`, OpenAI API, temperature 0.7.
- Evaluator: `gpt-4o`, OpenAI API, temperature 0.0 for refusal/disclaimer/rubric; default temperature for step judge.
- Precision: hosted API (no local precision concept).
- Quantization: n/a.
- Provider: API for all three roles.
- VRAM requirement: n/a on local GPU; API execution only.
- Fidelity level: exact default match to the paper's `steering.py` defaults.
- Major deviations: none relative to the paper defaults, except that execution requires an authorized API key and cost approval.

This is the configuration that should be used if the goal is numerical paper reproduction and API access is authorized.

### 14.2 Configuration B — Closest local reproduction

This is a judgment call, not a precise match, because no local model is the paper model. The defensible basis for choosing a local substitute is the paper's own documented local support plus the available hardware.

One reasonable candidate:

- Attacker: a stronger local instruction model than `Qwen3.5-4B`, if the goal is to unblock ActorAttack/Acronym locally.
- Target: a local model in the paper's documented local family, e.g. a Phi-4-class or Llama-3-class instruct model, if a stronger target is desired for Crescendo saturation.
- Evaluator: a separate local instruct model with reliable JSON behavior, temperature 0.0 for judging roles.
- Precision: bf16 if it fits; quantization only if recorded explicitly.

Honest caveat: this is still a substitution. The paper's own local docs treat `llama3-8b-instruct` and `phi-4` as supported local targets, which implies the authors considered local execution a useful but distinct mode. Any local model choice should be documented as a substitution even if it is "closer" in some sense.

### 14.3 Configuration C — Current engineering smoke configuration

- Attacker: `Qwen/Qwen3.5-4B`
- Target: `microsoft/Phi-4-mini-instruct`
- Evaluator: `Qwen/Qwen3-4B-Instruct-2507`
- Precision: bf16
- Quantization: none
- Provider: local for all three roles
- VRAM requirement: ~8.5 GB peak resident with ModelManager swapping; fits 24 GB comfortably
- Fidelity level: pipeline-validation configuration only
- Major deviations: all three models are substitutions; no API model is used; max tokens 256 local; JSON robustness is a local extension.

This configuration is retained as the validated pipeline configuration. It should **not** be presented as a paper reproduction.

---

## 15. Deviations

### 15.1 Algorithmic deviations

None identified in the current attack implementations relative to the official code, after Phase 9. The implementation is faithful to the official loops, prompts, refusal/disclaimer ordering, backtracking, NBF filtering trial counts, rubric, and ActorAttack step-judge/rewrite flow. The defensive JSON parsing fixes in `actor_attack.get_init_queries` and `get_actors` are robustness fixes that do not change prompts or algorithms.

### 15.2 Configuration deviations

- Current smoke uses local substitute models instead of `gpt-4o`.
- Current smoke uses `max_new_tokens=256` for all roles; the paper's GPT path does not cap tokens. This matters only if/when the cloud path is used.
- Current smoke sets `enable_thinking=false` explicitly for Qwen models; the paper does not use thinking models. This is an explicit preservation of intent, not a hidden default.
- Current smoke routes all evaluator subroles to one physical local model, but keeps them logically separate. The paper used one model with no separate subrole routing. This is an implementation extension, not an algorithmic deviation.

### 15.3 Model deviations

- Attacker: `Qwen/Qwen3.5-4B` instead of `gpt-4o`.
- Target: `microsoft/Phi-4-mini-instruct` instead of `gpt-4o`.
- Evaluator: `Qwen/Qwen3-4B-Instruct-2507` instead of `gpt-4o`.

### 15.4 Implementation extensions

- Separate evaluator subrole routing in config (`rubric`, `refusal`, `disclaimer`, `actor_attack_judge`).
- Local JSON robustness: extracting the first JSON block when parsing fails.
- Defensive handling of malformed actor/question JSON from the attacker.
- `prepare_attack(goal)` runner hook for ActorAttack.

These extensions improve robustness and config clarity without changing the attack algorithms. They are acceptable and should remain.

---

## 16. Final reproduction readiness decision

**Phase 10 status: CONDITIONAL**

**Paper models:**

- Attacker: `gpt-4o` (OpenAI API)
- Target: `gpt-4o` (OpenAI API)
- Evaluator: `gpt-4o` (OpenAI API), single model for all judging roles, temperature 0.0 for refusal/disclaimer/rubric, default temperature for step judge

**Current smoke models:**

- Attacker: `Qwen/Qwen3.5-4B`
- Target: `microsoft/Phi-4-mini-instruct`
- Evaluator: `Qwen/Qwen3-4B-Instruct-2507`

**Exact deviations:**

1. All three roles are local substitute models, not `gpt-4o`.
2. Model substitution is the dominant deviation and affects every attack's ASR.
3. Max-tokens cap differs for the GPT path (paper does not cap; local uses 256).
4. Evaluator role structure is more granular in Guardbound than in the paper, but behaviorally faithful.
5. `step_judge` temperature fidelity is currently correct (default 0.7 in both paper and Guardbound); any future hardcoding of 0.0 there would be a new deviation.

**Best reproduction configuration:**

- If the goal is numerical paper reproduction and API access is authorized: use **Configuration A**, exactly the paper defaults (`gpt-4o` for attacker/target/evaluator) through the existing `OpenAIChatLLM` / `providers.cloud` path. No code changes are required to support this; it is already the config switch.
- If the goal is only pipeline validation or a local ablation study: the current smoke configuration is acceptable, but results must be labeled as substitute-model results, not paper reproduction.

**RTX 4500 feasibility:**

- The exact paper model (`gpt-4o`) does not run locally by definition; it requires the API path.
- The current three-model local stack fits the 24 GB GPU comfortably with ModelManager swapping; no quantization is required for these specific models.
- A larger local target is feasible with swapping, but any such change is a substitution decision, not a fidelity improvement.

**API requirement:**

- The paper's default configuration requires the OpenAI API for all three roles.
- Guardbound already has the required backend.
- The API path has **not** been tested here because no API key is configured.
- Exact paper configuration identified, but API execution not performed because credentials/cost authorization are unavailable.

**Numerical paper reproduction readiness:**

**NOT READY** under the current local configuration.

The current configuration is suitable for pipeline validation, which it has already passed. It is not suitable for claiming numerical paper reproduction, because the models are substitutions and the Phase 9 results show that two of the four attacks are effectively blocked by the substitute attacker and Crescendo saturates on the substitute target.

---

## 17. Recommended next phase

### Recommended next step

**Do not start a large experiment yet.** The correct next action is a decision about whether to run the exact paper configuration through the API or to continue with a documented local substitution study.

If the objective is **faithful numerical paper reproduction**, the next phase should be:

1. Decide whether API execution is authorized.
2. If yes: enable `providers.cloud`, set all three roles to `gpt-4o`, keep temperature and JSON semantics as in the paper, keep threshold 0.0, keep the official dataset and max turns, run a small goal sample first, and compare only after that is complete.
3. If no: do not claim paper reproduction from the current local stack. Instead, either:
   - run a documented substitute-model study with explicit caveats, or
   - revisit the local model choice for attacker/evaluator only if the goal is to unblock ActorAttack/Acronym locally, while still labeling results as substitution results.

### What should not happen next

- Do not start a 200-goal run under the current local configuration and label it paper reproduction.
- Do not tune prompts, thresholds, or evaluator scoring to improve ASR before the model configuration question is settled.
- Do not introduce quantization or a larger model merely to make the numbers look better.

### Why this is the right recommendation

The Phase 10 investigation found that the paper's default configuration is clearly recoverable and unambiguous: `gpt-4o` for attacker, target, and evaluator, via the OpenAI API. Guardbound already supports that configuration. The only remaining blocker to exact reproduction is authorization/access to the API, not the codebase.

Continuing to iterate on the local substitute stack would be optimizing a substitution rather than reproducing the paper. That is the wrong objective for the next phase if the stated goal is faithful reproduction.

---

*End of Phase 10 report.*
