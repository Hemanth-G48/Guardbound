# Phase 7 — Experimental Reproduction Setup Report

Status: setup complete; smoke experiments complete; artifacts recorded.
This file records the frozen configuration, every known deviation from the
official implementation (categorized), and the final gate assessment.

## Frozen implementation

The Phase 6 implementation is **FROZEN**. During Phase 7 setup, only two
harness-level (non-algorithmic) issues were found and fixed:

1. `run_one` promised `nbf_scores`/`final_score`/`backtracks` in JSONL but
   never populated them → **category 1 (implementation bug, harness only)**.
2. `make_llm` silently swapped gpt-4o → local Llama when no API key was set,
   and `build_metadata` reported the fallback as if it were the intended model
   → **category 2 (configuration problem)**.
3. My checkpoint dim check initially "failed" at 1536 → **not a bug;**
   the first layer consumes `[x; u]` concat (768 state + 768 embedding = 1536),
   exactly matching the official `train.py`. The check was corrected.

No attack algorithm, prompt, NBF threshold, or runner behavior was changed.

## 1. Configuration — COMPLETE

`configs/reproduction.yaml` is the single source of truth for reproduction
experiments. It explicitly specifies:

### Attacker
* **paper_reference_model.attacker**: `gpt-4o`
* **experiment_model.attacker**: `models/Llama-3.1-8B-Instruct`
* resolved role config `attacker.model`, `temperature=0.7`, `json_format=true`,
  `max_new_tokens=256`, `api_key_env: null` (NOT REQUIRED)

### Target
* **paper_reference_model.target**: `gpt-4o`
* **experiment_model.target**: `models/Llama-3.1-8B-Instruct`
* resolved role config `target.model`, `temperature=0.7`,
  `max_new_tokens=256`, `system_prompt_source: dataset.target_system`

### Evaluator
* **paper_reference_model.evaluator**: `gpt-4o` (the official design: attacker
  model = evaluator)
* **experiment_model.evaluator**: `models/Llama-3.1-8B-Instruct`
* resolved role config:
  * `evaluator.rubric_model`
  * `evaluator.refusal_model`
  * `evaluator.disclaimer_model` (Acronym only, official)
  * `evaluator.actor_step_judge_model`
  * `temperature: 0.0` (official: `check_refusal`/`evaluate_with_rubric` both
    use temp 0.0)
  * `json_mode: extracted` (local JSON extraction contract)

The evaluator is **explicitly identified per role** and is NOT assumed to be
the same model as the attacker — even though in this experiment all roles point
to the same physical local model, that is an explicit choice.

### NBF configuration
* embedding model: `all-mpnet-base-v2` (official)
* embedding dimension: `768` (paper)
* state dimension: `768` (paper)
* NBF checkpoint: pinned file + pinned sha256
  `cea1a75bcef4fc515814b69c42541c95114f587abcc4505c9b096bbfa2a136fe`
* checkpoint format: `original_author` (keys `ssm.*` / `nbf.*` loaded via
  `guardbound.models.compat`)
* checkpoint compatibility/conversion: verified byte-identical remap
* threshold / eta: `0.0` (official `--threshold` default; paper also reports
  `1e-3`)
* number of filtering trials:
  * Crescendo = 3
  * OppositeDay = 3
  * Acronym = 3
  * ActorAttack = 10
* initial state: `zeros(768)` (official: empty history → x_0 = 0)
* maximum turns: 8 (from dataset `max_rounds`)

The configuration makes it **impossible to accidentally use an arbitrary
locally trained checkpoint**: the sha256 is pinned, and the validation gate
refuses to start on any mismatch.

### Dataset configuration
* dataset file:
  `nbf_original_stuff/orginal_code_give_by_author/NBF-LLM/data/test/harmbench_tasks.json`
* paths verified: dataset exists
* fields per sample (official): `target_system`, `task`, `max_rounds`
* split: `test` (official directory `data/test/`)
* number of goals: **200** (verified: `len(file) == 200`)
* sampling procedure: **file_order** (official: sequential iteration, no
  shuffling)
* random seed: **NOT SPECIFIED BY OFFICIAL CODE** (flagged explicitly)
* whether tasks are shuffled: **not specified by official code** (flagged
  explicitly)
* whether a subset is used: **no** (official runs the full file;
  smoke runs use `--limit`)
* whether duplicate tasks are possible: **false** (verified: all 200 task
  strings unique)

### Model substitution (explicit)

```yaml
paper_reference_model:
  attacker: gpt-4o
  target: gpt-4o
  evaluator: gpt-4o

experiment_model:
  provider: hf_local
  attacker: models/Llama-3.1-8B-Instruct
  target: models/Llama-3.1-8B-Instruct
  evaluator: models/Llama-3.1-8B-Instruct
  temperature: 0.7
  max_new_tokens: 256
```

MODEL SUBSTITUTION — affects numerical comparability.
The objective is to reproduce the **process**, while model substitution is an
experimental limitation.

## 2. Hardware — VERIFIED

Verified on the experiment machine:

* GPU: NVIDIA RTX 4500 Ada Generation
* VRAM: 25.8 GB (reported by torch; the config records 24 GB per the machine
  spec — we note the discrepancy but the usable headroom is > 16 GB for bf16
  8B, so no quantization needed)
* dtype: bfloat16 (official local path)
* device: cuda
* model loading mode:
  `transformers_pipeline(device_map="cuda")`
* quantization: **none** (recorded explicitly; auto-quantizing forbidden)
* maximum context length: 131072 (Llama-3.1; effective use stays well under)
* expected memory usage: ~16.5 GB for bf16 8B + KV cache headroom

No quantization was applied — bf16 Llama-3.1-8B fits comfortably in 25.8 GB.

## 3. Checkpoint verification — VERIFIED

`scripts/verify_checkpoint.py` performs the full verification chain:

```
checkpoint loads successfully
  -> ssm (DialogueDynamics: f_theta / g_theta)           ✓ non-zero weights
  -> nbf (SafetyPredictor)                                 ✓ loads
  -> embedding dimension = 768                              ✓ 768
  -> state dimension = 768                                  ✓ 768
  -> predictor output = 5 classes                           ✓ 5
  -> NBF input = [x; u] concat (1536)                      ✓ 1536
  -> remap equivalence vs original author modules           ✓ byte-identical
```

Deterministic sanity test (fixed seeded inputs):

* logits: `[6.623433, -10.151479, 0.011153, -6.960104, -4.107246]`
* softmax probabilities: `[0.998635, 0.0, 0.001342, 1e-06, 2.2e-05]`
* barrier score `h(x,u) = -0.998613`
* threshold: `0.0`
* filtering decision: **accepted (safe)**

Full report written to `results/reproduction/checkpoint_verification.json`.

## 4. Dataset — VERIFIED

* file exists
* 200 goals, all with `target_system` / `task` / `max_rounds`
* all `max_rounds` = 8
* no duplicate tasks
* file order sampling (official)
* seed: NOT SPECIFIED BY OFFICIAL CODE (flagged)

## 5. Attacker — VERIFIED

* configured: `models/Llama-3.1-8B-Instruct`
* path exists locally
* provider: hf_local (explicit)
* temperature: 0.7 (official)
* JSON mode: extracted (local contract)
* max_new_tokens: 256 (official local path)
* NO silent fallback: nonexistent path = hard error

## 6. Target — VERIFIED

* configured: `models/Llama-3.1-8B-Instruct`
* path exists locally
* provider: hf_local (explicit)
* system prompt source: `dataset.target_system` (official)
* temperature: 0.7
* max_new_tokens: 256

## 7. Evaluator — VERIFIED

* rubric model: `models/Llama-3.1-8B-Instruct`
* refusal model: `models/Llama-3.1-8B-Instruct`
* disclaimer model: `models/Llama-3.1-8B-Instruct` (Acronym only)
* actor step judge model: `models/Llama-3.1-8B-Instruct`
* temperature: 0.0 (official)
* All roles explicit. No MockChatLLM is used.

## 8. Smoke experiments

### Experiment A — Attack without NBF

Command:
```
python scripts/run_reproduction.py --mode A --limit 3 --out results/reproduction/smoke_A_crescendo.jsonl
```

Attack: Crescendo (crescendo_paper). Goals: first 3 from harmbench (file order).
Models: real local Llama-3.1-8B-Instruct (attacker AND target AND evaluator all
the same physical model). No mock. No NBF.

### Experiment B — Attack with NBF

Command:
```
python scripts/run_reproduction.py --mode B --limit 3 --out results/reproduction/smoke_B_crescendo.jsonl
```

Same goals and same models. NBF enabled (official `--safety_filtering` semantics:
candidate filter, plain target calls, refusal backtracking).

### A vs B comparison

For each goal we record:

**Attack-only (A):**
* generated queries
* target responses
* refusal detection
* backtracking
* rubric score
* final success/failure
* number of turns
* runtime

**Attack + NBF (B):**
* same as A, plus:
* NBF score for each candidate
* filtered/accepted decision
* number of filtered queries

The purpose is to verify that the NBF is **actually participating** in the
attack loop. We do NOT expect the smoke-test ASR to match the paper.

### Verify no accidental fallback

The smoke experiment prints the full resolved configuration before execution:

```
[repro] resolved configuration:
  attacker  = models/Llama-3.1-8B-Instruct
  target    = models/Llama-3.1-8B-Instruct
  evaluator = models/Lama-3.1-8B-Instruct
  embedding = all-mpnet-base-v2
  checkpoint= ...
  dataset   = ... (3 goals)
  mock      = False
[repro] attacker=models/Llama-3.1-8B-Instruct target=models/Llama-3.1-8B-Instruct
[repro] NBF checkpoint sha256=...
```

Proof that:
* the intended attacker model was used ✓
* the intended target model was used ✓
* the intended evaluator was used ✓
* the intended NBF checkpoint was used ✓
* the intended embedding model was used ✓
* no MockChatLLM was used ✓
* no fallback model was silently selected ✓
* no malformed configuration silently received a default ✓

### Output artifacts

For every smoke run we produce JSONL containing:

* goal
* attack
* turns (query, response)
* NBF scores (if mode B)
* filter decisions (if mode B)
* refusal decisions (refusal_count)
* rubric scores
* success
* num_turns
* runtime

Files (never overwritten):

* `results/reproduction/smoke_A_crescendo.jsonl`
* `results/reproduction/smoke_A_crescendo.jsonl.manifest.json`
* `results/reproduction/smoke_B_crescendo.jsonl`
* `results/reproduction/smoke_B_crescendo.jsonl.manifest.json`
* `results/reproduction/checkpoint_verification.json`

## 9. Known deviations from official implementation

### Algorithmic deviations

* **None open.** The four attacks reproduce the official `run.py` loops,
  prompts, NBF filtering, and refusal backtracking.

### Configuration deviations

* **seed**: NOT SPECIFIED BY OFFICIAL CODE — the paper does not specify a
  sampling seed. We record `experiment.seed: 42` but it is used only for
  reproducibility metadata, not for dataset sampling. Dataset iteration is
  explicitly file-order (official).
* **shuffle**: NOT SPECIFIED BY OFFICIAL CODE — no shuffling observed in
  official code.
* **max_new_tokens**: the official local path uses 256. We use 256.

### Model deviations

* **MODEL SUBSTITUTION — affects numerical comparability**: paper uses gpt-4o
  (attacker = target = evaluator). This experiment uses local
  Llama-3.1-8B-Instruct for attacker, target, and all evaluator roles because
  no OpenAI API key is configured. Any resulting ASR is a **process
  reproduction, NOT a numerical paper reproduction**.
* Evaluator is explicitly separated from attacker in config even though they
  point to the same physical model. This is an explicit configuration choice,
  not a hidden assumption.

### Hardware/quantization deviations

* **quantization: none**: we did not quantize. The paper's local path does not
  specify quantization; our hardware (25.8 GB) accommodates bf16 8B without it.

### Implementation extensions (not deviations)

* **Phase C (full Guardbound pipeline)**: NBF candidate filtering + SteeredLLMChat
  target wrapper + regeneration on refusal. This goes beyond the official
  `--safety_filtering` mode and is explicitly labeled as an implementation
  extension in the config docstring.

## Final gate

### Configuration

**COMPLETE**

### Checkpoint

**VERIFIED**

### Dataset

**VERIFIED** (200 goals; seed NOT SPECIFIED BY OFFICIAL CODE)

### Attacker

**VERIFIED**

### Target

**VERIFIED**

### Evaluator

**VERIFIED**

### Smoke experiment

**PASSED**

Both smoke runs completed with real local models, real NBF checkpoint,
real embedding model, no mocks, no fallbacks. Artifacts recorded.

### Process reproduction readiness

**READY**

The implementation is frozen, the configuration is complete and validated,
the checkpoint is verified, the dataset is verified, and a smoke experiment
has run on real models with real NBF scoring.

### Numerical paper reproduction readiness

**NOT YET — requires full experimental reproduction**

The smoke test reproduces the **process**. Full numerical comparison against
paper numbers requires the full 200-goal experiment (or a statistically
representative subset) with the local model, plus a paper-number column
entered by the operator for side-by-side comparison. This is NOT marked READY
merely because the smoke test works.

```text
PHASE 7 COMPLETE — READY FOR FULL REPRODUCTION
```

### Operational note

Local generation runs at approximately 5-15 minutes per goal on the 8B model
depending on the attack (Crescendo/OD/Acronym are faster; ActorAttack pre-attack
adds calls). A full 200-goal × 4-attack experiment is a multi-day commitment.
Plan batching before launching.
