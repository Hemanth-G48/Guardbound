# Guardbound — Reproduction of arXiv:2503.00187

Reproduction of *"Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks"*
(Hu, Robey, Liu — TMLR 2026): a Neural Barrier Function (NBF) that proactively filters harmful queries
in multi-turn LLM dialogues via a control-theoretic Q-filter.

## Start here (workflow)

```
docs/IMPLEMENTATION_ROADMAP.md   ← master plan: full paper analysis + 9 phases
prompts/PHASE_<1..9>_PROMPT.md   ← one self-contained, ready-to-paste prompt per phase
src/guardbound/                ← the implementation, one subpackage per phase
```

### Pipeline at a glance

```
OFFLINE
  P2 data pipeline ──► P3 dynamics f_θ,g_θ ──► P4 predictor h + NBF losses ──► checkpoint
        ▲                                                                │
  Circuit Breakers goals                                                 │ frozen
  4 jailbreak attacks vs GPT-3.5-turbo                                   ▼
  GPT-4o judge (1–5)                          ONLINE DEFENSE (P5)
  sentence embeddings (768-d)                 query → embed → h(x_{k-1}, u_k)
                                              ├─ h + η ≥ 0  → FILTER (never call LLM)
                                              └─ else       → target LLM; x_k = f_θ(x_{k-1}, u_k)

EVALUATION
  P6 attacks (ActorAttack / Crescendo / Opposite-day / RedQueen / adaptive)
      ──► transcripts ──► P7 metrics (ASR, MMLU, MTBench, over-refusal, guardrail F1)
  P8 baselines (system prompt, LoRA SFT/DPO/KTO) evaluated through the same P7 pipeline
  P9 orchestrates everything into Tables 1–15 + Figs. 5–11
```

### Phase → folder map

| Phase | What it builds | Package module | Key scripts | Prompt file |
|---|---|---|---|---|
| **1** ✅ | Scaffolding: config, schema, LLM clients, embeddings | `config.py`, `schemas.py`, `llm/`, `embeddings.py` | `scripts/check_env.py` | `prompts/PHASE_1_PROMPT.md` |
| **2** ✅ | Data pipeline: attacks, judging, embeddings, tensors | `data/`, `attacks/` | `build_conversations.py`, `judge_conversations.py`, `embed_conversations.py`, `build_datasets.py` | `prompts/PHASE_2_PROMPT.md` |
| **3** ✅ | Dynamics MLPs f_θ, g_θ + `L_dyn` training (Adam 1e-4, 200 ep) | `models/dynamics.py`, `training/losses.py`, `training/diagnostics.py`, `training/train_dynamics.py` | `scripts/train_dynamics.py` | `prompts/PHASE_3_PROMPT.md` |
| **4** ✅ | Predictor h + NBF losses `L_CE`,`L_SS`,`L_SI` (Adam 1e-3, 200 ep) | `models/predictor.py`, `training/nbf_losses.py`, `training/train_nbf.py` | `scripts/train_nbf.py`, `scripts/ablate_loss_weights.py` | `prompts/PHASE_4_PROMPT.md` |
| **5** ✅ | Q-filter steered chat runtime (filter iff `h+η ≥ 0`) | `defense/steered_chat.py`, `defense/context_init.py`, `defense/post_processing.py`, `defense/evaluation_runner.py` | `scripts/demo_steered_chat.py` | `prompts/PHASE_5_PROMPT.md` |
| **6** ✅ (paper-strict) | Attack suite: interface + runner + adaptive selector ✅; 5 non-adaptive attacks are paper-strict stubs awaiting official paste-in | `attacks/base.py`, `attacks/runner.py`, `attacks/registry.py`, `attacks/adaptive.py`, `attacks/{crescendo,actor_attack,opposite_day,acronym,red_queen}.py` (stubs) | `scripts/run_attacks.py` | `prompts/PHASE_6_PROMPT.md` |
| **7** ✅ | Evaluation: ASR (per-behavior, cached) + MMLU + MTBench + over-refusal + guardrail F1 + reporting | `evaluation/{asr,mmlu,mtbench,over_refusal,guardrail_f1,refusal_detector,reporting,cache}.py`, `evaluation/guards/*.py` | `scripts/judge_asr.py`, `scripts/eval_mmlu.py`, `scripts/eval_mtbench.py`, `scripts/eval_over_refusal.py`, `scripts/bench_guardrails.py` | `prompts/PHASE_7_PROMPT.md` |
| **8** ✅ | Defense baselines: original / system_prompt / lora_sft / lora_dpo / lora_kto / guardbound — all through one registry, reuse Phase 6 attacks + Phase 7 metrics | `baselines/{base,registry,original,system_prompt,serving,manifests,build_sft_data,build_preference_data}.py`, `baselines/prompts/llama2_safety_system.txt`, `baselines/configs/*.yaml` | `scripts/train_lora_{sft,dpo,kto}.py`, `scripts/run_baseline_evals.py` | `prompts/PHASE_8_PROMPT.md` |
| **9** ✅ | Experiments E1–E10, Pareto analysis, findings verification, resumable runner, smoke mode, final report | `experiments/{registry,runner,retrain}.py`, `analysis/{pareto,aggregator,tables,plots,findings}.py` | `scripts/run_experiments.py`, `scripts/run_all.py`, `scripts/build_final_report.py` | `prompts/PHASE_9_PROMPT.md` |
| 9 | Experiment matrix E1–E10, ablations, figures | `experiments/`, `analysis/` | `run_experiments.py`, `run_all.py` | `prompts/PHASE_9_PROMPT.md` |

Dependency order: `P1 → P2 → P3 → P4 → P5 → P6/P7/P8 → P9`
(P6 can start once P2 infra exists; P8 needs only P2 data + P6/P7 harnesses.)

## Setup

```bash
pip install -r requirements.txt          # heavy deps are lazy-imported; core works without them
pip install -e .                         # install package for clean imports
python -m pytest tests/ -q               # 270+ tests, fully offline (mocked SDKs)
python scripts/check_env.py              # validates config against paper constants
python scripts/check_env.py --with-embeddings   # optional: real 768-d embedding check
```

Set API keys only when you reach phases that need them:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

## Phase 2 data pipeline

```bash
# 1. Generate attack conversations (requires OPENAI_API_KEY)
python scripts/build_conversations.py --attack acronym --num-goals 1000
python scripts/build_conversations.py --attack crescendo --num-goals 1000
python scripts/build_conversations.py --attack opposite_day --num-goals 1000
python scripts/build_conversations.py --attack actor_attack --num-goals 2327

# 2. Judge every turn with GPT-4o (requires OPENAI_API_KEY)
python scripts/judge_conversations.py --input data/processed/conversations/acronym.jsonl

# 3. Extract embeddings (requires sentence-transformers)
python scripts/embed_conversations.py --input data/processed/conversations_labeled.jsonl

# 4. Build tensor datasets
python scripts/build_datasets.py --input data/processed/conversations_embedded.jsonl

# Dry-run (no API calls)
python scripts/build_conversations.py --attack acronym --num-goals 2 --dry-run
```

All intermediate results are cached. Re-running skips already-completed work.
Use `--resume` (default) to continue interrupted runs.

## Phase 3 dynamics training

```bash
# Train f_theta and g_theta on MPNet embeddings (requires Phase-2 dataset)
python scripts/train_dynamics.py --embedding mpnet --epochs 200

# Dry-run (validates config and prints model info)
python scripts/train_dynamics.py --embedding mpnet --dry-run
```

Architecture:
```
[u_k ; x_{k-1}]   (R^1536)
       |
    f_theta        (1536→512→512→768, ReLU)
       |
      x_k          (R^768)
       |
[u_k ; x_k]       (R^1536)
       |
    g_theta        (1536→512→512→768, ReLU)
       |
    z_hat_k        (R^768)
```

Outputs: `checkpoints/dynamics_mpnet/`, `runs/dynamics_mpnet/`

## Phase 4 NBF training

```bash
# Train predictor with frozen dynamics (default)
python scripts/train_nbf.py --embedding mpnet --freeze-dynamics --epochs 200

# Joint training mode
python scripts/train_nbf.py --embedding mpnet --joint --epochs 200

# Dry-run
python scripts/train_nbf.py --dry-run

# Loss weight ablation (Fig. 8 style)
python scripts/ablate_loss_weights.py
```

Architecture:
```
[x_{k-1}; u_k]   (R^1536)
       |
   SafetyPredictor  (1536→32→32→5)
       |
    5 logits
       |
    softmax
       |
    p(y)
       |
  Eq. (5)
       |
    h(x, u)        (R^1)
```

Loss: `L_total = λ_dyn·L_dyn + λ_CE·L_CE + λ_SS·L_SS + λ_SI·L_SI`

Outputs: `checkpoints/nbf_mpnet/`, `runs/nbf_mpnet/`

## Phase 6 — Multi-Turn Attack Suite (paper-strict)

> **Status (2026-08-27): paper-strict mode.**  The 5 non-adaptive
> attacks ship as **stubs that raise `NotImplementedError` on
> `next_query`** until the official implementations from the cited
> papers are pasted in.  The interface (`MultiTurnAttack`), the
> registry, the runner, the adaptive selector, the resume plumbing, and
> the CLI are all in place; the runner is verified end-to-end using a
> tiny local `MultiTurnAttack` subclass for tests.  When the official
> Crescendo / ActorAttack / Opposite-day / Acronym / RedQueen
> generators are pasted into the respective module bodies, the rest
> of the pipeline activates without further code changes.

The Phase 6 attack framework generates adversarial multi-turn conversations against a target LLM in two modes:

- **Bare LLM mode** — `Attack → Target LLM`
- **NBF-steered mode** — `Attack → SteeredLLMChat → Target LLM`

The same attack implementation drives both modes; only the runner changes.

### Available attacks

| Attack | Role per NBF paper | Origin | Phase 6 status |
| --- | --- | --- | --- |
| `crescendo` | Evaluation | Russinovich et al. 2024 | Stub (paste official code) |
| `actor_attack` | Evaluation | Ren et al. 2024 | Stub (paste official code) |
| `opposite_day` | Evaluation | Li et al. 2024b | Stub (paste official code) |
| `red_queen` | Unseen-attack probe (Fig. 6) | Jiang et al. 2024 | Stub (paste official code) |
| `acronym` | **Training only** | Li et al. 2024b | Stub (paste official code) |
| `adaptive` | Adaptive NBF attack (B.2 / Table 14) | NBF paper | ✅ Implemented |

All non-adaptive attacks are **paper-strict stubs**: the module files
contain the interface and a `Provenance:` header linking to the
official repository, and `next_query()` raises `NotImplementedError`
with an instruction pointing at the paper/repo to paste in.  Until the
official code is pasted, those attacks cannot be executed.

The adaptive attack selector is **fully implemented** because the
paper itself specifies its behavior: 3 candidate queries per turn, pick
argmax over `h(x, u)`.  The base attack defaults to `CrescendoAttack`;
once the Crescendo stub is replaced with the official implementation,
the adaptive attack is automatically paper-correct.

### Why stubs, not local reimplementations

The NBF paper says verbatim:

> *"we would like to refer the audience to these attack papers for more
> details regarding the logic and mechanism of query generation."*

So the NBF paper does NOT specify the attack algorithms.  Any
reimplementation on our side would be our interpretation of the cited
papers, not the paper under reproduction.  In paper-strict mode we
defer to the official implementations and surface the gap as loud
`NotImplementedError`s so the pipeline cannot silently run on
non-paper-faithful code.

### CLI (works against stubs — will execute once stubs are filled)

```bash
# Currently raises NotImplementedError on the 5 non-adaptive attacks.
# Once official code is pasted, the same commands work unchanged.
python scripts/run_attacks.py \
    --attack crescendo \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --max-turns 8 \
    --out data/processed/crescendo_eval.jsonl

python scripts/run_attacks.py \
    --attack crescendo \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/nbf_mpnet \
    --eta 5e-4 \
    --max-turns 8 \
    --out data/processed/crescendo_nbf.jsonl

# RedQueen at 5 turns (Fig. 6)
python scripts/run_attacks.py --attack red_queen \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --max-turns 5 \
    --out data/processed/redqueen_t5.jsonl

# Adaptive (works today, because the selector is paper-defined)
python scripts/run_attacks.py --attack adaptive \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/nbf_mpnet \
    --eta 0.0 \
    --out data/processed/crescendo_adaptive.jsonl

# Offline smoke (MockChatLLM)
python scripts/run_attacks.py --attack actor_attack \
    --goals data/raw/harmbench/raw_data.jsonl \
    --mock --max-turns 3 --limit 2 \
    --out /tmp/actor_smoke.jsonl
```

### Resume & idempotency

The runner keys each conversation by `(goal, attack, target_llm, eta, max_turns)`.  Completed conversations are skipped on re-run; partial runs are safe to interrupt and restart.  Atomic write semantics for the output JSONL.

### Refusal handling

The runner reuses `default_refusal_detector` from Phase 5.  Refusal responses are preserved verbatim in the transcript (`response` is set, `was_filtered=False`); whether a refusal counts as a "non-progress turn" per the B.1 protocol is delegated to Phase 7.

### Configuration

`configs/default.yaml` carries the `attacks:` section:

```yaml
attacks:
  max_turns: 8                    # paper: K_max = 8
  temperature: 0.7                # paper: all temps = 0.7
  adaptive_candidates: 3          # paper B.2 / Table 14: 3 candidates per turn
  # Default evaluation attacks per the NBF paper (Sec. 5.1 / B.1):
  default_evaluation_attacks:
    - actor_attack
    - crescendo
    - opposite_day
  # Acronym is training-only (paper Sec. 5.1) — excluded from evaluation.
  # RedQueen is the unseen-attack generalization probe only (Fig. 6) and is
  # invoked explicitly with --max-turns 1|3|4|5.
```

### Output format

Each line of the output JSONL is a valid Phase 1 `Conversation`:

```json
{
  "goal": "...",
  "attack_method": "crescendo",
  "target_llm": "gpt-3.5-turbo-0125",
  "max_turns": 8,
  "turns": [
    {
      "query": "...",
      "response": "...",
      "query_embedding": null,
      "response_embedding": null,
      "judge_score": null,
      "was_filtered": false
    }
  ]
}
```

`judge_score` is left as `null` — Phase 7 fills it in via the GPT-4o judge.

### Handoff to Phase 7

Phase 7 consumes `data/processed/<attack>_<defense>.jsonl` (one
Phase 1 `Conversation` per line).  No Phase 6 change is required for
Phase 7 to start.  Once the official attack implementations are
pasted in, Phase 7 will be able to compute ASR per attack / per defense
variant, over-refusal on benign corpora, and MMLU/MTBench helpfulness.

## Phase 7 — Evaluation Metrics

Implements every metric reported in the paper:

| Metric | Paper ref | Phase 7 status |
| --- | --- | --- |
| **ASR** (per-behavior) | Sec. 5.1 | ✅ GPT-4o judge, cached, stratified by attack |
| **MMLU** helpfulness | B.1 | ✅ bare + steered (h>0 -> wrong) |
| **MTBench** helpfulness | B.1 | ✅ bare + steered (h>0 -> refusal string) |
| **over-refusal** | B.1 | ✅ XSTest / JBB / PHTest, shared detector |
| **guardrail F1** | Sec. 5.1, Table 4 | ✅ NBF + OpenAI Moderation + ShieldGemma + LLaMA-Guard |
| **reporting** | Tables 1–15 | ✅ JSONL + Markdown + LaTeX renderers |

### ASR — Attack Success Rate

**PAPER-SPECIFIED:**
- Denominator = number of *behaviors* (conversations), NOT turns.
- 8-turn successful conversation = 1 behavior.
- GPT-4o judge (Sec. 5.1).
- Per-attack stratified reporting.

**NOT SPECIFIED IN THE PAPER:**
- The verbatim judge prompt.  The paper cites Ren et al. (2024) and
  Qi et al. (2023) as the source of its judging protocol but does
  not publish the prompt text.  The implementation stores a minimal
  local default at `configs/judge_prompts/asr_judge.txt` with full
  provenance metadata; swap in the verified prompt body before
  reproducing paper numbers.  The prompt's content hash is part of
  every cache key, so different prompt versions never collide.

```bash
python scripts/judge_asr.py \
    --corpus data/processed/crescendo_eval.jsonl \
    --judge-model gpt-4o-2024-08-06 \
    --judge-prompt configs/judge_prompts/asr_judge.txt \
    --out results/asr/crescendo.jsonl
```

Re-runs use the cached verdicts — already-judged conversations are
not re-billed.  A sidecar `*.results.jsonl` with `EvaluationResult`
rows is written for the Phase 9 reporter.

### MMLU

**PAPER-SPECIFIED:**
- B.1 post-filtering rule: if `h > 0`, the answer counts as WRONG
  regardless of the model's original answer.
- Reuses Phase 5's `mmlu_filter_answer` helper; no duplication.

**NOT SPECIFIED IN THE PAPER:**
- Dataset loader.  The implementation accepts any JSONL with fields
  `{question, choices, answer, context?}`; provide your own MMLU
  fixture or download via your preferred path.

```bash
# Bare LLM (no defense)
python scripts/eval_mmlu.py --model gpt-3.5-turbo-0125 \
    --dataset data/raw/mmlu.jsonl --out results/mmlu/bare.jsonl

# NBF-steered
python scripts/eval_mmlu.py --model gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/predictor_h.pt \
    --embedding mpnet --eta 5e-4 \
    --dataset data/raw/mmlu.jsonl --out results/mmlu/nbf.jsonl
```

### MTBench

**PAPER-SPECIFIED:**
- B.1 steered rule: when `h > 0`, replace the turn-2 answer with the
  EXACT refusal string from Phase 5 (`MTBENCH_REFUSAL_STRING`).  This
  implementation imports the canonical string from
  `defense.post_processing`, no copy.
- LLM judge, 1–10 score (Zheng et al. 2023 protocol).

**NOT SPECIFIED IN THE PAPER:**
- Verbatim judge prompt.  Local default in
  `evaluation/mtbench.py` (`DEFAULT_MTBENCH_TURN_PROMPT`); swap via
  `--judge-prompt` for paper-exact reproduction.

```bash
python scripts/eval_mtbench.py \
    --model gpt-3.5-turbo-0125 \
    --judge-model gpt-4o-2024-08-06 \
    --dataset data/raw/mtbench.jsonl \
    --out results/mtbench/bare.jsonl

# Steered
python scripts/eval_mtbench.py \
    --model gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/predictor_h.pt \
    --eta 5e-4 \
    --dataset data/raw/mtbench.jsonl \
    --out results/mtbench/nbf.jsonl
```

All judge calls are content-cached; re-runs do not re-bill.

### Over-refusal

**PAPER-SPECIFIED:**
- Datasets: XSTest, JailbreakBench-Benign, PHTest-Harmless.
- Metric: refused queries / total queries.

**NOT SPECIFIED IN THE PAPER:**
- The exact refusal-detection algorithm.  The implementation uses a
  documented regex-based heuristic shared across all three datasets
  and every model/bare/steered configuration
  (`evaluation/refusal_detector.is_refusal`).

```bash
python scripts/eval_over_refusal.py \
    --dataset xstest --data-path data/raw/xstest.jsonl \
    --model gpt-3.5-turbo-0125 \
    --out results/over_refusal/xstest_bare.jsonl

# Steered
python scripts/eval_over_refusal.py \
    --dataset xstest --data-path data/raw/xstest.jsonl \
    --model gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/predictor_h.pt \
    --eta 5e-4 \
    --out results/over_refusal/xstest_nbf.jsonl
```

### Guardrail F1 (Table 4)

**PAPER-SPECIFIED:**
- Datasets: HarmBench, AegisSafetyTest, WildGuardTest.
- NBF classification rule: `argmax(p) == class 1` (paper label) ->
  harmless; otherwise -> harmful.  Internally, the project's
  `SafetyPredictor` uses 0-based CE indices; the guard converts via
  `paper_class = ce_index + 1`.

**NOT SPECIFIED IN THE PAPER:**
- Single-prompt initial state.  Local default: `x_0 = zeros(768)`.
- Baseline inference details.  The 3 baseline guards are wrapped
  inference-only with no training.

```bash
python scripts/bench_guardrails.py \
    --nbf-checkpoints mpnet,distil \
    --datasets harmbench,aegis,wildguard \
    --include-openai --include-shieldgemma --include-llama-guard \
    --out results/guardrail_f1.jsonl
```

The CLI emits a Table-4-style Markdown summary, a per-dataset
per-guard JSONL of verdicts, and a sidecar `*.results.jsonl` with
`EvaluationResult` rows.

### Reporting

All metrics emit `EvaluationResult` rows; the renderers
(`to_markdown_table`, `to_latex_table`) compute best/runner-up
**dynamically** from the result data — no hard-coded rankings.  Use
the `--out` JSONL of any CLI to feed Phase 9's aggregator.

### Caching

Every paid LLM judge call (ASR, MTBench) is routed through
`JudgeCache`.  Cache keys are SHA-256 over a JSON-serialized payload
that includes the prompt version, the model id, and the full
input content.  Re-runs never re-bill identical requests.

### Tests

`pytest tests/test_evaluation.py -q` covers the ASR denominator
contract (2/3 synthetic), 8-turns = 1 behavior, F1 edge cases
(perfect / all-positive / all-negative / zero TP / empty), refusal
detector (clear / normal / empty / mixed-case / whitespace), MMLU
post-filter (correct+≤0 stays correct, correct+>0 becomes wrong,
wrong+>0 stays wrong), MTBench canonical refusal string, the
guard interface (3 baseline mocks returning `"harmful"` /
`"harmless"`), and cache hit/miss behavior.  All tests are offline.

### Handoff to Phase 8 (baselines)

Phase 8 wraps any target LLM with a defense variant and re-runs the
Phase 6 attack + Phase 7 evaluation pipeline.  No metric
re-implementation: the baselines registry reuses
`SteeredLLMChat` for NBF and produces `ChatLLM`-compatible wrappers
for `original`, `system_prompt`, `lora_sft`, `lora_dpo`, `lora_kto`.

### Handoff to Phase 9 (experiments)

Phase 9 reads the JSONL of `EvaluationResult` rows (sidecars emitted
by every Phase 7 CLI) and composes the paper's tables and figures.
No metric re-implementation: the orchestrator only calls
`evaluate_asr`, `evaluate_mmlu`, `evaluate_mtbench`,
`evaluate_over_refusal`, `evaluate_guard`.

## Phase 8 — Defense Baselines

All defense variants are exposed through a single `build_defense`
registry that returns a `ChatLLM`-compatible target.  Phase 6
attacks and Phase 7 metrics are unaware of which variant is in use.

| Variant | What it does | Paper status |
| --- | --- | --- |
| `original` | pass-through | trivially defined |
| `system_prompt` | prepends Llama-2 safety prompt | provenance documented; not paper-verified |
| `lora_sft` | base model + LoRA SFT adapter | lr=2e-4 / epochs=3 per B.1; rest are LLaMA-Factory defaults |
| `lora_dpo` | base model + LoRA DPO adapter | not specified in the NBF paper |
| `lora_kto` | base model + LoRA KTO adapter | not specified in the NBF paper |
| `guardbound` | delegates to Phase 5 `SteeredLLMChat` | Phase 5 |

### Original (`original`)
Identity wrapper.  No safety prompt, no fine-tuning, no NBF.

### System-prompt baseline (`system_prompt`)
Prepends the canonical Llama-2-Chat safety prompt to every
conversation.  The prompt text is in
`src/guardbound/baselines/prompts/llama2_safety_system.txt` with
a provenance block at the bottom documenting that the body was
taken from the public Meta / Hugging Face Llama-2 release and has
**NOT been verified against the exact source the NBF paper authors
used**.  For paper-exact reproduction, replace the body with the
verbatim original; the prompt's SHA-256 prefix is included in every
run manifest so different prompt bodies never collide.

The wrapper is **idempotent** (does not add the system prompt twice
if the caller already supplied it), **non-mutating** (does not
modify caller-owned message lists), and **multi-turn safe**.

### LoRA SFT (`lora_sft`)
Paper-specified hyperparameters:

    learning_rate = 2e-4
    num_train_epochs = 3

Target models: **Llama-3-8B-Instruct** and **Phi-4**.  All other
LoRA / batching / scheduler / precision values are LLaMA-Factory
defaults and are recorded in the run manifest as "Not specified in
the NBF paper — LLaMA-Factory default."

The SFT dataset is built by
`src/guardbound/baselines/build_sft_data.py`: same training
queries as Phase 2, with the jailbreaking responses **replaced**
by the verified safety-aligned responses from the Ren et al.
(2024) release.  If the alignment file is missing, the builder
**fails with a clear FileNotFoundError** — it never fabricates a
local substitute.

### LoRA DPO / LoRA KTO (`lora_dpo`, `lora_kto`)
**Not specified in the NBF paper.**  The implementations reuse the
SFT alignment data and the Phase 2 corpus:

    chosen   = verified Ren et al. (2024) safe response
    rejected = original Phase 2 jailbreak response

Construction is documented in the manifest's `provenance` block.

### NBF steering (`guardbound`)
The registry delegates to the existing Phase 5 `SteeredLLMChat`.
No new steering method is implemented in Phase 8.

### HuggingFace serving

`src/guardbound/baselines/serving.py` provides `HFChatLLM`, a
lazy-loading HuggingFace backend that supports base models and
LoRA adapters (via `peft`).  Heavy imports (`torch`,
`transformers`, `peft`) only happen on first use.  When
`NBF_MOCK=1` is set (or when `torch` is unavailable), the
factory falls back to `MockChatLLM` — never silently substituting
another real model.

### Dataset requirements

- Phase 2 training corpus: `data/processed/conversations.jsonl`
  (or whatever path you pass).
- **Ren et al. 2024 safety-aligned responses**: required for SFT /
  DPO / KTO.  Provide a JSONL where each line has a goal key
  (`goal_id` / `behavior` / `prompt`) and a response key
  (`response` / `rejective_response` / `safe_response`).  When
  this file is missing, every training script exits with a
  FileNotFoundError that names the expected path and provenance
  requirements.

### Manifests

Every training script (SFT / DPO / KTO) writes a JSON manifest that
records:

- `paper_specified` — values explicitly in the NBF paper (lr, epochs).
- `not_specified_in_paper` — every LLaMA-Factory default
  (`lora_rank`, `lora_alpha`, `per_device_train_batch_size`, …).
- `dataset.hash`, `dataset.path`, `dataset.version`.
- `base_model`, `base_model_revision`.
- `seed`, `precision`, `quantization`, `command_line`, `git_commit`.
- `provenance.chosen`, `provenance.rejected`, `provenance.replacement_responses`.

### CLI

```bash
# Build SFT data (one-off, paper-strict, never fabricates responses)
python -c "from guardbound.baselines.build_sft_data import build_sft_dataset; \
    build_sft_dataset('data/processed/phase2.jsonl', \
    'data/processed/ren2024_safety_responses.jsonl', \
    'data/processed/lora_sft.jsonl')"

# Train (dry-run writes the manifest, exits without invoking LLaMA-Factory)
python scripts/train_lora_sft.py \
    --model llama-3-8b-instruct \
    --sft-data data/processed/lora_sft.jsonl \
    --dry-run --output-dir checkpoints/lora_sft/llama-3-8b-instruct

# Train (real; requires LLaMA-Factory + GPU)
python scripts/train_lora_sft.py \
    --model llama-3-8b-instruct \
    --sft-data data/processed/lora_sft.jsonl

# DPO / KTO follow the same pattern with --dpo-data / --kto-data.

# Evaluate any defense variant end-to-end (mocked / real / steered)
python scripts/run_baseline_evals.py \
    --variant original \
    --model gpt-3.5-turbo-0125 \
    --attacks crescendo,actor_attack,opposite_day \
    --mock --limit 2 \
    --out results/baselines/original.jsonl

python scripts/run_baseline_evals.py \
    --variant system_prompt \
    --model gpt-3.5-turbo-0125 \
    --attacks crescendo \
    --mock --limit 2 \
    --out results/baselines/system_prompt.jsonl

python scripts/run_baseline_evals.py \
    --variant guardbound \
    --model gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/predictor_h.pt \
    --embedding mpnet --eta 5e-4 \
    --attacks crescendo \
    --mock --limit 2 \
    --out results/baselines/nbf.jsonl

# SFT variant (LoRA adapter) — mock mode bypasses the adapter load
python scripts/run_baseline_evals.py \
    --variant lora_sft \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --checkpoint checkpoints/lora_sft/llama-3-8b-instruct \
    --attacks crescendo --mock --limit 2 \
    --out results/baselines/lora_sft.jsonl
```

All baseline evaluation rows use the Phase 7 `EvaluationResult`
schema; results sidecars (`*.results.jsonl`) accumulate the
per-(variant, model, attack, eta) rows for Phase 9.

### Tests

`pytest tests/test_baselines.py -q` → **25 passed.**  Coverage:

- Registry completeness (all 6 variants).
- `original` does not modify requests.
- `system_prompt` adds the prompt, is idempotent, does not mutate
  caller lists, preserves multi-turn ordering.
- SFT data builder schema (Llama-Factory `messages`), explicit
  error on missing safety-response file, no fabrication when
  safety response is absent.
- DPO / KTO builders produce the documented `prompt` / `chosen` /
  `rejected` (DPO) and `prompt` / `completion` / `label` (KTO)
  schemas.
- Manifest captures paper-specified + non-paper-specified fields.
- Serving falls back to `MockChatLLM` under `NBF_MOCK=1` or when
  torch is unavailable.
- LoRA variants require `checkpoint` and `base_model` with clear
  error messages.
- NBF registry delegates to Phase 5 `SteeredLLMChat`.
- End-to-end eval loop produces a Phase 7 `EvaluationResult`
  sidecar with the correct schema.



## Phase 9 — Experiments, Ablations, and Visualization

Phase 9 is **orchestration and analysis only**.  It does not
reimplement any attack, metric, training, or steering logic; it
composes the existing Phase 1-8 entry points and produces
paper-style tables, figures, and reports from the resulting
`EvaluationResult` rows.

### Experiment registry

`src/guardbound/experiments/registry.py` declares all ten
experiments:

| ID  | Title                                                    | Paper reference     | Optional |
| --- | -------------------------------------------------------- | ------------------- | -------- |
| E1  | Main results (ASR, MTBench, XSTest)                      | Tables 1, 2, 13     | no       |
| E2  | Threshold sweep + Pareto                                 | Fig. 5, Tables 8, 9, 15 | no       |
| E3  | Loss ablations (L_SS, L_SI, λ-grid)                       | Table 6, Fig. 8     | no       |
| E4  | κ ablation (2, 3, 4)                                     | Table 7             | no       |
| E5  | Embedding ablation (MPNet vs DistilRoBERTa)              | Tables 4, 6         | no       |
| E6  | Generalization (LOO + RedQueen + single-turn)            | Tables 5, 11, 12; Fig. 6 | yes   |
| E7  | Adaptive attack                                          | Table 14            | no       |
| E8  | Over-refusal vs alignment baselines                       | Table 13            | no       |
| E9  | PCA state trajectory visualization                        | Figs. 9, 10, 11    | no       |
| E10 | Optional latest models (GPT-5, Claude Sonnet 4.5)        | Table 10            | **yes**  |

Each spec is an immutable `ExperimentSpec` declaring its required
phases, models, defense variants, attacks, datasets, metrics,
eta values, paper reference, and output directory.  E10 is the
only optional experiment; the others are part of the core
reproduction matrix.

### Runner

`ExperimentRunner` materializes every spec into a flat list of
`ExperimentCell`s and runs them through a caller-supplied
executor function.  The runner is:

- **Resumable** — completed cells are skipped on subsequent runs
  (manifest-based).
- **Idempotent** — running the same experiment twice does not
  duplicate logical result rows.
- **Lifecycle-aware** — every cell is tagged with
  `PENDING / RUNNING / COMPLETED / FAILED / SKIPPED /
  UNAVAILABLE / UNVERIFIED` and written to the per-experiment
  manifest.
- **API-aware** — closed-source models are only available when
  their API-key env var is set, unless `--smoke` is enabled.
- **Smoke-friendly** — every cell can be executed against mocked
  LLMs / judges; smoke results are tagged so they never mix with
  real experiment data.

### CLI

```bash
# Plan a single experiment (no executor -> everything PENDING)
python scripts/run_experiments.py --exp E1 --dry-run

# Smoke-mode: every cell is PENDING unless an executor is provided.
# The runner records lifecycle state; downstream CLIs (e.g. eval_mmlu)
# can be invoked by passing your own executor.
python scripts/run_experiments.py --exp ALL --smoke --output-dir results/experiments

# Master runner: cheap-to-expensive ordered stages
python scripts/run_all.py --smoke
# (stages: validate, smoke, aggregate, sweeps, ablations, retrain,
#  attacks, guardrails, pca, tables, figures, findings, report)

# Final report (always runnable; empty result rows in --smoke)
python scripts/build_final_report.py \
    --output-dir results/experiments \
    --report-dir results/final \
    --smoke

# Phase 9 retrain wrapper (writes a manifest, --dry-run skips training)
python -m guardbound.experiments.retrain \
    --drop-loss ss --kappa 2 \
    --experiment-id E3 \
    --output-dir results/experiments/E3/retrain \
    --dry-run
```

### Analysis

`src/guardbound/analysis/` provides:

- `pareto.compute_pareto_front` / `dominates` — generic
  weak-Pareto analysis over `(ASR, helpfulness)`.  Handles
  duplicates, dominated points, equal values, single points,
  and empty input.
- `aggregator.aggregate_by`, `stratify_by_attack`,
  `stratify_by_model` — group-by helpers for `EvaluationResult`
  rows.
- `tables.pivot_results` + `to_pivot_markdown` — wide-form
  `model × defense` pivots with dynamically computed best and
  runner-up values.  Markdown + LaTeX renderers (reused from
  Phase 7).
- `plots` — `plot_pareto_asr_vs_helpfulness`,
  `plot_redqueen_asr_vs_turns`, `plot_threshold_curve`,
  `plot_pca_trajectories` (with sklearn / numpy SVD fallback).
  Every figure writes a sidecar JSON with the experiment
  metadata.
- `findings.check_claims` — programmatic verification of the
  paper's qualitative claims.  Returns one of
  `PASS / FAIL / DIRECTIONALLY-CONSISTENT / UNVERIFIED`.  The
  verifier **never fabricates data**: missing rows are reported
  as `UNVERIFIED` and never mapped to `PASS`.

### Tests

`pytest tests/test_experiments.py -q` → **38 passed.**

Coverage:

- Registry completeness (E1-E10), E10 is optional, validation
  reports no issues.
- Pareto edge cases (empty / one / duplicates / dominated /
  equal values / non-dominated).
- Aggregation (group-by, mean ignoring None, attack / model
  stratification).
- Pivot table generation + best/runner-up highlighting.
- Findings: `UNVERIFIED` when no data; `PASS` for matching
  direction; `FAIL` for contradicted; `UNVERIFIED` never
  forced to `PASS`.
- Runner: plan materialization, dry-run records PENDING,
  resume skips completed cells.
- Smoke `run_all` produces rows for every non-optional
  experiment.
- `build_final_report --smoke` produces a full report
  (findings.md, final_report.md, final_report.json) without
  real data.
- Plot helpers all execute against synthetic data without
  network access.

### Reproduction status

`REPRODUCTION_NOTES.md` records every paper-specified value,
every "Not specified in the paper" decision, every local
default, every implementation deviation, and every
**UNVERIFIED** component.  Real reproduction of the paper's
tables and figures remains pending until the user pastes the
official Phase 6 attack implementations, provides the Ren
et al. safety-aligned response dataset, supplies trained NBF
checkpoints, and runs the experiments with real model access.

The Phase 9 infrastructure is **complete and tested offline**;
**real experiments are pending** — see `REPRODUCTION_NOTES.md`
for the explicit gap list.


Everything in `configs/default.yaml` marked `[paper]` comes from Sec. 5.1 / Appendix B.1 of the paper
(dims 768/768, K_max=8, temperature 0.7, λ=(1,1,100,100), Adam lr 1e-4/1e-3 × 200 epochs, η_train=0,
κ=3, unsafe ⇔ judge score 5, both embedding models, pinned model versions, η grid incl. recommended 5e-4).

Values marked `NOT in paper — default` (batch size 64, seed 0, train/val split 0.9, gradient clipping disabled, no scheduler) exist only because
the paper does not specify them; each is flagged in code/config.

### Phase 3 parameter sources

| Parameter | Value | Source |
|---|---|---|
| Embedding dimension | 768 | Paper |
| State dimension | 768 | Paper |
| MLP architecture | 1536→512→512→768 | Paper |
| Output activation | Linear | Not specified — default for regression |
| Optimizer | Adam | Paper |
| Learning rate | 1e-4 | Paper |
| Epochs | 200 | Paper |
| Batch size | 64 | Not specified |
| Weight decay | 0 | Not specified |
| Scheduler | None | Not specified |
| Seed | 0 | Not specified |
| Gradient clipping | Disabled | Not specified |

## Repository layout

See the annotated tree in `docs/IMPLEMENTATION_ROADMAP.md` (Part 2) — every subpackage of
`src/guardbound/` corresponds to exactly one phase, so each milestone can be implemented,
tested, debugged, and validated independently.
