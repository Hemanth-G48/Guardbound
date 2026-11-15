# Guardbound

A Neural Barrier Function (NBF) that proactively filters harmful queries in multi-turn LLM dialogues via a control-theoretic Q-filter.

## What it does

Guardbound intercepts queries before they reach the target LLM. Given the conversation history and a new query, it predicts whether allowing this turn could lead to a harmful outcome. If the safety score `h(x, u) + η ≥ 0`, the query is filtered and the LLM is never called.

**Pipeline:**
- Offline: Build attack datasets → Train dynamics model (f_θ, g_θ) → Train NBF predictor (h)
- Online: Query → Embed → Predict → Filter if unsafe

## Setup

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest tests/ -q
python scripts/check_env.py
```

Set API keys as needed: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

### Bundled reference assets (Git LFS)

Large `.pt` and `.npz` files are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

The repository bundles `nbf_original_stuff/`, `NBF-LLM-Dialogue-Embedding/`,
`data/raw/`, `data/processed/`, `checkpoints/`, `results/`, `runs/`, the paper
PDF, and the small `data/utility/` MTBench/XSTest fixtures. Local model weights
under `models/` are still external and must be copied or downloaded separately.
`.env` remains local; copy `.env.example` when a paid cloud run is explicitly
authorized.

## Data pipeline

```bash
# Generate attack conversations
python scripts/build_conversations.py --attack acronym --num-goals 1000
python scripts/build_conversations.py --attack crescendo --num-goals 1000
python scripts/build_conversations.py --attack opposite_day --num-goals 1000
python scripts/build_conversations.py --attack actor_attack --num-goals 2327

# Judge with GPT-4o
python scripts/judge_conversations.py --input data/processed/conversations/acronym.jsonl

# Extract embeddings
python scripts/embed_conversations.py --input data/processed/conversations_labeled.jsonl

# Build tensor datasets
python scripts/build_datasets.py --input data/processed/conversations_embedded.jsonl

# Dry-run (no API calls)
python scripts/build_conversations.py --attack acronym --num-goals 2 --dry-run
```

## Train dynamics model

```bash
python scripts/train_dynamics.py --embedding mpnet --epochs 200
python scripts/train_dynamics.py --embedding mpnet --dry-run
```

Outputs: `checkpoints/dynamics_mpnet/`, `runs/dynamics_mpnet/`

## Train NBF predictor

```bash
# Freeze dynamics (default)
python scripts/train_nbf.py --embedding mpnet --freeze-dynamics --epochs 200

# Joint training
python scripts/train_nbf.py --embedding mpnet --joint --epochs 200

# Loss weight ablation
python scripts/ablate_loss_weights.py
```

Outputs: `checkpoints/nbf_mpnet/`, `runs/nbf_mpnet/`

## Run attacks

```bash
python scripts/run_attacks.py \
    --attack crescendo \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --max-turns 8 \
    --out data/processed/crescendo_eval.jsonl

# With NBF steering
python scripts/run_attacks.py \
    --attack crescendo \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/nbf_mpnet \
    --eta 5e-4 \
    --max-turns 8 \
    --out data/processed/crescendo_nbf.jsonl

# Adaptive attack
python scripts/run_attacks.py --attack adaptive \
    --goals data/raw/harmbench/raw_data.jsonl \
    --target gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/nbf_mpnet \
    --eta 0.0 \
    --out data/processed/crescendo_adaptive.jsonl
```

## Evaluation

```bash
# Attack Success Rate
python scripts/judge_asr.py \
    --corpus data/processed/crescendo_eval.jsonl \
    --judge-model gpt-4o-2024-08-06 \
    --judge-prompt configs/judge_prompts/asr_judge.txt \
    --out results/asr/crescendo.jsonl

# MMLU
python scripts/eval_mmlu.py --model gpt-3.5-turbo-0125 \
    --dataset data/raw/mmlu.jsonl --out results/mmlu/bare.jsonl

# MTBench
python scripts/eval_mtbench.py \
    --model gpt-3.5-turbo-0125 \
    --judge-model gpt-4o-2024-08-06 \
    --dataset data/raw/mtbench.jsonl \
    --out results/mtbench/bare.jsonl

# Over-refusal
python scripts/eval_over_refusal.py \
    --dataset xstest --data-path data/raw/xstest.jsonl \
    --model gpt-3.5-turbo-0125 \
    --out results/over_refusal/xstest_bare.jsonl

# Guardrail F1
python scripts/bench_guardrails.py \
    --nbf-checkpoints mpnet,distil \
    --datasets harmbench,aegis,wildguard \
    --include-openai --include-shieldgemma --include-llama-guard \
    --out results/guardrail_f1.jsonl
```

## Baselines

```bash
# Build SFT data
python -c "from guardbound.baselines.build_sft_data import build_sft_dataset; \
    build_sft_dataset('data/processed/phase2.jsonl', \
    'data/processed/ren2024_safety_responses.jsonl', \
    'data/processed/lora_sft.jsonl')"

# Train LoRA SFT
python scripts/train_lora_sft.py \
    --model llama-3-8b-instruct \
    --sft-data data/processed/lora_sft.jsonl

# Evaluate defenses
python scripts/run_baseline_evals.py \
    --variant guardbound \
    --model gpt-3.5-turbo-0125 \
    --barrier-checkpoint checkpoints/predictor_h.pt \
    --embedding mpnet --eta 5e-4 \
    --attacks crescendo \
    --mock --limit 2 \
    --out results/baselines/nbf.jsonl
```

## Experiments

```bash
# Single experiment (dry-run)
python scripts/run_experiments.py --exp E1 --dry-run

# Smoke mode
python scripts/run_experiments.py --exp ALL --smoke --output-dir results/experiments

# Master runner
python scripts/run_all.py --smoke

# Final report
python scripts/build_final_report.py \
    --output-dir results/experiments \
    --report-dir results/final \
    --smoke
```

## Repository layout

```
src/guardbound/
  config.py, schemas.py       # Configuration and data schemas
  llm/                        # LLM client wrappers
  embeddings.py               # Sentence embedding utilities
  data/, attacks/             # Data pipeline and attack implementations
  models/
    dynamics.py               # f_θ, g_θ dynamics MLPs
    predictor.py              # Safety predictor h
  training/
    losses.py, diagnostics.py # Training utilities
    train_dynamics.py         # Dynamics training script
    train_nbf.py              # NBF training script
  defense/
    steered_chat.py           # Q-filter runtime
    context_init.py           # Initial state handling
    post_processing.py        # Refusal strings
    evaluation_runner.py      # Evaluation harness
  evaluation/                 # ASR, MMLU, MTBench, over-refusal, guardrail F1
  baselines/                   # Defense baselines (original, system_prompt, LoRA variants)
  experiments/                # E1-E10 experiment registry and runner
  analysis/                   # Pareto analysis, tables, plots, findings
scripts/
  check_env.py                # Validate configuration
  build_conversations.py      # Generate attack conversations
  judge_conversations.py      # Judge with GPT-4o
  embed_conversations.py      # Extract embeddings
  build_datasets.py           # Build tensor datasets
  train_dynamics.py           # Train dynamics model
  train_nbf.py                # Train NBF predictor
  run_attacks.py              # Run attack suite
  judge_asr.py                # Evaluate ASR
  eval_mmlu.py, eval_mtbench.py, eval_over_refusal.py
  bench_guardrails.py         # Guardrail F1 benchmark
  train_lora_sft.py, train_lora_dpo.py, train_lora_kto.py
  run_baseline_evals.py       # Evaluate defense baselines
  run_experiments.py          # Run experiments
  run_all.py                  # Master orchestration
  build_final_report.py       # Generate final report
```
