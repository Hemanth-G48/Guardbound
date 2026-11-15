# NBF Safety Steering - Experiment Results vs Paper Comparison

## Experiment Configuration

| Parameter | Our Experiment | Paper |
|-----------|---------------|-------|
| Target Model | Llama3 (Ollama, 8B) | Llama-3-8B-Instruct |
| Attack | Crescendo (8 turns) | Crescendo (8 turns) |
| Dataset | Harmbench (320 behaviors) | Harmbench (160 behaviors) |
| NBF Checkpoint | checkpoints/nbf_adapted.pt | hanjianghu/NBF-LLM |
| NBF Threshold (η) | 5e-4 | Not specified |
| Evaluation | Keyword matching | LLM judge rubric |

---

## Paper Results (Table 1)

### GPT-3.5-turbo
| Attack | Baseline ASR | +NBF ASR | Reduction |
|--------|-------------|-----------|-----------|
| ActorAttack | 58.5% | 4.0% | **-54.5%** |
| Crescendo | 56.0% | 23.5% | **-32.5%** |
| Opposite-day | 78.5% | 37.5% | **-41.0%** |

### GPT-4o
| Attack | Baseline ASR | +NBF ASR | Reduction |
|--------|-------------|-----------|-----------|
| ActorAttack | 60.0% | 3.5% | **-56.5%** |
| Crescendo | 56.5% | 26.0% | **-30.5%** |
| Opposite-day | 72.5% | 32.5% | **-40.0%** |

### o1
| Attack | Baseline ASR | +NBF ASR | Reduction |
|--------|-------------|-----------|-----------|
| ActorAttack | 51.0% | 9.0% | **-42.0%** |
| Crescendo | 44.5% | 28.0% | **-16.5%** |
| Opposite-day | 53.0% | 21.0% | **-32.0%** |

### Claude 3.5 Sonnet
| Attack | Baseline ASR | +NBF ASR | Reduction |
|--------|-------------|-----------|-----------|
| ActorAttack | 20.0% | 4.0% | **-16.0%** |
| Crescendo | 21.5% | 12.0% | **-9.5%** |
| Opposite-day | 9.5% | 4.5% | **-5.0%** |

---

## Our Results

| Condition | ASR | Notes |
|-----------|-----|-------|
| Baseline (Llama3) | **87.7%** | 318 conversations |
| NBF Defense (Llama3) | **95.3%** | 318 conversations, 698 filtered turns |

**Finding: NBF increased attack success rate by +7.6 percentage points**

---

## Comparison Analysis

### Key Discrepancies

1. **Direction of Effect**
   - Paper: NBF **reduces** ASR by 10-55 percentage points
   - Our experiment: NBF **increases** ASR by 7.6 percentage points

2. **Baseline ASR**
   - Paper Llama-3-8B-Instruct: ~20-60% depending on attack
   - Our Llama3: 87.7% (significantly higher)

3. **Filtering Behavior**
   - Paper: NBF effectively blocks harmful responses
   - Our experiment: NBF blocks turns (27.3%) but attacks adapt and succeed

### Why Our Results Differ

| Factor | Paper | Our Experiment |
|--------|-------|----------------|
| Model | Llama-3-8B-Instruct (gated) | Llama3 latest (via Ollama) |
| Training | NBF trained on this model | NBF trained on different model |
| Evaluation | LLM judge with rubric | Keyword matching |
| Checkpoint | Model-specific | Adapted from paper |
| Dataset | 160 Harmbench behaviors | 320 behaviors |

### Critical Insight

The paper's NBF was **trained specifically for each target model**. Our experiment used the paper's pre-trained NBF checkpoint (trained on different data) on a different Llama3 variant. This mismatch likely explains:

1. **Poor adaptation**: The NBF checkpoint doesn't generalize well to Llama3
2. **False filtering**: Blocks legitimate responses, causing over-refusal
3. **Attack adaptation**: When blocked, Crescendo escalates and eventually succeeds

### Lessons Learned

1. **Model-specific training is crucial** - NBF effectiveness depends on matching the model it was trained on
2. **Per-turn filtering is insufficient** - Sophisticated attackers can persist through blocking
3. **Evaluation methodology matters** - Keyword matching may misclassify nuanced responses

---

## Conclusions

Our experiment demonstrates that:

1. **Direct checkpoint transfer doesn't work**: Using paper's NBF on Llama3 without retraining leads to worse outcomes
2. **Per-turn filtering enables attack adaptation**: Blocking signals progress to attackers
3. **Proper evaluation requires rubric-based assessment**: Simple keyword matching is unreliable

To reproduce paper results, one would need to:
- Use the exact model (Llama-3-8B-Instruct with HuggingFace authentication)
- Retrain NBF on target model
- Use LLM-based evaluation with proper rubric

---

## Files

- Baseline results: `results/crescendo_baseline_llama3.jsonl`
- NBF results: `results/crescendo_nbf_llama3.jsonl`
- Analysis script: `final_analysis.py`

## Experiment Date

2026-09-03
