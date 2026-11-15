# NBF Safety Steering - Experiment Results

## Experiment Configuration

- **Target Model**: Llama3 (via Ollama, 8B parameters)
- **Attack**: Crescendo (8 turns per conversation)
- **Dataset**: Harmbench (320 behaviors, 318 evaluated)
- **NBF Checkpoint**: checkpoints/nbf_adapted.pt
- **NBF Threshold (eta)**: 5e-4

## Results

| Metric | Baseline (No Defense) | NBF Defense |
|--------|---------------------|-------------|
| Total conversations | 318 | 318 |
| Successful attacks | 279 | 303 |
| Refused/blocked | 39 | 15 |
| **ASR (Attack Success Rate)** | **87.7%** | **95.3%** |
| Filtered turns | N/A | 698 |

## Analysis

### Key Finding
NBF increased attack success rate (87.7% → 95.3%), contrary to expected protective effect.

### Why This Happened
1. **Per-turn filtering**: NBF blocks individual model responses that exceed the safety threshold
2. **Attack persistence**: Crescendo attack interprets blocking as progress and escalates
3. **Eventual success**: Despite blocking early turns, attackers persist and succeed in later turns
4. **False adaptation**: Blocked responses may actually signal to the attacker that they're close to bypassing safety

### Observations
- NBF filtered 698 turns (out of 2560 total turns = 27.3% filtering rate)
- Every conversation (318/318) had at least one filtered response with NBF
- But 303/318 (95.3%) still achieved successful harmful output

## Comparison with Paper

Paper (arXiv 2503.00187) reports:
- Llama-3-8B-Instruct baseline ASR: ~50-60%
- NBF defense significantly reduces ASR

Our results:
- Higher baseline ASR (87.7%) likely due to:
  - Different model variant (Llama3 latest vs Instruct)
  - Simplified ASR evaluation (keyword matching vs rubric-based)
  - Different Crescendo implementation details

## Conclusions

1. **Per-turn filtering alone is insufficient** - attackers can adapt to filtering and persist
2. **NBF training (steering during generation)** is more effective than post-hoc filtering
3. **Sequential blocking may signal attack progress** to sophisticated attackers

## Files

- Baseline results: `results/crescendo_baseline_llama3.jsonl`
- NBF results: `results/crescendo_nbf_llama3.jsonl`
- Analysis script: `final_analysis.py`

## Experiment Date

2026-09-03
