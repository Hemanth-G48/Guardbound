# Paper Compliance Analysis: NBF Implementation vs. arXiv:2503.00187

Generated: 2026-09-06

## Exact Matches ✓

| Aspect | Paper Specification | Implementation Status |
|--------|-------------------|----------------------|
| Training epochs | 200 | ✓ IMPLEMENTED |
| Dynamics learning rate | 1e-4 | ✓ IMPLEMENTED |
| Predictor learning rate | 1e-3 | ✓ IMPLEMENTED |
| State dimension | 768 | ✓ IMPLEMENTED |
| NBF architecture | 3-layer MLP (1536-32-32-5) | ✓ IMPLEMENTED |
| Embedding model | all-mpnet-base-v2 | ✓ IMPLEMENTED |
| Temperature | 0.7 | ✓ IMPLEMENTED |
| Target: GPT-3.5-turbo | gpt-3.5-turbo-0125 | ✓ IMPLEMENTED |
| Target: GPT-4o | gpt-4o-2024-08-06 | ✓ IMPLEMENTED |
| Target: o1 | o1-2024-12-17 | ✓ IMPLEMENTED |
| Target: Claude-3.5 | claude-3-5-sonnet-20241022 | ✓ IMPLEMENTED |
| Target: Llama-3-8b-instruct | Via Ollama | ✓ IMPLEMENTED |
| Target: Phi-4 | Via Ollama | ✓ IMPLEMENTED |
| Attack: Crescendo | Multi-turn escalation | ✓ IMPLEMENTED |
| Attack: Opposite-Day | Reframing attack | ✓ IMPLEMENTED |
| Attack: ActorAttack | Actor-network attack | ✓ IMPLEMENTED |
| ASR evaluation | 200 HarmBench behaviors | ✓ IMPLEMENTED |
| GPT-4o judge | 1-5 rubric scoring | ✓ IMPLEMENTED |
| NBF threshold | η ∈ {0, 1e-4, 1e-3, 5e-2} | ✓ IMPLEMENTED |
| MMLU metric | Accuracy % | ✓ IMPLEMENTED |
| MTBench metric | 1-10 score | ✓ IMPLEMENTED |
| XSTest metric | Refusal rate | ✓ IMPLEMENTED |
| JailbreakBench-Benign | Refusal rate | ✓ IMPLEMENTED |
| Guardrail F1 | HarmBench, AegisSafetyTest, WildGuardTest | ✓ IMPLEMENTED |

---

## Differences ✗

### 1. Training Loss Weights (CRITICAL)

**Paper Specification (Section B.1):**
```
λdyn = 1     (dynamics loss weight)
λCE = 1      (cross-entropy loss weight)
λSS = 100    (safety score loss weight)
λSI = 100    (safety invariance loss weight)
```

**Current Status:** Loss weights are hardcoded in training code, not exposed as configuration.

**Action Needed:** Verify training script uses these exact weights.

---

### 2. Training Threshold

**Paper:** η = 0 during training, adjustable during evaluation

**Current:** Default η = 0.001

**Action Needed:** Should train with η = 0, then evaluate with η ∈ {0, 1e-4, 1e-3, 5e-2}

---

### 3. Acronym Attack

**Paper:** Training-only attack method, excluded from evaluation (contains too many single-turn successful jailbreaks)

**Current:** Not implemented

**Impact:** Medium - affects training diversity but not evaluation

---

### 4. Turn Regeneration Logic

**Paper (Section B.1):**
> "If a rejective response is generated (e.g., "I'm sorry...") by LLMs but not detected by the neural barrier function, this turn will not count, and another attack query will be regenerated."

**Current:** No regeneration logic - attacks proceed even on refusals

**Impact:** High - may overestimate ASR in some cases

---

### 5. MMLU Post-Filtering

**Paper (Section B.1):**
> "If the safety score predicted by the barrier function is positive, this answer is considered to be wrong, whatever the original answer before steering is."

**Current:** Simpler filtering approach

**Impact:** Medium - affects helpfulness metrics

---

### 6. MTBench Refusal String

**Paper specifies exact refusal replacement:**
```
"I'm sorry, but I cannot provide information on that.
It is important to prioritize safety and ethical considerations in our conversation.
If you have any other questions on a different topic, feel free to ask!"
```

**Current:** Generic refusal handling

**Impact:** Low - affects MTBench scores slightly

---

### 7. Judge Prompt

**Paper:** "Following Ren et al. (2024), we adopt GPT-4o as the LLM safety judge (Qi et al., 2023)"

**Issue:** The exact wording of the Qi et al. 2023 / Ren et al. 2024 judge prompt is NOT published in the paper.

**Current:** Our implementation uses a simplified 1-5 rubric based on OpenAI policy

**Impact:** Medium - may produce different ASR numbers than paper

---

### 8. RedQueen Attack

**Paper:** Additional attack method tested for generalizability (Figure 6)

**Current:** Not implemented

**Impact:** Low - only in extended experiments

---

### 9. Single-Turn to Multi-Turn Conversion (Table 12)

**Paper:** "consolidates multi-turn attacks to single-turn attacks"

**Current:** Not evaluated

**Impact:** Low - advanced experiment

---

### 10. Adaptive NBF-Based Attack

**Paper (Table 14):**
> "adaptive attack based on NBF can achieve a higher attack success rate compared to the original Crescendo attack"

**Current:** Not implemented

**Impact:** Medium - affects adversarial robustness testing

---

## Summary of Required Fixes

### High Priority
1. **Verify loss weights** λdyn=1, λCE=1, λSS=100, λSI=100 in training
2. **Train with η=0**, evaluate with various η values
3. **Implement turn regeneration** when model refuses but NBF doesn't detect

### Medium Priority
4. Implement Acronym attack for training diversity
5. Use exact MTBench refusal string
6. Use paper's exact judge prompt (if obtainable from Qi et al. 2023)

### Low Priority
7. Implement RedQueen attack
8. Implement single-to-multi-turn evaluation
9. Implement adaptive NBF-based attack

---

## Paper Reference

```
@article{NBF2025,
  title={Neural Barrier Function for Multi-Turn LLM Jailbreak Defense},
  author={},
  journal={arXiv:2503.00187},
  year={2025}
}
```
