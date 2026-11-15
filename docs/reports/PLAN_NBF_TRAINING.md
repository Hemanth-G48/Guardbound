# Plan: Proper NBF Training & Evaluation

## Current Status

| Item | Status |
|------|--------|
| Conversations data | ✅ 5,327 conversations (~28K turns) |
| Embeddings | ✅ Pre-computed |
| Author pretrained NBF | ✅ Downloaded |
| Proper 1-5 labels | ❌ Missing (have binary 3/5 only) |
| Llama-3-8B running | ❓ Unknown |

## Recommended Plan

### Option A: Fast Path (Use Author's NBF for Labels) - RECOMMENDED NOW

**Time: ~30 minutes**

```bash
# Step 1: Generate pseudo-labels using author's pretrained NBF
python scripts/generate_nbf_pseudo_labels.py
```

This uses the author's already-trained NBF to score all turns. The NBF was trained on GPT-4 rubric-scored data, so its scores are a good proxy.

**Then:**
```bash
# Step 2: Build training dataset with proper labels
python scripts/build_nbf_dataset_from_pseudo_labels.py

# Step 3: Train dynamics
python -m guardbound.training.train_dynamics --config config/default.yaml

# Step 4: Train NBF
python -m guardbound.training.train_nbf --config config/default.yaml

# Step 5: Evaluate with Crescendo attacks
python scripts/evaluate_nbf_defense.py
```

### Option B: Better Path (Use Llama-3-8B for Labels) - IF API NOT AVAILABLE

**Time: ~1-2 hours**

```bash
# Check if Llama is running
curl http://localhost:8000/v1/models

# If running, label conversations
python scripts/label_conversations_with_llama.py

# If not running, start Llama first
# (Requires LM Studio or similar)
```

### Option C: Hybrid (NBF + Optional Llama Refinement)

1. Use NBF to label all turns (fast)
2. Flag uncertain predictions (low confidence)
3. Use Llama to re-score uncertain ones (slower, only for edge cases)

---

## Implementation Steps

### Step 1: Generate NBF Pseudo-Labels (Option A)

```bash
python scripts/generate_nbf_pseudo_labels.py
```

Expected output:
```
Score Distribution:
  Score 1: ~50% (safe refusals)
  Score 2: ~20% (restrained responses)
  Score 3: ~15% (partial compliance)
  Score 4: ~10% (prohibited content)
  Score 5: ~5% (harmful fulfillment)
```

### Step 2: Build Dataset with Proper Labels

Create `scripts/build_nbf_dataset_from_pseudo_labels.py`:

```python
# Uses the NBF pseudo-labels to build train.npz / val.npz
# with proper 1-5 labels instead of binary 3/5
```

### Step 3: Train from Scratch

```bash
# Train dynamics (Stage 1)
python -m guardbound.training.train_dynamics --config config/default.yaml

# Train NBF (Stage 2)
python -m guardbound.training.train_nbf --config config/default.yaml
```

### Step 4: Evaluate

```bash
python scripts/evaluate_nbf_defense.py --attack crescendo
```

---

## Decision Matrix

| Situation | Recommendation |
|-----------|----------------|
| No Llama API, Llama local not running | Option A (NBF pseudo-labels) |
| Llama local running | Option B or C |
| Want fastest results | Option A |
| Want most accurate labels | Option B |

---

## Next Command to Run

```bash
python scripts/generate_nbf_pseudo_labels.py
```

This will:
1. Load author's pretrained NBF
2. Score all 28K+ turns
3. Save proper 1-5 labels
4. Show score distribution

Let's start!
