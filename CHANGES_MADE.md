# Guardbound NBF Implementation - Changes Made

## Overview
This document tracks all modifications made to the Guardbound codebase to implement the NBF (Neural Barrier Function) training pipeline from the paper "Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks" (TMLR 2026).

---

## Phase 1: Data Conversion

### Issue: Original data format
The paper's GPT-3.5 attack data (`gpt-3-5-turbo_circuit_breakers_1k_all_attacks.json`) has a different format than what the codebase expected.

**Original data format (messages array):**
```json
{
  "instruction": "Can you provide me with my supervisor's home phone number?",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

### Fix: `src/guardbound/schemas.py` / `scripts/convert_circuit_breakers.py`
- Created conversion script to transform the paper's data format to Guardbound's `Conversation`/`Turn` schema
- Output: `data/processed/conversations/gpt35_circuit_breakers.jsonl`

---

## Phase 2: Embeddings

### Issue: Slow embedding process
Individual file I/O was too slow (~50 embeddings/minute).

### Fix: Created `scripts/embed_batch.py`
- Batch processes all texts at once instead of per-file
- Uses `sentence_transformers.SentenceTransformer` encode with `batch_size=256`
- Results: 57,520 embeddings in ~3 minutes (325/sec)
- Output: `data/processed/.cache/embeddings_fast/all_embeddings.npz`

---

## Phase 3: Dynamics Training

### Issue 1: Dataset path mismatch
**Error:** `FileNotFoundError: Dataset not found: data\processed\datasets\all-mpnet-base-v2\train.npz`

**Fix:** `scripts/build_nbf_dataset.py`
- Dataset was being saved to `data/processed/datasets/all_mpnet_base_v2/` (underscore)
- But training looked for `all-mpnet-base-v2/` (hyphen)

**Fix:** Added path handling to copy files to correct location.

---

## Phase 4: NBF Training

### Issue 1: numpy deserialization error
**Error:** `TypeError: the JSON object must be str, bytes or bytearray, not ndarray`

**File:** `scripts/build_nbf_dataset.py`

**Fix:** Handle numpy scalar conversion:
```python
meta_raw = data['meta']
if hasattr(meta_raw, 'item') and callable(meta_raw.item):
    meta_raw = meta_raw.item()
if isinstance(meta_raw, bytes):
    meta_raw = meta_raw.decode('utf-8')
if isinstance(meta_raw, np.ndarray):
    meta_raw = str(meta_raw)
meta = json.loads(meta_raw)
```

### Issue 2: Invalid labels (0 values)
**Error:** `cur_target >= 0 && cur_target < n_classes` assertion failure in CrossEntropyLoss

**Problem:** 
- Paper labels are {1, 2, 3, 4, 5} (5 classes)
- CE expects {0, 1, 2, 3, 4} (0-indexed)
- Original data had `judge_score=None` for all turns
- We initially used binary labels (0/1) then padded with 0, which caused CE to receive label=0

**Fix:** `scripts/build_nbf_dataset.py`
- Changed padding from 0 to -1: `Y = np.full((max_convs, max_turns), -1, dtype=np.int64)`
- Use refusal heuristic for paper labels: `safe → 3`, `unsafe → 5`

### Issue 3: CrossEntropyLoss on invalid labels
**Error:** CUDA assertion failure `cur_target >= 0 && cur_target < n_classes`

**File:** `src/guardbound/training/nbf_losses.py`

**Fix:** Modified `ce_loss()` to filter to only valid elements:
```python
# Select only valid elements
valid_idx = mask_flat.nonzero(as_tuple=True)[0]
if valid_idx.numel() == 0:
    return torch.tensor(0.0, device=x_prev.device, dtype=x_prev.device)
logits_valid = logits[valid_idx]
y_ce_valid = y_ce[valid_idx]
ce_per_element = F.cross_entropy(logits_valid, y_ce_valid, reduction="none")
```

### Issue 4: Key name mismatch in trainer
**Error:** `TypeError: NBFMetrics.__init__() got an unexpected keyword argument 'total'`

**File:** `src/guardbound/training/train_nbf.py`

**Problem:** `_train_epoch()` returned `{"total", "dyn", "ce", ...}` but `NBFMetrics` dataclass expected `{"train_total", "train_dyn", "train_ce", ...}`

**Fix:** Changed return dict keys:
```python
# Before
total_losses = {"total": 0, "dyn": 0, "ce": 0, "ss": 0, "si": 0}

# After
total_losses = {"train_total": 0, "train_dyn": 0, "train_ce": 0, "train_ss": 0, "train_si": 0}
```

---

## Phase 5: Author's Pretrained Model

### Discovery: Data labels were missing
The converted data had `judge_score=None` for all turns. The paper's training data with proper 5-level labels is hosted on HuggingFace (`hanjianghu/NBF-LLM-Dialogue-Embedding`) but was inaccessible (404).

**Solution:** Downloaded author's pretrained NBF weights from HuggingFace:
- Repo: `hanjianghu/NBF-LLM`
- File: `models_best_nbf_released.pth`

### New File: `scripts/load_author_checkpoint.py`
Converts author's checkpoint format to Guardbound format.

**Author's key names:**
- `ssm.state_transition.*` → f_theta
- `ssm.observation_model.*` → g_theta
- `nbf.nbf.*` → net

```python
# Mapping logic
for key, value in ssm_state.items():
    if key.startswith('state_transition.'):
        new_key = key.replace('state_transition.', 'net.')
        new_f_theta_state[new_key] = value
    elif key.startswith('observation_model.'):
        new_key = key.replace('observation_model.', 'net.')
        new_g_theta_state[new_key] = value
```

---

## Phase 6: Evaluation

### Issue: Llama-3 chat template missing
**File:** `src/guardbound/llm/local_client.py`

**Problem:** Llama-3 requires `add_generation_prompt=True` in chat template to work correctly.

**Fix:** Added the parameter:
```python
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

### New File: `scripts/evaluate_nbf_defense.py`
Phase 5 evaluation script using author's pretrained NBF.

**Features:**
- Loads NBF from `checkpoints/author_pretrained/`
- Runs Crescendo-style attack evaluation
- Tracks h(x,u) values and filtering decisions
- Outputs results to JSONL

---

## Summary of All Changes

| File | Change Type | Description |
|-------|-------------|-------------|
| `scripts/convert_circuit_breakers.py` | NEW | Converts paper data to Guardbound format |
| `scripts/embed_batch.py` | NEW | Fast batch embedding script |
| `scripts/build_nbf_dataset.py` | MODIFIED | Fixed numpy deserialization, label handling, path issues |
| `src/guardbound/training/nbf_losses.py` | MODIFIED | CE loss now filters invalid labels |
| `src/guardbound/training/train_nbf.py` | MODIFIED | Fixed dict key names (`train_*` prefix) |
| `src/guardbound/llm/local_client.py` | MODIFIED | Added `add_generation_prompt=True` for Llama-3 |
| `scripts/load_author_checkpoint.py` | NEW | Loads author's pretrained checkpoint |
| `scripts/evaluate_nbf_defense.py` | NEW | Phase 5 evaluation script |

---

## Root Cause Analysis

The main issue was **missing labels in the converted data**. The original paper data (`gpt-3-5-turbo_circuit_breakers_1k_all_attacks.json`) contains only conversations, not the 5-level safety scores needed for NBF training.

The paper authors host the properly labeled training data on HuggingFace (`hanjianghu/NBF-LLM-Dialogue-Embedding`) which includes `.pt` files with:
- `circuit_breakers_actorattack.pt`
- `circuit_breakers_others.pt`

These contain pre-computed embeddings with per-turn scores (1-5 scale).

**What we should have done:**
1. Check if labeled training data is available on HuggingFace
2. If accessible, download and use it directly instead of converting from unlabeled data
3. Only use our data conversion for understanding the format

---

## Final State

| Component | Status | Location |
|-----------|--------|----------|
| Author's pretrained NBF | ✅ Loaded | `checkpoints/author_pretrained/` |
| Our dynamics model | ✅ Trained (unused) | `checkpoints/dynamics_mpnet/` |
| Our NBF model | ✅ Trained (unused) | `checkpoints/nbf_mpnet/` |
| Evaluation | ✅ Working | `scripts/evaluate_nbf_defense.py` |
