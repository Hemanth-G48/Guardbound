---
license: mit
language:
- en
task_categories:
- text-classification
tags:
- safety
- jailbreak
- dialogue
- embeddings
- llm-safety
- multi-turn
- sentence-transformers
size_categories:
- 1K<n<10K
---

# NBF-LLM Dialogue Embedding Data

Training and validation dialogue embedding data from the paper:

**Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**  
Hanjiang Hu, Alexander Robey, Changliu Liu  
[arXiv:2503.00187](https://arxiv.org/abs/2503.00187) | [GitHub](https://github.com/HanjiangHu/NBF-LLM)

## Dataset Description

This dataset contains pre-computed [sentence-transformer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (all-mpnet-base-v2, 768-dim) embeddings of multi-turn jailbreaking dialogue trajectories. Each trajectory is a list of dialogue steps; each step is a dict with:

- `u`: user turn embedding (torch.Tensor, shape `[768]`)
- `y`: assistant turn embedding (torch.Tensor, shape `[768]`)
- `score`: safety score (int, 1–5)

## Files

| File | Split | Attack Methods | Queries |
|------|-------|----------------|---------|
| `circuit_breakers_actorattack.pt` | Train | ActorAttack | 1k Circuit Breakers |
| `circuit_breakers_others.pt` | Train | Crescendo, Acronym, Opposite-Day | 1k Circuit Breakers |
| `harmbench_actorattack.pt` | Validation | ActorAttack | 200 HarmBench |
| `harmbench_others.pt` | Validation | Crescendo, Acronym, Opposite-Day | 200 HarmBench |

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download training data
cb_actorattack = torch.load(
    hf_hub_download("hanjianghu/NBF-LLM-Dialogue-Embedding", "circuit_breakers_actorattack.pt"),
    map_location="cpu", weights_only=False
)
cb_others = torch.load(
    hf_hub_download("hanjianghu/NBF-LLM-Dialogue-Embedding", "circuit_breakers_others.pt"),
    map_location="cpu", weights_only=False
)

# Each is a list of trajectories; each trajectory is a list of step dicts
print(f"Train trajectories: {len(cb_actorattack) + len(cb_others)}")
step = cb_actorattack[0][0]
print(f"Step keys: {step.keys()}")  # dict_keys(['u', 'y', 'score'])
print(f"Embedding dim: {step['u'].shape}")  # torch.Size([768])
```

## Data Collection

Multi-turn attack dialogues were collected from:
- [ActorAttack](https://github.com/AI45Lab/ActorAttack)
- [Automated-Multi-Turn-Jailbreaks](https://github.com/AIM-Intelligence/Automated-Multi-Turn-Jailbreaks) (Crescendo, Acronym, Opposite-Day)

## Citation

```bibtex
@article{hu2025steering,
  title={Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks},
  author={Hu, Hanjiang and Robey, Alexander and Liu, Changliu},
  journal={arXiv preprint arXiv:2503.00187},
  year={2025}
}
```
