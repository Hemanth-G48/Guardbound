# Ambiguity Audit — "Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks" (arXiv 2503.00187v3)

## SPECIFIED
- Embedding models: `all-mpnet-base-v2` (default), `all-distilroberta-v1` (sentence-transformers)
- Embedding dim n = 768; state dim m = 768
- Dynamics: x_k = f_θ(x_{k-1}, u_k), z_k = g_θ(x_k, u_k), x_0 = 0_m (Eq. 3)
- f_θ, g_θ: 3-layer ReLU MLPs, shape 1536-512-512-768
- h: 3-layer ReLU MLP, shape 1536-32-32-5 (softmax over 5 judge-score classes)
- Safety label set Y = {1..5} from GPT-4o judge; unsafe ⇔ score 5; safe ⇔ scores 1–4
- Losses: L_dyn (Eq. 4), L_CE (Eq. 6), L_SS (Eq. 11), L_SI (Eq. 12); joint objective Eq. (Sec 4.3) min λ_dyn L_dyn + λ_CE L_CE + λ_SS L_SS + λ_SI L_SI
- Loss weights: λ_dyn = 1, λ_CE = 1, λ_SS = 100, λ_SI = 100
- Dynamics training: Adam, lr 1e-4, 200 epochs
- NBF training: Adam, lr 1e-3, 200 epochs
- Steering threshold during training η = 0; adjustable at evaluation (0, 1e-4 … 1e-2, 5e-2)
- Non-invariant turns κ = 3 default; ablated κ ∈ {2, 4}
- Training data: multi-turn jailbreak conversations against GPT-3.5-turbo, goals from 1k Circuit Breakers training samples; attacks Acronym (881/1000 successful), Crescendo (404/1000), Opposite-day (509/1000), ActorAttack (460/2327)
- Test data: 200 HarmBench behaviors, released by Ren et al. (2024), decontaminated
- Defense rule: filter query if h(x_{k-1}, û) + η ≥ 0 (Q-filter); max 8 turns per attack; rejected-but-not-filtered turns regenerated
- Attacks evaluated: ActorAttack, Crescendo, Opposite-day (+Acronym train-only); RedQueen & SafeMT_ATTACK_600/MHJ for unseen generalization
- Metrics: ASR via GPT-4o judge; MMLU; MTBench; over-refusal on XSTest / JailbreakBench-Benign / PHTest-Harmless; F1 prompt-harmfulness detection on HarmBench / AegisSafetyTest / WildGuardTest
- Prompt-harmfulness classification rule: prompt harmless iff argmax predicted score = 1
- Baselines: original model, safe system prompt (Llama-2-Chat template), LoRA SFT (LLaMA-Factory, lr 2e-4, 3 epochs, safety-aligned responses from Ren et al.), LoRA DPO/KTO (over-refusal comparison); guardrail baselines OpenAI Moderation, ShieldGemma-2B, LLaMA Guard-7B
- Target LLMs: GPT-3.5-turbo-0125, GPT-4o-2024-08-06, o1-2024-12-17, Claude-3.5-Sonnet-20241022, Llama-3-8b-instruct, Llama-3.1-80b/70b, Phi-4, GPT-5, Claude Sonnet 4.5; temperature 0.7 everywhere
- MTBench post-steering replacement text when h > 0 (verbatim refusal string given in B.1)
- MMLU post-filtering: system prompts treated as pre-question turns to initialize dynamics; answer counted wrong if h > 0
- Hardware: 4× A6000 GPUs, 512 GB RAM
- Adaptive attack: sample candidate queries 3× from attack method, choose the one maximizing the NBF value

## PARTIALLY SPECIFIED
- Output-layer activation of f_θ, g_θ: "ReLU-based MLPs" stated; whether output layer is linear is implied but not explicitly stated.
- Whether stage-2 training (predictor) keeps f_θ, g_θ frozen or jointly fine-tunes all three nets: text says predictor trained "based on the pretrained neural dialogue dynamics" (implying frozen), yet Sec 4.3 presents a single joint objective over f_θ, gθ, h. Both readings are defensible → implement frozen-dynamics stage-2 as primary reading, expose joint fine-tuning as an option.
- Batch size and optimizer schedule details (weight decay, schedulers): not specified.
- Train/validation split ratio within the ~3.8k collected conversations: not specified.
- Number of turns K in training conversations: bounded by attack implementations ("at most 8 turns" at eval); exact distribution of K in the training set not specified.
- How U_{k−1} (query context embedding set) is realized during training: losses only use dataset queries u_k, u_{k+1}; no explicit context-set sampling procedure given.
- GPT-4o judge prompt used for 1–5 scoring: paper cites Qi et al. (2023) / Ren et al. (2024) protocols without giving the verbatim prompt.
- ASR judging criteria details: "following Ren et al. (2024)" — inherit that protocol.
- LoRA rank/alpha/target modules for SFT/DPO/KTO baselines: not specified (LLaMA-Factory defaults implied).
- Which embedding normalizes (if any) are applied by sentence-transformers defaults.

## UNSPECIFIED
- Random seeds; number of runs per experiment; variance reporting.
- Exact architecture activation between layers order (assumed standard Linear→ReLU).
- Any data-cleaning thresholds on judge scores other than unsafe=5/safe∈{1..4}.
- PCA visualization specifics beyond "top two components".
- Implementation language/framework (PyTorch implied by ecosystem but not stated).
