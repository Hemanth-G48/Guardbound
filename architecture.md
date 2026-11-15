# Guardbound — How It Works

This describes the **processes and options** in the system, not the file
layout. Read `project.md` first for the purpose. Names in `code font` are
there only so you can find the implementation; the explanation stands without
them.

There are four workflows, run roughly in this order:

1. **Build training data** from attack conversations
2. **Train** the dialogue dynamics and the barrier function
3. **Defend** — run a conversation through the Q-filter
4. **Evaluate** — attack the defended and undefended models and measure

---

## 1. Building training data

**Purpose:** get conversations labelled turn by turn for how safe they are.

1. **Generate multi-turn attack conversations.** An *attacker* LLM is given a
   harmful goal (from HarmBench, circuit-breakers goals, etc.) and uses one of
   the attack strategies against a *target* LLM:
   - **Crescendo** — gradually escalate from benign to harmful.
   - **ActorAttack** — build a chain of related "actors" and topics that lead
     toward the goal.
   - **Opposite-day** — get the model to invert its answers.
   - **Acronym** — hide the request in an acronym spelled out over turns
     (used for training only; the paper drops it from evaluation because many
     of its attacks succeed in one turn).
   - **Red-Queen** — used as a held-out attack for generalization tests.
2. **Judge each turn.** A judge LLM scores the conversation after every turn on
   a 1–5 scale. 5 is unsafe, 1–4 are safe.
   - *Option A:* GPT-4o judge (the paper's setup).
   - *Option B:* a local Llama-based judge producing "pseudo-labels", used when
     avoiding paid calls. A retraining path exists that uses these labels.
3. **Embed.** Every user message is turned into a 768-d sentence embedding
   (default `all-mpnet-base-v2`; `all-distilroberta-v1` is the ablation).
4. **Assemble tensors.** Each conversation becomes a sequence of embeddings plus
   per-turn labels, padded to at most 8 turns.

Output: a tensor dataset of (embedding sequence, label sequence).

---

## 2. Training

Two stages. The idea: first learn how conversations *move*, then learn which
states/inputs are dangerous.

**Stage 1 — dynamics.** Learn two small networks:
- `f(x, u)` → next hidden state (how a message changes the dialogue state),
- `g(x, u)` → predicted embedding of the model's response.
The state starts at zero. Trained to reproduce the observed response
embeddings (Adam, lr 1e-4, 200 epochs).

**Stage 2 — barrier function.** A small classifier looks at
`[previous state; incoming message embedding]` and predicts the 5 safety
levels. The barrier value is `h = p(unsafe) − max p(safe)`; positive means
"looks unsafe". Three losses are combined:
- **Cross-entropy** on the judge labels.
- **Safe-set loss** — force `h` to the correct sign for turns that are
  already safe/unsafe.
- **Safety-invariance loss** — if the conversation is safe now, then after
  admitting a safe message it must *stay* safe (a one-step-ahead hinge). This
  is what makes `h` a *barrier*, not just a classifier. κ controls how many
  non-invariant turns are tolerated (default 3).

Options:
- **Frozen vs joint:** by default the dynamics are frozen during stage 2;
  `--joint` trains them together.
- **Loss-weight ablation** (drop a loss, vary λ_SS/λ_SI in {10,100,1000}).
- **Embedding model** (mpnet vs distilroberta — full retrain).
- **Author checkpoint path:** the authors' released NBF can be loaded and
  adapted into our format, to compare ours against theirs or to run the
  pipeline without training.

Output: dynamics + barrier checkpoints, which together form one NBF.

---

## 3. The defense at inference time (the Q-filter)

For each incoming user message `U_k`:

```
embed U_k → u_k
h_k = h(x_{k-1}, u_k)
if h_k + η ≥ 0:      FILTER   → return a refusal; no LLM call; state unchanged
else:                ADMIT    → call the target LLM; x_k = f(x_{k-1}, u_k)
```

Key points a newcomer must know:
- **Sign convention.** A message is admitted only if `h < −η`. Getting this
  sign wrong was an earlier bug (D-01); the rule now lives in one place and
  every filter in the codebase uses it.
- **η is the single tuning knob.** Larger η filters more: lower ASR, more
  over-refusal. Headline value 1e-3; practical recommendation 5e-4; sweeps from
  0 up to 5e-2.
- **A filtered turn doesn't advance the state**, so a blocked message can't
  poison later scoring.
- **Initial state.** The state starts at zero, or can be initialized from an
  existing context (system prompt / prior conversation).
- **Wrapper design.** The defense is a wrapper around any chat LLM, so the
  attacks and metrics don't know whether the target is defended.
- Some evaluations also have a *retry* mode, where a filtered turn lets the
  attacker re-propose a different message.

### Defense variants (all share the same wrapper interface)

| Variant | What it does |
|---|---|
| `original` | No defense — pass-through |
| `system_prompt` | Llama-2-style safety system prompt |
| `lora_sft` / `lora_dpo` / `lora_kto` | Base model fine-tuned for safety (data from Ren et al. 2024 safety responses) |
| `guardbound` | The NBF Q-filter above |

---

## 4. Evaluation

### 4a. Attacking the target

Three roles are always present: **attacker** LLM, **target** LLM (optionally
defended), **evaluator** LLM (decides if the attack succeeded). A run is, per
goal: up to 8 turns of attacker → (filter) → target, then the evaluator scores
the transcript. **ASR** is the fraction of goals with a successful attack.

Attack options:
- **Standard** (Crescendo, ActorAttack, Opposite-day) against any defense variant.
- **Adaptive** — the attacker can query the barrier function and steer around
  it (worst case over 3 samples). This tests whether the defense is
  robust to an attacker that knows it exists.

Rules the harness enforces so numbers are comparable:
- Failed runs (API errors, invalid evaluator output, zero-turn rows) are
  **excluded from the ASR denominator and counted separately**; a result of 0%
  from nothing running is impossible by construction.
- Every run writes a **manifest**: git commit and dirty flag, dataset hashes,
  prompt and rubric hashes, goal counts, usage, completion status.

### 4b. Utility and over-refusal

Does the defense hurt normal use?
- **MMLU** (knowledge) and **MTBench** (chat quality, GPT-4o judged) — when a
  message is filtered, the answer is replaced/penalized appropriately.
- **Over-refusal** — XSTest, JBB, PHTest: harmless prompts that look scary.

### 4c. Guardrail benchmark

Compare the NBF *as a classifier* against Llama Guard, ShieldGemma and OpenAI
moderation on F1 over HarmBench, Aegis and WildGuard.

### 4d. Experiment suite (E1–E10)

Declarative specs that combine the pieces above:

| ID | Question |
|---|---|
| E1 | Main results: ASR + helpfulness + over-refusal, all defenses |
| E2 | Sweep η |
| E3 | Which loss terms matter |
| E4 | Effect of κ |
| E5 | MPNet vs DistilRoBERTa |
| E6 | Generalization to attacks held out of training |
| E7 | Adaptive attacker |
| E8 | Over-refusal vs alignment baselines |
| E9 | PCA of state trajectories, original vs steered |
| E10 | Newest models, if API access exists |

A runner executes them (with `--dry-run` / `--smoke` modes needing no models),
and a report builder aggregates results into Pareto plots, tables, and
claim checks against the paper.

---

## Where models run (local vs cloud)

The attacker, target and evaluator are chosen by configuration, never by
changing attack code:

- **Local** — open models via HuggingFace/Ollama. Because three large models
  don't fit on one 24 GB GPU, a **model manager swaps them in and out**
  ("sequential residency"). Attacker models are selectable by profile
  (Qwen3, GLM-4.6V, Ornith-1.5 …) so the attacker can be swapped as the only
  variable.
- **Cloud** — OpenAI / Anthropic. Paid access is **fail-closed**: a run needs
  explicit authorization, and a local "qualification" gate (is the NBF and
  setup valid? is the attacker capable enough?) must pass before any paid
  client is even constructed. Usage is logged per call.
- **Mock** — fake LLMs so the whole pipeline, tests and `--dry-run` modes run
  for free.

---

## The reproduction-fidelity layer

Because the goal is a *faithful* reproduction, there is a protocol layer on
top of everything:

- A **paper protocol** (R1) defines the settings the paper states: which
  attacks are in the main evaluation, temperatures 0.7, η = 1e-3, target model
  `gpt-4o-2024-08-06`, 8 turns, 200 goals. Configs are validated against it,
  so a wrong setting can't silently reach a paid run.
- Where the paper is silent (e.g. the attacker model), the authors' code is
  used and the choice is labelled "code-recovered".
- A series of **phases** (8 → 53) progressively built the pipeline, then
  audited it against the paper, then hardened the qualification and
  authorization steps. Each phase has a report in `docs/reports/` or `docs/`;
  `HANDOFF.md` summarizes the most recent state.

---

## End-to-end summary

```
goals ─► attacker vs target ─► judged conversations ─► embeddings ─► tensors
                                                                       │
                                  stage 1: dynamics (f, g)  ◄──────────┤
                                  stage 2: barrier h (CE + safe-set + invariance)
                                                  │
                                           NBF checkpoint
                                                  │
 user/attacker message ─► embed ─► h + η ≥ 0 ? ─┬─ yes: refuse (LLM never called)
                                                 └─ no : target LLM answers, state advances
                                                  │
              ASR · MMLU · MTBench · over-refusal · guardrail F1 · adaptive attack
```
