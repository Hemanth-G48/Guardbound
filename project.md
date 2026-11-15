# Guardbound — What This Project Is

## The problem

Large language models refuse obviously harmful requests, but they can be talked
into harmful output over **several turns of conversation**. A "multi-turn
jailbreak" never asks for the harmful thing directly. It starts with innocent
questions, builds context, and nudges the model step by step until the final
answer is something the model would have refused in turn one. Crescendo,
ActorAttack, Opposite-day and Acronym are well-known attacks of this kind.

Most defenses look at one message at a time, or train the model itself
(alignment fine-tuning, safety system prompts). Single-message checks miss the
attack because no single message is bad. Fine-tuning tends to make the model
refuse harmless requests too (over-refusal) and hurts general ability.

## The idea we are implementing

This project reproduces the paper *"Steering Dialogue Dynamics for Robustness
against Multi-turn Jailbreaking Attacks"* (Hu, Robey, Liu, arXiv:2503.00187v3;
the PDF is in `docs/paper/`).

The paper treats a conversation as a **dynamical system**, the way a control
engineer treats a car or a robot:

- The conversation has a hidden **state** that summarizes where the dialogue
  has been heading.
- Each new user message is an **input** that moves the state.
- A **Neural Barrier Function (NBF)** `h` is trained to predict, from the
  current state plus the incoming message, whether letting this message through
  could push the conversation toward an unsafe outcome.
- A **Q-filter** sits in front of the LLM. If the barrier says "unsafe"
  (`h + η ≥ 0`), the message is blocked and the LLM is never called. Otherwise
  the LLM answers and the state advances.

The result is a defense that is **proactive** (it blocks the step that leads
toward harm, not just the final harmful request), works **outside the model**
(no retraining of the target LLM, so it works on closed models like GPT-4o),
and is **tunable** with one knob, `η`, which trades safety against
over-refusal.

## What we are trying to achieve

1. **Reproduce the paper's results.** Build the full pipeline (data → dynamics
   model → barrier function → steered chat) and show that the defense lowers
   Attack Success Rate (ASR) on multi-turn attacks while keeping the model
   helpful (MMLU, MTBench) and not over-refusing (XSTest and similar).
2. **Compare against alternatives.** Original undefended model, a safety system
   prompt, and LoRA-fine-tuned alignment (SFT / DPO / KTO), plus other
   guardrail classifiers (Llama Guard, ShieldGemma, OpenAI moderation, …) on
   F1.
3. **Test the limits.** Ablations (loss terms, threshold `η`, embedding model,
   number of non-invariant turns), generalization to attacks held out of
   training, and an **adaptive attacker** that knows about the filter and tries
   to get around it.
4. **Do it rigorously and cheaply.** A large part of the work so far has been
   auditing our reproduction against the paper and the authors' code (the
   D-01…D-16 defect register) so that any number we report is trustworthy, and
   running locally on open models before spending money on cloud APIs.

## What the system produces

- A trained dynamics model and barrier function (checkpoints), including ones
  adapted from the authors' released weights for comparison.
- Attack transcripts against defended and undefended targets, judged for
  success.
- Tables and plots: ASR vs `η`, ASR vs helpfulness (Pareto), over-refusal,
  guardrail F1, state-trajectory visualizations.

## Where the project stands

- The local pipeline works end to end and has been exercised with open models
  on a single 24 GB GPU.
- Work has moved from "build it" to "prove the reproduction is faithful":
  verifying the filter rule's sign and threshold, ActorAttack steering, the ASR
  denominator, dataset and checkpoint hashes, prompt fidelity, and run
  manifests.
- The headline reproduction run targets `gpt-4o-2024-08-06` for attacker,
  target and evaluator at temperature 0.7, 8 turns, 200 HarmBench goals,
  `η = 1e-3`.
- **Paid cloud calls are deliberately gated.** Nothing runs against a paid API
  without explicit human approval; a failed qualification check stops the run
  before any provider client is created. `HANDOFF.md` records that no paid
  request had been made as of its writing. Later phase reports in
  `docs/reports/` and `docs/` continue the audit work.

## Ground rules for anyone (human or AI) continuing the work

- The paper is the authority; where the paper is silent, the authors' released
  code is the fallback, and that choice is labelled.
- Don't change attack algorithms, evaluator rubrics, or the NBF
  architecture/scoring to "improve" results. The goal is faithful reproduction.
- Don't start paid runs, and never print or log API keys.
- Don't treat a green test suite as proof the methodology is right; the audit
  reports are the evidence.

## Pointers

- How the processes fit together: `architecture.md`
- Commands to run each stage: `README.md`
- Latest status and constraints: `HANDOFF.md`
- Detailed evidence per phase: `docs/reports/`, `docs/audit/`
