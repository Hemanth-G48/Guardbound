"""Phases 3-4 — Trainers for the dynamics stage and predictor/NBF stage.

Implemented by prompts/PHASE_3_PROMPT.md and PHASE_4_PROMPT.md. Planned modules:
    losses.py           L_dyn (Eq.4), L_CE (Eq.6), L_SS (Eq.11), L_SI (Eq.12)
    train_dynamics.py   Stage 1: Adam lr 1e-4, 200 epochs
    train_nbf.py        Stage 2: Adam lr 1e-3, 200 epochs; frozen dynamics default
    diagnostics.py      per-turn MSE, loss-curve plots
"""
from __future__ import annotations
