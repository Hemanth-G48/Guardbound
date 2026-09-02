"""Attack adapters for multi-turn adversarial conversations.

Phase 6 framework supporting multiple attack methods with both
bare-LLM and NBF-steered modes.

Attack provenance:
    crescendo     — paper-derived (Russinovich et al., 2024)
    opposite_day  — paper-derived (Li et al., 2024b)
    actor_attack  — paper-derived (Ren et al., 2024)
    acronym       — paper-derived (train-only, excluded from evaluation)
    red_queen     — paper-derived (unseen-attack generalization)
    adaptive      — NBF-paper-specific (adaptive worst-case selection)
"""

from __future__ import annotations

from .base import MultiTurnAttack
from .crescendo import CrescendoAttack
from .actor_attack import ActorAttack
from .opposite_day import OppositeDayAttack
from .acronym import AcronymAttack
from .red_queen import RedQueenAttack
from .adaptive import AdaptiveNBFAttack
from .runner import run_attack
from .registry import get_attack, available_attacks, DEFAULT_EVALUATION_ATTACKS

ATTACK_PROVENANCE = {
    "crescendo": {
        "method_name": "Crescendo",
        "original_paper": "Multilingual Jailbreak Challenges in Large Language Models (Russinovich et al., 2024)",
        "official_repository": "https://github.com/IBM/ai-anchoring",
        "implementation_status": "paper-derived",
        "local_deviations": "Adapter wrapping the Crescendo multi-turn escalation pattern; official implementation uses Azure-specific API calls",
    },
    "opposite_day": {
        "method_name": "Opposite Day",
        "original_paper": "Defending ChatGPT against Jailbreaking with System Prompt Optimization (Li et al., 2024b)",
        "official_repository": "Not available",
        "implementation_status": "paper-derived",
        "local_deviations": "Reimplemented from paper description; opposite-perspective reframing pattern",
    },
    "actor_attack": {
        "method_name": "ActorAttack",
        "original_paper": "Tastle: Large Language Model Attacks with Two-Hop Tampering (Ren et al., 2024)",
        "official_repository": "https://github.com/wjlan/LLM-Attacks-Two-Hop",
        "implementation_status": "paper-derived",
        "local_deviations": "Adapter based on paper description; self-discovered clues / actor-based decomposition",
    },
    "acronym": {
        "method_name": "Acronym",
        "original_paper": "Referenced in NBF paper as training-only attack",
        "official_repository": "Not available",
        "implementation_status": "paper-derived",
        "local_deviations": "Train-only attack; excluded from evaluation. Uses abbreviation-based obfuscation",
    },
    "red_queen": {
        "method_name": "RedQueen",
        "original_paper": "Referenced in NBF paper for unseen-attack generalization",
        "official_repository": "Not available",
        "implementation_status": "paper-derived",
        "local_deviations": "Reimplemented from paper description; used for 1/3/4/5-turn experiments",
    },
    "adaptive": {
        "method_name": "Adaptive NBF Attack",
        "original_paper": "Section 4.3 of NBF paper (Hu et al., 2026)",
        "official_repository": "This project",
        "implementation_status": "paper-specified",
        "local_deviations": "3 candidates per turn, argmax-h selection, default base attack: Crescendo",
    },
}

__all__ = [
    "MultiTurnAttack",
    "CrescendoAttack",
    "ActorAttack",
    "OppositeDayAttack",
    "AcronymAttack",
    "RedQueenAttack",
    "AdaptiveNBFAttack",
    "run_attack",
    "get_attack",
    "available_attacks",
    "DEFAULT_EVALUATION_ATTACKS",
    "ATTACK_PROVENANCE",
]
