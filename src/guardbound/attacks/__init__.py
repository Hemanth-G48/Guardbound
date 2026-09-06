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
from .crescendo_paper import CrescendoAttackPaper, CrescendoAttackPaperSimple
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
    "crescendo_paper": {
        "method_name": "Crescendo (Paper Implementation)",
        "original_paper": "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (Russinovich et al., 2024)",
        "official_repository": "https://github.com/IBM/ai-anchoring",
        "implementation_status": "full-paper",
        "local_deviations": "Full implementation with attacker LLM, scoring rubric, safety filtering, and backtracking",
    },
    "crescendo_paper_simple": {
        "method_name": "Crescendo (Simplified Paper)",
        "original_paper": "Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack (Russinovich et al., 2024)",
        "official_repository": "https://github.com/IBM/ai-anchoring",
        "implementation_status": "paper-derived",
        "local_deviations": "Simplified version with escalation templates and backtracking, no separate attacker LLM required",
    },
    "opposite_day": {
        "method_name": "Opposite Day",
        "original_paper": "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts (Li et al., 2024)",
        "official_repository": "Author TMLR_supp_code",
        "implementation_status": "full-paper",
        "local_deviations": "Full implementation with GPT-4 dynamic generation, backtracking, and LLM-based evaluation",
    },
    "actor_attack": {
        "method_name": "ActorAttack",
        "original_paper": "Derail Yourself: Multi-turn LLM Jailbreak through Self-discovered Clues (Ren et al., 2024)",
        "official_repository": "Author TMLR_supp_code",
        "implementation_status": "full-paper",
        "local_deviations": "Full implementation with GPT-4 dynamic generation, actor-network theory, and query rewriting",
    },
    "acronym": {
        "method_name": "Acronym",
        "original_paper": "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts (Li et al., 2024)",
        "official_repository": "Author TMLR_supp_code",
        "implementation_status": "full-paper",
        "local_deviations": "Full implementation with GPT-4 dynamic generation, acronym obfuscation, and disclaimer checking",
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
    "CrescendoAttackPaper",
    "CrescendoAttackPaperSimple",
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
