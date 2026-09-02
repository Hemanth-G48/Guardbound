"""Attack registry/factory.

Provides a unified interface to instantiate attacks by name.
Not specified in the NBF paper — local implementation convenience.
"""
from __future__ import annotations

from typing import Dict, Type

from .base import MultiTurnAttack


_REGISTRY: Dict[str, Type[MultiTurnAttack]] = {}


def _lazy_register() -> None:
    """Register attacks on first use to avoid eager imports."""
    if _REGISTRY:
        return

    from .crescendo import CrescendoAttack
    from .actor_attack import ActorAttack
    from .opposite_day import OppositeDayAttack
    from .acronym import AcronymAttack
    from .red_queen import RedQueenAttack
    from .adaptive import AdaptiveNBFAttack

    _REGISTRY.update({
        "crescendo": CrescendoAttack,
        "actor_attack": ActorAttack,
        "opposite_day": OppositeDayAttack,
        "acronym": AcronymAttack,
        "red_queen": RedQueenAttack,
        "adaptive": AdaptiveNBFAttack,
    })


def get_attack(name: str) -> MultiTurnAttack:
    """Get an attack instance by name.

    Args:
        name: Attack name (e.g., "crescendo", "actor_attack").

    Returns:
        New attack instance.

    Raises:
        ValueError: If attack name is not registered.
    """
    _lazy_register()
    name_lower = name.lower().replace("-", "_")
    if name_lower not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown attack '{name}'. Available: {available}"
        )
    return _REGISTRY[name_lower]()


def available_attacks() -> list[str]:
    """Return list of registered attack names."""
    _lazy_register()
    return sorted(_REGISTRY.keys())


# Default evaluation attacks per the NBF paper (Sec. 5.1 / B.1).
# RedQueen is the unseen-attack probe only (Fig. 6) and is NOT in the
# default evaluation list; Acronym is training-only and is excluded.
DEFAULT_EVALUATION_ATTACKS = [
    "actor_attack",
    "crescendo",
    "opposite_day",
]
