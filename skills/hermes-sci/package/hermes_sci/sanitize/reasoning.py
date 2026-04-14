"""Strip reasoning-model artefacts (<think>, <|thinking|>, etc.).

Rules live in data/sanitize_rules.yaml under `reasoning_artefacts`.
"""
from __future__ import annotations

from .rules import apply_rules, compiled_rules

_RULES = compiled_rules("sanitize_rules.yaml", "reasoning_artefacts")


def strip_reasoning(s: str) -> str:
    return apply_rules(s, _RULES)
