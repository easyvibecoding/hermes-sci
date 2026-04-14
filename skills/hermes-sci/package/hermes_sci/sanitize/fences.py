"""Strip markdown ```code``` fences, keeping inner content."""
from __future__ import annotations

from .rules import apply_rules, compiled_rules

_RULES = compiled_rules("sanitize_rules.yaml", "code_fences")


def strip_code_fences(s: str) -> str:
    return apply_rules(s, _RULES)
