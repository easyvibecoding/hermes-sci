"""Strip LaTeX commands we can't support (\\input, \\include, \\includegraphics)."""
from __future__ import annotations

from .rules import apply_rules, compiled_rules

_RULES = compiled_rules("sanitize_rules.yaml", "bad_commands")


def strip_bad_commands(s: str) -> str:
    return apply_rules(s, _RULES)
