"""Markdown → LaTeX conversion of `**bold**`, `*italic*`, `# heading`."""
from __future__ import annotations

from .rules import apply_rules, compiled_rules

_RULES = compiled_rules("sanitize_rules.yaml", "markdown")


def md_to_latex(s: str) -> str:
    return apply_rules(s, _RULES)
