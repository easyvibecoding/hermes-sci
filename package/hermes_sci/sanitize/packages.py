"""Fallbacks for LaTeX commands whose packages we do NOT include.

Rules live in data/package_fallbacks.yaml. Add an entry when a new provider
(siunitx, cleveref, microtype, etc.) starts leaking commands that break
compilation for our portable template.
"""
from __future__ import annotations

from .rules import apply_rules, compiled_rules

_RULES = compiled_rules("package_fallbacks.yaml", "fallbacks")


def apply_package_fallbacks(s: str) -> str:
    return apply_rules(s, _RULES)
