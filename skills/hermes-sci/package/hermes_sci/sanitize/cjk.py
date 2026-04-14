"""Strip CJK / fullwidth leakage from LLM output.

Unicode ranges are data-driven (data/sanitize_rules.yaml :: cjk_ranges).
Contribute a new range (Hangul, Hiragana, etc.) by editing the YAML.
"""
from __future__ import annotations

import re

from .rules import raw_list

_RANGES: list[str] = raw_list("sanitize_rules.yaml", "cjk_ranges")

if _RANGES:
    _CJK_RE = re.compile("[" + "".join(_RANGES) + "]")
else:
    _CJK_RE = re.compile(r"(?!x)x")  # never matches


def strip_cjk(s: str) -> str:
    return _CJK_RE.sub("", s)
