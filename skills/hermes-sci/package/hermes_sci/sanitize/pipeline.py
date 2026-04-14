"""Sanitize pipeline — list-composed LaTeX cleaning.

Order matters:
  1. strip reasoning-model artefacts (<think>...) BEFORE fence removal, or
     a ``` inside <think> could corrupt the fence scanner
  2. strip markdown code fences — unwrap inner content
  3. strip CJK leakage
  4. unwrap unsupported package commands (siunitx, etc.)
  5. markdown → LaTeX (bold / italic / heading)
  6. strip bad \\input \\include
  7. wrap lonely \\item
  8. escape prose specials (%, &, <, >) — runs last so earlier passes'
     output is also escaped

To add a pass: write a module with a `str → str` function, append to
SANITIZE_PIPELINE. For pure regex passes, prefer adding to
`data/sanitize_rules.yaml` — no Python change needed.
"""
from __future__ import annotations

from typing import Callable

from .bad_cmds import strip_bad_commands
from .cjk import strip_cjk
from .escape import escape_prose_specials
from .fences import strip_code_fences
from .items import wrap_lonely_items
from .markdown import md_to_latex
from .packages import apply_package_fallbacks
from .reasoning import strip_reasoning

Pass = Callable[[str], str]

SANITIZE_PIPELINE: list[Pass] = [
    strip_reasoning,
    strip_code_fences,
    strip_cjk,
    apply_package_fallbacks,
    md_to_latex,
    strip_bad_commands,
    wrap_lonely_items,
    escape_prose_specials,
]


def sanitize_latex(s: str) -> str:
    for p in SANITIZE_PIPELINE:
        s = p(s)
    return s.strip()
