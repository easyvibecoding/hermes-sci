"""Drop orphan `$` signs left behind by truncated LLM output.

Failure mode this targets: MiniMax / any LLM sometimes truncates an inline
equation mid-sentence, e.g.

    We compare against the ensemble variant with $

followed immediately by `\\section{Experiments}`. The unmatched `$` opens
math mode that never closes, so pdflatex fails with "Missing $ inserted"
anywhere downstream — producing a cascade of misleading errors.

We can't reliably guess what the LLM intended to write, so we don't try to
reconstruct the equation. We drop the orphan instead: outside a section the
unclosed `$` is unsalvageable prose anyway, and leaving it breaks the whole
paper. A logged warning surfaces the event so the author can fix the prose
by hand if they care about that sentence.

Scope: only considers `$` that are NOT inside a display-math or tabular
environment (those match as a whole via _MATH_BLOCK below and are preserved
verbatim). Paired `$...$` inline math is fine — we remove only the last
unmatched `$` when the count is odd.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("hermes_sci.sanitize.math_balance")

# Display / block math environments whose internal `$` we must NOT count.
_MATH_BLOCK = re.compile(
    r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}"
    r"|\\begin\{align\*?\}.*?\\end\{align\*?\}"
    r"|\\begin\{gather\*?\}.*?\\end\{gather\*?\}"
    r"|\\begin\{multline\*?\}.*?\\end\{multline\*?\}"
    r"|\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}"
    r"|\\begin\{array\}.*?\\end\{array\}"
    r"|\\\[.*?\\\]"
    r"|\$\$.*?\$\$",
    re.DOTALL,
)

# Escaped dollar (literal `$` character) — already fine, must be ignored by
# the counter. Matches `\$`.
_ESCAPED_DOLLAR = re.compile(r"\\\$")


def balance_inline_math(s: str) -> str:
    """If the prose section has an odd number of unescaped `$`, drop the
    last one. Returns input unchanged when already balanced.
    """
    # Black out regions the scanner must ignore: math blocks + literal \$.
    blanked = _MATH_BLOCK.sub(lambda m: " " * len(m.group(0)), s)
    blanked = _ESCAPED_DOLLAR.sub("  ", blanked)

    dollar_positions = [i for i, ch in enumerate(blanked) if ch == "$"]
    if len(dollar_positions) % 2 == 0:
        return s

    # Odd count → drop the LAST unpaired `$`. Heuristic: most LLM-truncation
    # cases leave the stray `$` at the very end of a sentence. Dropping the
    # last occurrence recovers the balanced state and preserves any earlier
    # `$x$` pairs the author did finish.
    last = dollar_positions[-1]
    log.warning("dropping orphan $ at offset %d (odd count = %d)",
                last, len(dollar_positions))
    return s[:last] + s[last + 1:]
