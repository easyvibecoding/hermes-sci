"""Orphan `$` detection & removal.

Regression for the demo paper run where MiniMax truncated an inline
equation mid-sentence (`with $` followed by `\\section{...}`), leaving an
unclosed math mode that crashed pdflatex with a misleading error pointing
far downstream.
"""
from __future__ import annotations

from hermes_sci.sanitize.math_balance import balance_inline_math


def test_balanced_inline_math_unchanged():
    src = r"we set $x = 1$ and $y = 2$; done."
    assert balance_inline_math(src) == src


def test_no_math_unchanged():
    src = "plain text with no dollar signs at all."
    assert balance_inline_math(src) == src


def test_orphan_dollar_at_sentence_end_dropped():
    src = "We compare against the ensemble variant with $\n\n\\section{Experiments}"
    out = balance_inline_math(src)
    assert "$" not in out
    assert r"\section{Experiments}" in out


def test_orphan_dollar_preserves_earlier_balanced_pair():
    src = r"we have $x=1$ and later a stray $ appears."
    out = balance_inline_math(src)
    # Earlier balanced `$x=1$` must survive.
    assert "$x=1$" in out
    # Stray one removed.
    assert out.count("$") == 2


def test_display_math_ignored_by_counter():
    src = (
        r"prose $x$ more." "\n"
        r"\begin{equation}" "\n"
        r"a = b" "\n"
        r"\end{equation}" "\n"
        r"\[ c = d \]" "\n"
        r"final."
    )
    # Two `$` in prose (balanced) + `$$` inside display shouldn't confuse us.
    assert balance_inline_math(src) == src


def test_escaped_dollar_ignored():
    src = r"literal \$100 sign and paired $x$ — balanced."
    # `\$` is a LaTeX literal dollar, must not be counted as inline-math delim.
    assert balance_inline_math(src) == src


def test_triple_dollar_in_prose_drops_last():
    src = "three stray $ dollars $ in $ a row"
    out = balance_inline_math(src)
    assert out.count("$") == 2  # one removed → even count restored


def test_abstract_with_underscore_sanitized_by_writeup():
    """Regression: ideas.json produces abstract text that bypasses the
    per-section sanitize pipeline (it comes from ideation, not writeup).
    write_paper must sanitize it explicitly before render_tex, otherwise
    `perplexity_gap` etc. crashes pdflatex on page 1."""
    from hermes_sci.sanitize import sanitize_latex
    abstract = (
        "We train on (input, perplexity_gap) pairs where perplexity_gap "
        "measures draft-target discrepancy."
    )
    out = sanitize_latex(abstract)
    # Both occurrences escaped.
    assert out.count(r"perplexity\_gap") == 2
    assert "perplexity_gap" not in out.replace(r"\_", "@")


def test_dollar_inside_display_math_does_not_mask_prose_orphan():
    """A bare `$` in prose + a self-contained display-math block
    should still detect the prose `$` as orphan."""
    src = (
        r"prose $ orphan here." "\n"
        r"\begin{equation}" "\n"
        r"x + y = z" "\n"
        r"\end{equation}"
    )
    out = balance_inline_math(src)
    assert "$" not in out.split(r"\begin{equation}")[0]
