"""Prose-level underscore escape.

Regression for the demo paper pipeline run where the LLM wrote
`(lambda and epsilon_target)` in running prose. Unescaped `_` outside
math mode makes pdflatex emit "Missing $ inserted" and bail.
"""
from __future__ import annotations

from hermes_sci.sanitize.escape import escape_prose_specials


def test_prose_underscore_escaped():
    src = "hyperparameters (lambda and epsilon_target) require tuning"
    out = escape_prose_specials(src)
    assert r"epsilon\_target" in out
    assert "epsilon_target" not in out.replace(r"\_", "@")


def test_inline_math_underscores_preserved():
    src = r"the threshold $\epsilon_{target}$ is set to $x_1$"
    out = escape_prose_specials(src)
    # Everything inside $...$ survives untouched.
    assert r"$\epsilon_{target}$" in out
    assert r"$x_1$" in out


def test_display_math_underscores_preserved():
    src = (
        r"See eq:" "\n"
        r"\begin{equation}" "\n"
        r"  f(x) = w_i^\top x_1" "\n"
        r"\end{equation}" "\n"
        r"end."
    )
    out = escape_prose_specials(src)
    assert "w_i^\\top x_1" in out  # untouched inside equation
    assert out.endswith("end.")


def test_already_escaped_underscore_untouched():
    src = r"see file\_name for details"
    out = escape_prose_specials(src)
    assert out == src


def test_subscript_like_underscore_in_prose_kept_raw():
    r"""`word_{sub}` in prose probably means the LLM forgot `$...$`.
    We leave it alone — escaping to `word\_{sub}` would look worse, and
    `_{` is almost always followed by math content; the author can fix
    the math-mode wrapping themselves. Documented behavior, not an oversight.
    """
    src = r"the term x_{target} appears unescaped"
    out = escape_prose_specials(src)
    # `_{` is NOT escaped — regex requires lookahead `(?!\{)`.
    assert r"x_{target}" in out


def test_underscore_in_textbf_argument_escaped():
    src = r"\textbf{epsilon_target} is our hyperparam"
    out = escape_prose_specials(src)
    assert r"\textbf{epsilon\_target}" in out


def test_all_prose_specials_together():
    src = "100% improvement in R&D; compare A<B, use file_name for x>y"
    out = escape_prose_specials(src)
    assert r"100\%" in out
    assert r"R\&D" in out
    assert r"file\_name" in out
    assert "$<$" in out and "$>$" in out
