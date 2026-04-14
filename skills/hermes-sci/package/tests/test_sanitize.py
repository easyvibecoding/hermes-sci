"""Sanitize pipeline smoke tests — one per pass + end-to-end.

Each pass is a pure `str -> str` transform, so tests are fast and
deterministic. These are smoke tests, not exhaustive — regressions
on specific strings the pipeline has historically choked on.
"""
from __future__ import annotations

from hermes_sci.sanitize import sanitize_latex
from hermes_sci.sanitize.cjk import strip_cjk
from hermes_sci.sanitize.fences import strip_code_fences
from hermes_sci.sanitize.markdown import md_to_latex
from hermes_sci.sanitize.reasoning import strip_reasoning


def test_strip_cjk_removes_chinese():
    assert strip_cjk("prose 中文 mix") == "prose  mix"


def test_strip_cjk_preserves_ascii():
    assert strip_cjk("ASCII only, no changes") == "ASCII only, no changes"


def test_strip_code_fences_unwraps():
    src = "text\n```latex\n\\section{Intro}\n```\nmore"
    out = strip_code_fences(src)
    assert r"\section{Intro}" in out
    assert "```" not in out


def test_md_to_latex_bold_italic():
    out = md_to_latex("This is **bold** and *italic* text.")
    assert r"\textbf{bold}" in out
    assert r"\textit{italic}" in out or r"\emph{italic}" in out


def test_strip_reasoning_removes_think_tags():
    src = "<think>I should explain</think>\\section{A}"
    out = strip_reasoning(src)
    assert "<think>" not in out
    assert r"\section{A}" in out


def test_full_pipeline_idempotent_on_clean_input():
    clean = r"\subsection{Results}" + "\nOur method achieves 28.3 BLEU."
    once = sanitize_latex(clean)
    twice = sanitize_latex(once)
    assert once == twice


def test_full_pipeline_handles_messy_llm_output():
    messy = (
        "<think>Let me plan...</think>\n"
        "```latex\n"
        "# Introduction\n"
        "This is **our** contribution 中文.\n"
        "```"
    )
    out = sanitize_latex(messy)
    assert "<think>" not in out
    assert "```" not in out
    assert "中文" not in out
    assert r"\textbf{our}" in out
