"""Escape prose-level LaTeX specials (%, &, <, >) without touching math or
tabular content.

Strategy: tokenise on math/tabular/cite/ref segments (kept verbatim) and
escape each prose chunk between them.

Also fixes a systematic LLM over-escape: `\\&` inside tabular bodies
(where `&` is the column separator) is un-escaped.

Logic-heavy. The segment boundaries are complex enough that pulling them
into a YAML rule-set would lose clarity; prefer code here.
"""
from __future__ import annotations

import re

_MATH_SEGMENT = re.compile(
    r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}"
    r"|\\begin\{align\*?\}.*?\\end\{align\*?\}"
    r"|\\begin\{gather\*?\}.*?\\end\{gather\*?\}"
    r"|\\begin\{multline\*?\}.*?\\end\{multline\*?\}"
    r"|\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}"
    r"|\\begin\{array\}.*?\\end\{array\}"
    r"|\\begin\{matrix\}.*?\\end\{matrix\}"
    r"|\\begin\{pmatrix\}.*?\\end\{pmatrix\}"
    r"|\\begin\{bmatrix\}.*?\\end\{bmatrix\}"
    r"|\\\[.*?\\\]"
    r"|\$\$.*?\$\$"
    r"|\$[^$\n]+\$"
    r"|\\cite[tp]?\*?\{[^}]*\}"
    r"|\\ref\{[^}]*\}|\\label\{[^}]*\}|\\url\{[^}]*\}|\\href\{[^}]*\}\{[^}]*\}",
    re.DOTALL,
)

_TABULAR_ENV = re.compile(
    r"\\begin\{(tabular\*?|array|matrix|pmatrix|bmatrix)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)

_BARE_PERCENT = re.compile(r"(?<!\\)%")
_BARE_LT = re.compile(r"(?<![\\$])<(?!=)(?=\s|\d|[A-Za-z])")
_BARE_GT = re.compile(r"(?<![\\$])>(?!=)(?=\s|\d|[A-Za-z])")
_BARE_AMP = re.compile(r"(?<!\\)&(?![A-Za-z]{2,6};)")
# Prose underscores: `epsilon_target`, `file_name` → must be \_ or LaTeX tries
# to open a subscript (which only works in math mode). Skip `\_` (already
# escaped) and `_{...}` (common math-like usage the LLM forgot to wrap in $).
_BARE_UNDERSCORE = re.compile(r"(?<!\\)_(?!\{)")


def _unescape_table_amps(s: str) -> str:
    def repl(m: re.Match) -> str:
        body = m.group(2).replace(r"\&", "&")
        return f"\\begin{{{m.group(1)}}}{body}\\end{{{m.group(1)}}}"
    return _TABULAR_ENV.sub(repl, s)


def _escape_chunk(prose: str) -> str:
    prose = _BARE_PERCENT.sub(r"\\%", prose)
    prose = _BARE_AMP.sub(r"\\&", prose)
    prose = _BARE_LT.sub(r"$<$", prose)
    prose = _BARE_GT.sub(r"$>$", prose)
    prose = _BARE_UNDERSCORE.sub(r"\\_", prose)
    return prose


def escape_prose_specials(s: str) -> str:
    s = _unescape_table_amps(s)
    chunks: list[str] = []
    last = 0
    for m in _MATH_SEGMENT.finditer(s):
        chunks.append(_escape_chunk(s[last:m.start()]))
        chunks.append(m.group(0))
        last = m.end()
    chunks.append(_escape_chunk(s[last:]))
    return "".join(chunks)
