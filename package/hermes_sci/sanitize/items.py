"""Wrap orphan \\item lines in a \\begin{itemize}...\\end{itemize}.

Logic-heavy (stateful line walk), so this pass stays in Python rather than
being expressed as a single regex rule.
"""
from __future__ import annotations

import re

_LIST_OPEN = re.compile(r"\\begin\{(itemize|enumerate|description)\}")
_LIST_CLOSE = re.compile(r"\\end\{(itemize|enumerate|description)\}")


def wrap_lonely_items(s: str) -> str:
    lines = s.split("\n")
    out: list[str] = []
    depth = 0
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if _LIST_OPEN.match(stripped):
            depth += 1
            out.append(ln)
            i += 1
            continue
        if _LIST_CLOSE.match(stripped):
            depth = max(0, depth - 1)
            out.append(ln)
            i += 1
            continue
        if depth == 0 and stripped.startswith("\\item"):
            block: list[str] = []
            while i < len(lines) and (
                lines[i].strip().startswith("\\item") or lines[i].strip() == ""
            ):
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out.append(r"\begin{itemize}")
            out.extend(block)
            out.append(r"\end{itemize}")
            continue
        out.append(ln)
        i += 1
    return "\n".join(out)
