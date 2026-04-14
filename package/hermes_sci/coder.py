"""Coding agent — placeholder for Phase 2.

The design is: delegate coding-heavy steps to `claude -p` subprocess (via the
hybrid claude_proxy) or to claude-code SDK directly. Skipping for Phase 1.
"""
from __future__ import annotations


def run_coding_loop(*_args, **_kwargs):
    raise NotImplementedError(
        "hermes_sci.coder ships in Phase 2. Current Phase-1 pipeline skips "
        "the experiment stage."
    )
