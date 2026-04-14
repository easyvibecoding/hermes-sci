"""hermes-sci — Hermes-native AI Scientist.

Public API:
    from hermes_sci import ideate, writeup, review, run_pipeline
    from hermes_sci.config import resolve_backend
    from hermes_sci.llm import make_client
"""
from __future__ import annotations

__version__ = "0.1.0"

from .ideation import ideate
from .review import review
from .writeup import writeup
from .orchestrator import run_pipeline

__all__ = ["ideate", "writeup", "review", "run_pipeline", "__version__"]
