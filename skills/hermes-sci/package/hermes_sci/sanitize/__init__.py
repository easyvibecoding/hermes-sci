"""hermes_sci.sanitize — modular LaTeX sanitize pipeline.

Usage:
    from hermes_sci.sanitize import sanitize_latex
    cleaned = sanitize_latex(raw_llm_output)

See pipeline.py for the ordered pass list; most regex rules live in
hermes_sci/data/*.yaml and can be extended without Python changes.
"""
from .pipeline import SANITIZE_PIPELINE, sanitize_latex

__all__ = ["SANITIZE_PIPELINE", "sanitize_latex"]
