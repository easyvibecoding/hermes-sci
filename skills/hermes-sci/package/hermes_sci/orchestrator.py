"""End-to-end pipeline orchestration.

Phase 1 (this commit):
    ideate → pick top idea → writeup → review
    (experiment step stubbed — accepts user-supplied results markdown)

Phase 2 (next commit): coder + experiment runner
Phase 3: BFTS tree search (v2)
"""
from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Optional

from .config import BackendConfig
from .ideation import ideate, save_ideas, Idea
from .results import Results
from .review import review, save_review
from .writeup import writeup

log = logging.getLogger("hermes_sci.orchestrator")


def _rank_ideas(ideas: list[Idea]) -> list[Idea]:
    """Sort by (Interestingness + Novelty + Feasibility) descending."""
    return sorted(ideas, key=lambda i: -(i.interestingness + i.novelty + i.feasibility))


def run_pipeline(
    cfg: BackendConfig,
    *,
    topic: str,
    out_dir: pathlib.Path,
    num_ideas: int = 3,
    results: Optional[str | Results] = None,
    results_markdown: Optional[str] = None,   # deprecated alias for results=str
    skip_experiment: bool = True,   # Phase-1 default
    skip_review: bool = False,
    skip_compile: bool = False,
    model: Optional[str] = None,
    critique: bool = True,
    coherence: bool = False,
    parallel: bool = True,
    concurrency: Optional[int] = None,
    annotate_unverified_claims: bool = False,
) -> dict:
    """Full: ideate → pick best → writeup → review.

    Returns a dict with paths to every produced artefact.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"out_dir": str(out_dir), "stages": {}}
    t0 = time.time()

    # Stage 1: ideation
    log.info("stage 1: ideation")
    ideas = ideate(cfg, mode="open", topic=topic, num_ideas=num_ideas, model=model)
    if not ideas:
        report["error"] = "ideation produced zero ideas"
        return report
    save_ideas(ideas, out_dir / "ideas.json")
    report["stages"]["ideation"] = {
        "ideas_json": str(out_dir / "ideas.json"),
        "num_ideas": len(ideas),
    }

    picked = _rank_ideas(ideas)[0]
    log.info("picked idea: %s (score=%d)", picked.name,
             picked.interestingness + picked.novelty + picked.feasibility)

    # Stage 2: results integration (Phase 3 — user-supplied, not auto-run)
    results_payload = results if results is not None else results_markdown
    if results_payload is None and skip_experiment:
        log.info("stage 2: no results supplied → placeholder writeup")
        results_payload = (
            "No author-supplied experiment results. The writeup will describe "
            "the planned protocol qualitatively without numerical claims."
        )
    if isinstance(results_payload, Results):
        exp_info = {"type": "structured",
                    "metrics": len(results_payload.metrics),
                    "tables": len(results_payload.tables)}
    elif isinstance(results_payload, str):
        exp_info = {"type": "markdown", "chars": len(results_payload)}
    else:
        exp_info = {"type": "none"}
    report["stages"]["experiment"] = exp_info

    # Stage 3: writeup
    log.info("stage 3: writeup")
    paper_dir = out_dir / "paper"
    w = writeup(cfg, idea=picked.raw, out_dir=paper_dir,
                results=results_payload, model=model, skip_compile=skip_compile,
                critique=critique, coherence=coherence, parallel=parallel,
                concurrency=concurrency,
                annotate_unverified_claims=annotate_unverified_claims)
    report["stages"]["writeup"] = w

    # Stage 4: review (only if PDF was produced)
    pdf_path = w.get("pdf")
    if skip_review or not pdf_path:
        log.info("stage 4: review SKIPPED (skip_review=%s pdf=%s)",
                 skip_review, bool(pdf_path))
    else:
        log.info("stage 4: review")
        rev = review(cfg, paper=pdf_path, model=model, ensemble=3)
        save_review(rev, out_dir / "review.json")
        report["stages"]["review"] = {
            "review_json": str(out_dir / "review.json"),
            "overall": rev.get("Overall"),
            "decision": rev.get("Decision"),
        }

    report["elapsed_sec"] = round(time.time() - t0, 2)
    (out_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
