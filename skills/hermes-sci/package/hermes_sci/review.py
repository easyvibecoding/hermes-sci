"""Paper peer-review via LLM ensemble.

Mirrors the v1 NeurIPS-style review rubric: strengths / weaknesses / overall
score 1-10 / decision (Accept/Reject). Uses n=5 ensemble by default, aggregates
by median overall score + concatenated bullet lists.
"""
from __future__ import annotations

import json
import logging
import pathlib
import statistics
import time
from typing import Optional

from .config import BackendConfig
from .llm import complete_batch, extract_json
from .progress import Progress, ProgressCallback, emit, noop as _noop_progress

log = logging.getLogger("hermes_sci.review")

REVIEW_SYSTEM = """\
You are an experienced senior reviewer for a top-tier ML conference (NeurIPS /
ICML / ICLR). You evaluate papers strictly on technical merit, novelty, and
empirical rigour.
"""

REVIEW_PROMPT_TMPL = """\
Below is the full text of a research paper. Review it using the NeurIPS rubric.

<paper>
{paper_text}
</paper>

Return ONLY a JSON object wrapped in ```json ... ``` fences with these keys:

  "Summary"       3-5 sentence summary
  "Strengths"     list of 3-5 strengths
  "Weaknesses"    list of 3-5 weaknesses
  "Originality"   1-4
  "Quality"       1-4
  "Clarity"       1-4
  "Significance"  1-4
  "Soundness"     1-4
  "Presentation"  1-4
  "Contribution"  1-4
  "Overall"       integer 1-10
  "Confidence"    integer 1-5
  "Decision"      "Accept" or "Reject"
"""


def _extract_pdf_text(pdf_path: pathlib.Path) -> str:
    try:
        import pymupdf4llm  # higher fidelity if available
        return pymupdf4llm.to_markdown(str(pdf_path))
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError(
            "PDF review requires extras: pip install 'hermes-sci[review]'"
        ) from e
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def review(
    cfg: BackendConfig,
    *,
    paper: pathlib.Path | str,
    ensemble: int = 5,
    temperature: float = 0.3,
    model: Optional[str] = None,
    max_chars: int = 60000,
    progress: ProgressCallback = _noop_progress,
) -> dict:
    """Produce an aggregated review dict for a paper (PDF or plain-text path).

    Returns the aggregated review plus `all_reviews` for inspection.
    """
    paper_path = pathlib.Path(paper)
    if paper_path.suffix.lower() == ".pdf":
        text = _extract_pdf_text(paper_path)
    else:
        text = paper_path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        log.info("truncating paper text %d → %d chars", len(text), max_chars)
        text = text[:max_chars]

    prompt = REVIEW_PROMPT_TMPL.format(paper_text=text)
    log.info("reviewing: ensemble=%d model=%s", ensemble, model or cfg.model)
    emit(progress, Progress(kind="stage_start", stage="review",
                            message=f"ensemble={ensemble}"))
    t0 = time.time()
    completions = complete_batch(
        cfg, system=REVIEW_SYSTEM, user=prompt,
        model=model, temperature=temperature, max_tokens=3000, n=ensemble,
    )

    all_reviews: list[dict] = []
    for c in completions:
        parsed = extract_json(c)
        if isinstance(parsed, dict):
            all_reviews.append(parsed)

    if not all_reviews:
        emit(progress, Progress(kind="stage_end", stage="review",
                                message="no valid reviews parsed",
                                meta={"duration_s": time.time() - t0}))
        return {"error": "no valid reviews parsed", "raw": completions}

    # aggregate
    def _nums(k): return [r[k] for r in all_reviews if isinstance(r.get(k), (int, float))]

    agg = {
        "Summary": all_reviews[0].get("Summary", ""),
        "Strengths": sum((r.get("Strengths", []) for r in all_reviews), []),
        "Weaknesses": sum((r.get("Weaknesses", []) for r in all_reviews), []),
    }
    for k in ("Originality", "Quality", "Clarity", "Significance",
              "Soundness", "Presentation", "Contribution",
              "Overall", "Confidence"):
        vals = _nums(k)
        agg[k] = statistics.median(vals) if vals else None

    decisions = [r.get("Decision", "") for r in all_reviews]
    accepts = sum(1 for d in decisions if str(d).lower().startswith("accept"))
    agg["Decision"] = "Accept" if accepts > len(decisions) / 2 else "Reject"
    agg["ensemble_size"] = len(all_reviews)
    agg["all_reviews"] = all_reviews
    emit(progress, Progress(kind="stage_end", stage="review",
                            message=f"{agg['Decision']} overall={agg.get('Overall')}",
                            meta={"duration_s": time.time() - t0,
                                  "ensemble_size": len(all_reviews),
                                  "decision": agg["Decision"]}))
    return agg


def save_review(r: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
