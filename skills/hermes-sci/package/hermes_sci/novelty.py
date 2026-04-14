"""Novelty check via Semantic Scholar / OpenAlex + LLM judgement.

For Phase 1 this is a thin wrapper that searches S2 for the top related work
and asks the LLM to judge whether the idea is novel. Not wired into the
default pipeline yet — available via hermes_sci.novelty.check(idea).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover
    requests = None  # resolved lazily in functions below

from .config import BackendConfig
from .llm import complete, extract_json

log = logging.getLogger("hermes_sci.novelty")

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_SEARCH = "https://api.openalex.org/works"


def _s2_search(query: str, limit: int = 5, api_key: Optional[str] = None) -> list[dict]:
    if requests is None:
        log.warning("novelty needs extras: pip install 'hermes-sci[novelty]'")
        return []
    headers = {"x-api-key": api_key} if api_key else {}
    try:
        r = requests.get(S2_SEARCH,
                         params={"query": query, "limit": limit,
                                 "fields": "title,abstract,year,authors"},
                         headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json().get("data", []) or []
    except Exception as e:  # noqa: BLE001 — requests.RequestException or None
        log.warning("S2 search failed: %s", e)
    return []


def _openalex_search(query: str, limit: int = 5, mail: Optional[str] = None) -> list[dict]:
    if requests is None:
        log.warning("novelty needs extras: pip install 'hermes-sci[novelty]'")
        return []
    try:
        r = requests.get(OPENALEX_SEARCH,
                         params={"search": query, "per-page": limit,
                                 "mailto": mail or ""},
                         timeout=15)
        if r.status_code == 200:
            hits = r.json().get("results", []) or []
            return [{"title": h.get("title"),
                     "abstract": (h.get("abstract_inverted_index") and
                                  " ".join(h["abstract_inverted_index"].keys())),
                     "year": h.get("publication_year")} for h in hits]
    except Exception as e:  # noqa: BLE001 — requests.RequestException or None
        log.warning("OpenAlex search failed: %s", e)
    return []


NOVELTY_SYSTEM = """\
You are a research novelty auditor. Given an idea and a list of related papers,
decide whether the idea is sufficiently novel to warrant a new paper.
"""

NOVELTY_PROMPT_TMPL = """\
IDEA:
  Title: {title}
  Hypothesis: {hypothesis}
  Abstract: {abstract}

CANDIDATE PRIOR WORK:
{prior}

Return ONLY a JSON object wrapped in ```json ... ``` with keys:
  "Novel"           boolean
  "Reasoning"       1-3 sentences
  "Closest Prior"   list of titles (at most 3)
  "Score"           integer 1-10 (10 = highly novel)
"""


def check(
    cfg: BackendConfig,
    idea: dict,
    *,
    engine: str = "s2",
    mail: Optional[str] = None,
    s2_key: Optional[str] = None,
    model: Optional[str] = None,
) -> dict:
    """Return a novelty judgement dict."""
    q = (idea.get("Title") or "") + " " + (idea.get("Short Hypothesis") or "")
    if engine == "s2":
        hits = _s2_search(q, api_key=s2_key or os.environ.get("S2_API_KEY"))
    else:
        hits = _openalex_search(q, mail=mail or os.environ.get("OPENALEX_MAIL_ADDRESS"))
    prior = "\n".join(f"- ({h.get('year','?')}) {h.get('title','')}: "
                      f"{(h.get('abstract') or '')[:240]}" for h in hits[:5]) or "(none found)"

    prompt = NOVELTY_PROMPT_TMPL.format(
        title=idea.get("Title", ""),
        hypothesis=idea.get("Short Hypothesis", ""),
        abstract=idea.get("Abstract", ""),
        prior=prior,
    )
    text, _ = complete(cfg, system=NOVELTY_SYSTEM, user=prompt, model=model,
                       temperature=0.2, max_tokens=800)
    parsed = extract_json(text)
    if not isinstance(parsed, dict):
        return {"error": "parse failed", "raw": text, "prior": hits}
    parsed["prior_hits"] = hits[:5]
    return parsed
