"""Research idea generation.

Two modes:
  - workshop: supply a workshop call-for-papers markdown; LLM generates ideas
    that fit the scope. (v1-style)
  - open: free-form topic string; LLM explores arbitrary angles. (v2-style
    template-free)

Output: list[dict] ideas with stable schema:
    {Name, Title, Short Hypothesis, Related Work, Abstract,
     Experiments, Risk Factors and Limitations, Interestingness,
     Feasibility, Novelty}
"""
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Literal, Optional

from .config import BackendConfig
from .llm import complete, extract_json

log = logging.getLogger("hermes_sci.ideation")

Mode = Literal["workshop", "open"]

IDEATION_SYSTEM = """\
You are an experienced AI/ML researcher helping plan a new research project.
Your goal is to propose concrete, novel, feasible ideas that could become
publishable papers. Favour depth over breadth; every idea must include a
testable hypothesis and a minimal experiment plan.
"""

SCHEMA = """\
Return ONLY a JSON array of idea objects wrapped in ```json ... ``` fences.
Each idea object must have ALL of these keys:

  "Name"                          short_snake_case_id
  "Title"                         full paper-style title
  "Short Hypothesis"              1-2 sentences, falsifiable
  "Related Work"                  1 paragraph citing prior work
  "Abstract"                      4-6 sentences
  "Experiments"                   numbered list, each step concrete
  "Risk Factors and Limitations"  bullet list
  "Interestingness"               integer 1-10
  "Feasibility"                   integer 1-10
  "Novelty"                       integer 1-10
"""


def _prompt_workshop(workshop_md: str, num_ideas: int, reflect: bool) -> str:
    reflect_line = (
        f"\nAfter your first draft, critique and revise once before answering. "
        if reflect else ""
    )
    return f"""\
Below is a workshop's call-for-papers. Propose {num_ideas} distinct research
ideas that would fit this venue.

<workshop>
{workshop_md.strip()}
</workshop>
{reflect_line}
{SCHEMA}
"""


def _prompt_open(topic: str, num_ideas: int, reflect: bool) -> str:
    reflect_line = (
        f"\nAfter your first draft, critique and revise once before answering. "
        if reflect else ""
    )
    return f"""\
Topic for open-ended research ideation:

    {topic.strip()}

Propose {num_ideas} distinct research ideas. Each should take the topic in a
meaningfully different direction.{reflect_line}

{SCHEMA}
"""


@dataclasses.dataclass
class Idea:
    name: str
    title: str
    hypothesis: str
    abstract: str
    experiments: str
    related_work: str
    risks: str
    interestingness: int
    feasibility: int
    novelty: int
    raw: dict  # original LLM object, preserved

    @classmethod
    def from_dict(cls, d: dict) -> "Idea":
        def g(k, dflt=""): return d.get(k, dflt)
        return cls(
            name=str(g("Name", "untitled")),
            title=str(g("Title")),
            hypothesis=str(g("Short Hypothesis")),
            abstract=str(g("Abstract")),
            experiments=str(g("Experiments")),
            related_work=str(g("Related Work")),
            risks=str(g("Risk Factors and Limitations")),
            interestingness=int(g("Interestingness", 0) or 0),
            feasibility=int(g("Feasibility", 0) or 0),
            novelty=int(g("Novelty", 0) or 0),
            raw=d,
        )


def ideate(
    cfg: BackendConfig,
    *,
    mode: Mode = "open",
    topic: Optional[str] = None,
    workshop_md_path: Optional[pathlib.Path] = None,
    num_ideas: int = 5,
    reflect: bool = True,
    model: Optional[str] = None,
) -> list[Idea]:
    """Generate research ideas.

    Args:
        cfg: BackendConfig from resolve_backend().
        mode: 'open' (topic string) or 'workshop' (CFP markdown).
        topic: required for mode='open'.
        workshop_md_path: required for mode='workshop'.
        num_ideas: target count.
        reflect: ask for self-critique before final answer.

    Returns:
        list of Idea dataclasses. Empty list on parse failure.
    """
    if mode == "open":
        if not topic:
            raise ValueError("mode='open' requires topic=...")
        user = _prompt_open(topic, num_ideas, reflect)
    else:
        if not workshop_md_path:
            raise ValueError("mode='workshop' requires workshop_md_path=...")
        workshop_md = pathlib.Path(workshop_md_path).read_text(encoding="utf-8")
        user = _prompt_workshop(workshop_md, num_ideas, reflect)

    log.info("ideating: mode=%s num_ideas=%d model=%s", mode, num_ideas, model or cfg.model)
    text, _ = complete(cfg, system=IDEATION_SYSTEM, user=user, model=model,
                       temperature=0.9, max_tokens=6000)

    parsed = extract_json(text)
    if not isinstance(parsed, list):
        log.error("ideation output was not a JSON array; got %s", type(parsed).__name__)
        return []

    ideas: list[Idea] = []
    for d in parsed:
        if not isinstance(d, dict):
            continue
        try:
            ideas.append(Idea.from_dict(d))
        except (TypeError, ValueError) as e:
            log.warning("skipping malformed idea: %s", e)
    return ideas


def save_ideas(ideas: list[Idea], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_ideas": len(ideas),
        "ideas": [i.raw for i in ideas],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
