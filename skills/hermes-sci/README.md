# hermes-sci

Hermes-native autonomous research writer. Ship a compileable ML paper PDF
from a topic or idea in 2–5 minutes, using Hermes' currently-configured LLM
provider. No OpenAI / Anthropic API keys required.

Replaces legacy pipelines that depend on [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist):
no `conda`, no `aider-chat`, no `google.generativeai` — a ~1 200-line Python
package drives the full pipeline.

## Install as a Hermes Skill

Drop this directory under `~/.hermes/my-skills/` and add to
`~/.hermes/config.yaml`:

```yaml
skills:
  external_dirs:
    - ~/.hermes/my-skills
```

Then install the bundled Python package:

```bash
bash ~/.hermes/my-skills/hermes-sci/scripts/install.sh
```

Requirements: `pdflatex` + `chktex` (TeX Live / MacTeX).

## Usage

See `SKILL.md` for the full Quick Reference. Short version:

```bash
# 5-minute end-to-end draft with anti-hallucination audit
hermes-sci pipeline \
    --topic "your research question here" \
    --results-json ./your_results.json \
    --num-ideas 3 \
    -o ./run/
```

Outputs:
- `run/ideas.json` — scored research ideas
- `run/paper/paper.pdf` — NeurIPS-style PDF
- `run/paper/verification_report.json` — numerical claim audit
- `run/review.json` — 5-sample ensemble review

## Architecture

```
hermes-sci/
├── SKILL.md          ← Hermes skill manifest (loaded by hermes agent)
├── README.md         ← this file
├── scripts/
│   └── install.sh    ← editable pip install
└── package/
    ├── pyproject.toml
    └── hermes_sci/
        ├── config.py        — resolves Hermes provider/endpoint/API key
        ├── llm.py           — unified OpenAI-compat client + peak-hour throttle
        ├── hardware.py      — MPS-aware tier detection (high/medium/limited/cpu_only)
        ├── ideation.py      — topic → scored JSON ideas
        ├── novelty.py       — S2 / OpenAlex + LLM judge (optional)
        ├── writeup.py       — LaTeX template + 6 safety layers + pdflatex retry
        ├── review.py        — NeurIPS ensemble rubric
        ├── results.py       — author-supplied results schema
        ├── verify.py        — anti-hallucination numerical audit
        ├── orchestrator.py  — ideate → writeup → review pipeline
        ├── cli.py           — `hermes-sci {ideate|writeup|review|pipeline}`
        └── latex/
            ├── icml2024.tex.j2
            └── references.bib
```

## Design Principles

1. **Hermes-native** — read `~/.hermes/config.yaml` to inherit the current
   provider/model; zero extra credentials for MiniMax / OpenAI / DeepSeek /
   Moonshot / Gemini / Groq / Together / xAI / Zhipu.
2. **Honest compute claims** — hardware detection is passed into the prompt
   so the paper doesn't invent A100 runs from an M2 laptop.
3. **Anti-hallucination** — numerical claims in the paper are audited
   against a user-supplied results blob. Unverified numbers get flagged or
   coloured red.
4. **Peak-hour aware** — MiniMax Starter weekdays 15:00–17:30 Asia/Shanghai
   collapse to 1 concurrent call. Automatic semaphore throttle; off-peak
   defaults to 7-way parallel.
5. **Lightweight** — core deps are only `openai + jinja2`; `requests`,
   `pypdf`, `pyyaml`, `anthropic` are optional extras.

## License

Apache-2.0
