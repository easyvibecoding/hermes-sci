# hermes-sci

Hermes-native AI Scientist — autonomous research pipeline reimagined on top of
Hermes' current LLM provider (MiniMax / OpenAI / DeepSeek / ...) with optional
delegation of coding-heavy steps to `claude -p`. No conda, no `aider-chat`, no
`google.generativeai`.

Replaces the legacy [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
and [AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) core loops
with a ~700-line Python package.

## Install

```bash
pip install -e ~/.hermes/my-skills/hermes-sci/package
```

Dependencies: `openai>=1.40`, `jinja2`, `requests`, `pypdf`, `pyyaml`.
Optional: `anthropic>=0.30` for hybrid backend (claude -p shim).

## Usage

```bash
# Generate research ideas
hermes-sci ideate --topic "test-time MoE routing" --num-ideas 5 -o ideas.json

# Write a paper (Phase 1: skip experiment, LLM writes around the idea)
hermes-sci writeup --ideas-json ideas.json --idx 0 -o ./paper

# Peer review
hermes-sci review --paper ./paper/paper.pdf --ensemble 5

# End-to-end
hermes-sci pipeline --topic "speculative decoding via KV-cache distillation" \
    --num-ideas 3 -o ./run/
```

All commands accept `--backend {minimax,hybrid}` (default `minimax`) and
read `~/.hermes/config.yaml` to auto-detect the currently selected provider.

## Architecture

| Module | Status | Purpose |
|---|---|---|
| `config.py`       | ✅ | Read Hermes `model.default`/`provider`, map to endpoint + API key |
| `llm.py`          | ✅ | Unified OpenAI-compat client + Anthropic shim + JSON extraction |
| `ideation.py`     | ✅ | Open-ended topic or workshop-CFP → JSON ideas with rubric scores |
| `review.py`       | ✅ | NeurIPS-style rubric, n-sample ensemble, median aggregation |
| `writeup.py`      | ✅ | jinja2 LaTeX template + per-section LLM prompts + pdflatex/bibtex |
| `novelty.py`      | ✅ | Semantic Scholar / OpenAlex + LLM judge (not wired into default pipeline) |
| `orchestrator.py` | ✅ | Linear pipeline: ideate → writeup → review |
| `coder.py`        | 🚧 Phase 2 | Coding agent (delegate to `claude -p`) |
| `treesearch.py`   | 🚧 Phase 3 | BFTS parallel agent (v2 core) |

## What Phase 1 gives you

Given a one-line topic, `hermes-sci pipeline` produces:
- `ideas.json` (5+ ideas with rubric scores)
- `paper/paper.tex` + `paper/paper.pdf` (7-page ICML-style layout)
- `review.json` (median scores + Accept/Reject decision)

in ~5 minutes end-to-end against MiniMax. No GPUs, no conda, no cloned repo.

## Roadmap

- **Phase 2**: `coder.py` delegates coding steps to `claude -p` — enables real
  experiments ("run this protocol on my data"). ~1 session.
- **Phase 3**: `treesearch.py` ports v2's BFTS + parallel_agent for open-ended
  exploration across 4+ parallel branches. ~1-2 sessions.
- **Phase 4**: VLM review (inspect figures in paper PDF), English-only docs,
  publish as a Hermes skill on GitHub tap.
