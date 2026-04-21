# easyvibecoding/hermes-sci

Skills hub for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Install via:

```bash
hermes skills tap add easyvibecoding/hermes-sci
hermes skills install easyvibecoding/hermes-sci/hermes-sci
```

## Skills

- [`skills/hermes-sci/`](skills/hermes-sci/) — Hermes-native autonomous
  research paper writer: ideation → LaTeX writeup → peer review, wired
  to whatever LLM provider Hermes is currently configured to use.
  Anti-hallucination numerical-claim audit, Apple Silicon MPS-aware,
  modular sanitize pipeline with rules-as-data YAML.

## Related projects

- [**vibe-sci**](https://github.com/easyvibecoding/vibe-sci) — provider-neutral
  spin-off of `hermes-sci` that removes the Hermes-runtime coupling. Same
  ideation → LaTeX writeup → peer-review → anti-hallucination pipeline,
  but runs against `claude -p` subprocess, any OpenAI-compatible endpoint,
  or a rule-based fallback. Use `vibe-sci` outside Hermes workflows; use
  `hermes-sci` (this repo) when you want Hermes Agent to orchestrate.

## License

Apache-2.0.
