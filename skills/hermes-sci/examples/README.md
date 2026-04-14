# results.json examples

Three starter shapes for the `--results-json` argument. Copy the one closest
to your experiment, edit the numbers and strings, validate it:

```bash
python -m hermes_sci.results validate examples/results_ml.json
```

Then pass it to writeup or pipeline:

```bash
hermes-sci writeup --ideas-json ideas.json --idx 0 \
                   --results-json examples/results_ml.json \
                   -o ./paper/
```

| File | Shape | When to start here |
|------|-------|--------------------|
| `results_minimal.json` | 1 metric, nothing else | quick smoke test |
| `results_ml.json`      | CIFAR-100 training + ablation table | image classification, supervised ML |
| `results_nlp.json`     | WMT14 MT benchmark + quality table  | sequence-to-sequence, translation |

## Required fields

Only `metrics` is required, and each metric needs `name` + `value`. Everything
else is optional but **hurts paper quality and hallucination audit** if
omitted:

- `setup.{hardware, os, framework, dataset, model}` — verbatim strings the
  writeup must copy exactly (the LLM is told to treat them as authoritative
  and never paraphrase).
- `setup.hyperparams` — scanned for numeric claims the paper may cite.
- `tables[].id` — becomes LaTeX `\label{tab:<id>}`. Other sections can
  `\ref{tab:<id>}` instead of re-rendering.
- `tables[].owning_section` — (recommended) names the single section
  allowed to render this table; other sections must reference it. Used by
  the de-duplication pass to catch LLM-rendered copies.
- `raw_log` — free-form. Numbers found here are added to the audit
  registry so the paper can cite them without being flagged as hallucinated.

## Schema

The machine-readable schema is at
[`package/hermes_sci/data/results_schema.json`](../package/hermes_sci/data/results_schema.json).
It's also what `results.load()` validates against at runtime.

## Audit semantics

Every numeric token in the generated LaTeX is matched against
`metrics[].value`, every numeric cell in `tables[].rows`, every number in
`setup` (recursively), and every number in `raw_log`. Match tolerance:
0.5 % relative **or** 0.1 absolute, whichever is larger. A claim with no
match is flagged in `verification_report.json` and — with
`--annotate-unverified` — highlighted red in the PDF.

A paper with no `--results-json` passes an empty registry. All numeric
prose then lands in the unverified bucket, and the verification rate drops
accordingly. Supply real results for honest audit.
