"""User-supplied experiment results schema.

Phase 3: instead of auto-running experiments, the user provides a results
blob (JSON or Markdown). We parse it into a structured Results object so
writeup can cite exact numbers, and verify.py can audit the paper's
numerical claims against it.

JSON schema (all fields optional except `metrics`):

    {
      "setup": {
        "hardware": "Apple M2 MPS 16GB",
        "framework": "PyTorch 2.7",
        "dataset": "WMT14 En-De",
        "model": "Switch Transformer 86M",
        "hyperparams": {"lr": 1e-4, "epochs": 20, ...}
      },
      "metrics": [
        {"name": "BLEU", "value": 28.3, "unit": "",      "method": "HAMR",      "split": "test",       "context": "simple bucket"},
        {"name": "latency", "value": 19.4, "unit": "ms", "method": "HAMR",      "split": "test"},
        {"name": "BLEU", "value": 28.1, "unit": "",      "method": "top-2",     "split": "test",       "context": "simple bucket"}
      ],
      "tables": [
        {"id": "complexity_buckets", "caption": "Results by input complexity",
         "headers": ["Complexity", "Latency", "BLEU", "Experts"],
         "rows": [["Simple", "19.4", "28.1", "1.06"], ...]}
      ],
      "raw_log": "stdout / stderr dump (used as evidence trail)"
    }

A Markdown variant is also accepted — headings become sections, tables are
parsed by pipe syntax, loose numbers in prose are still allowed but will be
flagged as low-confidence by verify.py.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import re
from typing import Any, Optional


@dataclasses.dataclass
class Metric:
    name: str
    value: float
    unit: str = ""
    method: str = ""
    split: str = ""
    context: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Metric":
        try:
            v = float(d.get("value"))
        except (TypeError, ValueError) as e:
            raise ValueError(f"metric {d.get('name')!r} has non-numeric value") from e
        return cls(
            name=str(d.get("name", "")).strip(),
            value=v,
            unit=str(d.get("unit", "")).strip(),
            method=str(d.get("method", "")).strip(),
            split=str(d.get("split", "")).strip(),
            context=str(d.get("context", "")).strip(),
        )


@dataclasses.dataclass
class Table:
    id: str
    caption: str
    headers: list[str]
    rows: list[list[str]]

    @classmethod
    def from_dict(cls, d: dict) -> "Table":
        return cls(
            id=str(d.get("id", "")).strip(),
            caption=str(d.get("caption", "")).strip(),
            headers=[str(h) for h in (d.get("headers") or [])],
            rows=[[str(c) for c in row] for row in (d.get("rows") or [])],
        )


@dataclasses.dataclass
class Results:
    setup: dict[str, Any]
    metrics: list[Metric]
    tables: list[Table]
    raw_log: str = ""

    def all_numeric_values(self) -> list[float]:
        """Every number attestable to this results blob (for verify audit).

        Sources:
          - metrics[*].value
          - tables[*].rows[*][*] (numeric cells)
          - setup values (string-embedded floats: "86.3M params" → 86.3)
          - setup.hyperparams values
          - raw_log scanned for numbers
        """
        out: list[float] = [m.value for m in self.metrics]
        for t in self.tables:
            for row in t.rows:
                for cell in row:
                    try:
                        out.append(float(cell))
                    except (TypeError, ValueError):
                        continue
        # setup — scan strings for floats; walk dicts recursively
        out.extend(_scan_numbers(self.setup))
        if self.raw_log:
            out.extend(_scan_numbers(self.raw_log))
        return out

    def to_prompt_context(self) -> str:
        """Render as compact text for LLM prompt. Keeps specific numbers
        verbatim so the writer can drop them into prose."""
        lines = ["EXPERIMENT RESULTS (author-supplied — use these exact numbers):", ""]

        # Verbatim string block — names/versions/hardware to be copied EXACTLY,
        # not paraphrased from the LLM's memory.
        if self.setup:
            verbatim = []
            for k, v in self.setup.items():
                if isinstance(v, str) and v.strip():
                    verbatim.append((k, v.strip()))
            if verbatim:
                lines.append("VERBATIM STRINGS (copy exactly — do NOT paraphrase or re-transcribe from memory):")
                for k, v in verbatim:
                    lines.append(f"  {k}: {v}")
                lines.append("")
            lines.append("Setup:")
            for k, v in self.setup.items():
                if isinstance(v, dict):
                    sub = ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                    lines.append(f"  {k}: {sub}")
                else:
                    lines.append(f"  {k}: {v}")
            lines.append("")
        if self.metrics:
            lines.append("Metrics:")
            for m in self.metrics:
                ctx = f" ({m.context})" if m.context else ""
                sp = f", split={m.split}" if m.split else ""
                meth = f"{m.method}: " if m.method else ""
                lines.append(f"  - {meth}{m.name} = {m.value}{m.unit}{sp}{ctx}")
            lines.append("")
        if self.tables:
            lines.append("Tables:")
            lines.append("  LaTeX labels to use: " +
                         ", ".join(f"tab:{t.id}" for t in self.tables))
            lines.append("  When you render a \\begin{table}, add "
                         "\\label{tab:<id>} matching the ID below, and "
                         "cross-reference with Table~\\ref{tab:<id>}. "
                         "Never write 'Table ??' or a bare \\ref{}.")
            for t in self.tables:
                lines.append("")
                lines.append(f"  [\\label{{tab:{t.id}}}] {t.caption}")
                lines.append(f"    | {' | '.join(t.headers)} |")
                for row in t.rows:
                    lines.append(f"    | {' | '.join(row)} |")
            lines.append("")
        if self.raw_log:
            lines.append("Raw log (evidence):")
            lines.append(self.raw_log[:4000])
        return "\n".join(lines)


_NUM_IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?")


def _scan_numbers(obj) -> list[float]:
    acc: list[float] = []
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM_IN_TEXT.finditer(obj):
            try:
                acc.append(float(m.group(0)))
            except ValueError:
                continue
    elif isinstance(obj, dict):
        for v in obj.values():
            acc.extend(_scan_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            acc.extend(_scan_numbers(v))
    return acc


# ── Loaders ──────────────────────────────────────────────────────────


def load_json(path: pathlib.Path | str) -> Results:
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    return from_dict(data)


def from_dict(data: dict) -> Results:
    metrics = [Metric.from_dict(m) for m in (data.get("metrics") or [])]
    tables = [Table.from_dict(t) for t in (data.get("tables") or [])]
    return Results(
        setup=dict(data.get("setup") or {}),
        metrics=metrics, tables=tables,
        raw_log=str(data.get("raw_log") or ""),
    )


# ── Markdown fallback ────────────────────────────────────────────────


_MD_NUM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|ms|s|tok\/s|MB|GB)?")
_MD_TABLE_SEP = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*:?-+:?\s*\|?\s*$")


def load_markdown(path: pathlib.Path | str) -> Results:
    """Best-effort parse of a markdown results file.

    Extracts pipe tables and numeric lines. Prose stays in raw_log. This is
    intentionally permissive — for strict audit, supply JSON.
    """
    text = pathlib.Path(path).read_text(encoding="utf-8")
    tables: list[Table] = []
    metrics: list[Metric] = []

    # Pipe tables
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if "|" in lines[i] and i + 1 < len(lines) and _MD_TABLE_SEP.match(lines[i + 1]):
            headers = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            tables.append(Table(id=f"table{len(tables)+1}", caption="",
                                headers=headers, rows=rows))
            i = j
            continue
        i += 1

    # Simple "Name: value" lines  (e.g., "BLEU: 28.3", "latency: 19.4 ms")
    for line in lines:
        m = re.match(r"^\s*([A-Za-z][\w\s/().-]{0,40})\s*[:=]\s*"
                     r"(-?\d+(?:\.\d+)?)\s*(%|ms|s|tok/s|MB|GB)?\s*$", line)
        if m:
            name, val, unit = m.group(1).strip(), m.group(2), (m.group(3) or "")
            try:
                metrics.append(Metric(name=name, value=float(val), unit=unit))
            except ValueError:
                continue

    return Results(setup={}, metrics=metrics, tables=tables, raw_log=text)


def load(path: pathlib.Path | str) -> Results:
    p = pathlib.Path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        return load_json(p)
    return load_markdown(p)
