"""Cross-section table de-duplication.

The LLM sometimes re-emits the same `\\begin{table} ... \\end{table}` in
multiple sections (e.g. a complexity-buckets table in both Experiments and
Results) — pdflatex happily compiles two tables with the same \\label, which
then poisons every `\\ref{tab:...}` in the paper.

Dedup operates on the *dict* of section-name → LaTeX body (not a single
string), so it can choose which section keeps the full table and which ones
are demoted to a bare reference comment.

Strategy — two complementary signals:

  1. **Declared ownership** (preferred): when a table has `owning_section`
     set in results.json, its `\\begin{table} ... \\label{tab:<id>}` block
     is kept ONLY in that section; occurrences elsewhere are replaced by
     a short LaTeX comment referencing the label so `\\ref` still resolves.

  2. **Structural fingerprint** (fallback for tables without owning_section
     or without `\\label`): block hash = (normalized caption, column count,
     row count). First occurrence wins; later duplicates are replaced by
     a comment that points back to the first one.

This pass is a *cross-section* operation, hence it lives outside the
per-section SANITIZE_PIPELINE and is invoked explicitly from writeup.py
after `_gen_all_sections`.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

log = logging.getLogger("hermes_sci.sanitize.tables")

# Match a full LaTeX table environment. Non-greedy; survives nested tabular.
_TABLE_ENV = re.compile(
    r"\\begin\{table\*?\}(?P<body>.*?)\\end\{table\*?\}",
    re.DOTALL,
)
_CAPTION = re.compile(r"\\caption\{(?P<cap>.*?)\}", re.DOTALL)
_LABEL = re.compile(r"\\label\{(?P<id>tab:[A-Za-z0-9_:\-]+)\}")
_COLS = re.compile(r"\\begin\{tabular\*?\}(?:\[[^\]]*\])?\{(?P<spec>[^}]*)\}")
_ROW_END = re.compile(r"\\\\")


def _norm_caption(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)      # strip \commands
    s = re.sub(r"[{}~]", " ", s)               # strip braces / tildes
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _col_count(tabular_spec: str) -> int:
    """Count columns in a tabular spec like `|l|c|c|r|` → 4."""
    spec = re.sub(r"[^lcrpXmbj]", "", tabular_spec or "")
    return len(spec)


def _fingerprint(table_src: str) -> tuple[str, int, int]:
    cap_m = _CAPTION.search(table_src)
    cap = _norm_caption(cap_m.group("cap")) if cap_m else ""
    cols_m = _COLS.search(table_src)
    cols = _col_count(cols_m.group("spec")) if cols_m else 0
    rows = len(_ROW_END.findall(table_src))
    return (cap, cols, rows)


def _label_id(table_src: str) -> Optional[str]:
    m = _LABEL.search(table_src)
    return m.group("id") if m else None


def dedup_tables(
    sections: dict[str, str],
    *,
    table_ownership: Optional[dict[str, str]] = None,
) -> tuple[dict[str, str], list[dict]]:
    """Remove duplicate `\\begin{table}` blocks across sections.

    Args:
        sections: section_key → LaTeX body.
        table_ownership: map of "tab:<id>" → owning_section. A block with a
            label in this map is kept ONLY in its owning section. Pass {}
            to rely solely on structural fingerprinting.

    Returns:
        (new_sections, events) — `events` is a list of dicts describing each
        removed block (useful for verification_report.json and progress
        callbacks).
    """
    table_ownership = table_ownership or {}
    seen_fp: dict[tuple[str, int, int], tuple[str, str]] = {}
    # label_seen: label_id → (section_key that kept it)
    label_seen: dict[str, str] = {}
    events: list[dict] = []

    def _demote(label: Optional[str], kept_in: Optional[str]) -> str:
        """Replacement text for a removed table block."""
        if label and kept_in:
            return (f"% (duplicate table \\ref{{{label}}} removed — "
                    f"rendered in section {kept_in!r})\n")
        if label:
            return f"% (duplicate table \\ref{{{label}}} removed)\n"
        return "% (duplicate table removed)\n"

    out: dict[str, str] = {}
    for sec_key, body in sections.items():
        def _replace(m: re.Match) -> str:
            block = m.group(0)
            inner = m.group("body")
            label = _label_id(inner)
            # 1. Declared ownership — label known, section mismatch → demote.
            if label and label in table_ownership:
                owner = table_ownership[label]
                if owner and owner != sec_key:
                    events.append({
                        "reason": "owning_section",
                        "label": label,
                        "found_in": sec_key,
                        "owner": owner,
                    })
                    return _demote(label, owner)
                # If this is the owning section OR unassigned, fall through.
            # 2. Label already seen in an earlier section → demote.
            if label and label in label_seen:
                kept_in = label_seen[label]
                events.append({
                    "reason": "duplicate_label",
                    "label": label,
                    "found_in": sec_key,
                    "kept_in": kept_in,
                })
                return _demote(label, kept_in)
            # 3. Structural fingerprint match → demote.
            fp = _fingerprint(inner)
            if fp != ("", 0, 0) and fp in seen_fp:
                kept_in, kept_label = seen_fp[fp]
                events.append({
                    "reason": "fingerprint",
                    "fingerprint": list(fp),
                    "found_in": sec_key,
                    "kept_in": kept_in,
                    "kept_label": kept_label,
                })
                return _demote(label or kept_label, kept_in)
            # Keep it — record.
            if label:
                label_seen[label] = sec_key
            if fp != ("", 0, 0):
                seen_fp[fp] = (sec_key, label or "")
            return block

        out[sec_key] = _TABLE_ENV.sub(_replace, body)

    if events:
        log.info("dedup_tables: removed %d duplicate block(s) across %d sections",
                 len(events), len(sections))
    return out, events
