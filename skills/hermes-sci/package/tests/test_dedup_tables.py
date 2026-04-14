"""Cross-section table dedup via ownership + fingerprint."""
from __future__ import annotations

from hermes_sci.sanitize.tables import dedup_tables

# Two identical tables with a shared label that both sections may emit.
_TBL = r"""
\begin{table}[h]
\centering
\caption{Results by input complexity}
\label{tab:complexity}
\begin{tabular}{|l|c|c|}
\hline
Complexity & Latency & BLEU \\
\hline
Simple & 19.4 & 28.1 \\
Medium & 28.7 & 27.6 \\
Hard   & 42.1 & 26.8 \\
\hline
\end{tabular}
\end{table}
""".strip()


def _has_begin_table(s: str) -> bool:
    return r"\begin{table}" in s


def test_owning_section_wins():
    sections = {
        "experiments": "Prose.\n" + _TBL + "\nMore.",
        "results": "Other prose.\n" + _TBL + "\nDone.",
    }
    out, events = dedup_tables(
        sections, table_ownership={"tab:complexity": "experiments"}
    )
    assert _has_begin_table(out["experiments"])
    assert not _has_begin_table(out["results"])
    assert any(e["reason"] == "owning_section" for e in events)


def test_duplicate_label_first_wins_when_no_ownership():
    sections = {"experiments": _TBL, "results": _TBL}
    out, events = dedup_tables(sections)  # no ownership map
    assert _has_begin_table(out["experiments"])
    assert not _has_begin_table(out["results"])
    assert events[0]["reason"] == "duplicate_label"


def test_fingerprint_catches_unlabeled_duplicate():
    unlab = _TBL.replace(r"\label{tab:complexity}", "")
    sections = {"experiments": unlab, "results": unlab}
    out, events = dedup_tables(sections)
    assert _has_begin_table(out["experiments"])
    assert not _has_begin_table(out["results"])
    assert events[0]["reason"] == "fingerprint"


def test_different_tables_both_survive():
    other = (_TBL
             .replace("Results by input complexity", "Ablation over dropout rate")
             .replace("tab:complexity", "tab:ablation"))
    sections = {"experiments": _TBL, "results": other}
    out, events = dedup_tables(sections)
    assert _has_begin_table(out["experiments"])
    assert _has_begin_table(out["results"])
    assert events == []


def test_no_tables_is_noop():
    sections = {"method": "Prose only.", "experiments": "More prose."}
    out, events = dedup_tables(sections)
    assert out == sections
    assert events == []


def test_demotion_leaves_ref_resolvable_comment():
    sections = {"experiments": _TBL, "results": _TBL}
    out, _ = dedup_tables(
        sections, table_ownership={"tab:complexity": "experiments"}
    )
    # Dropped block is replaced by a LaTeX comment citing the label so a
    # nearby \ref{tab:complexity} still makes sense in the prose.
    assert r"\ref{tab:complexity}" in out["results"]
    assert out["results"].lstrip().startswith("%") or \
        "% (duplicate" in out["results"]


def test_owning_section_does_not_demote_first_hit():
    sections = {"experiments": _TBL}
    out, events = dedup_tables(
        sections, table_ownership={"tab:complexity": "experiments"}
    )
    assert _has_begin_table(out["experiments"])
    assert events == []
