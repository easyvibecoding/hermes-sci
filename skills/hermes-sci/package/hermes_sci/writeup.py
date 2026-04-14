"""Paper writeup — LaTeX template + LLM sections + quality passes.

Phase 2 enhancements (all optional toggles, all inside writeup.py):
  1. Hardware-aware prompt hint (so LLM doesn't claim 8xA100 on a laptop)
  2. Async parallel section generation (2-5x speedup)
  3. Section self-critique (draft → critique → revise)
  4. Coherence pass (transitions between sections)
  5. Citation whitelist (strip \\cite{...} keys not in references.bib)
  6. pdflatex-log-driven retry (target the one bad section, not full paper)

Sanitize passes moved to hermes_sci.sanitize (Phase 4 C). Regex rules live in
hermes_sci/data/*.yaml so community contributions don't need Python edits.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import pathlib
import re
import shutil
import subprocess
import time
from typing import Callable, Optional

from jinja2 import Environment, FileSystemLoader

from .config import BackendConfig
from .hardware import HardwareProfile, detect as detect_hardware, hint_for_prompt
from .llm import acomplete, complete, recommended_concurrency, is_minimax_peak
from .progress import Progress, ProgressCallback, emit, noop as _noop_progress
from .results import Results, from_dict as results_from_dict, load as results_load
from .sanitize import sanitize_latex as _sanitize_latex
from .sanitize.tables import dedup_tables as _dedup_tables
from .verify import audit as verify_audit, annotate_unverified

log = logging.getLogger("hermes_sci.writeup")

TEMPLATE_DIR = pathlib.Path(__file__).parent / "latex"
PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"


def _read_prompt(name: str) -> str:
    """Load a prompt file bundled under hermes_sci/prompts/."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


# ── Prompts (loaded from hermes_sci/prompts/*.md + .json) ──────────

SECTION_SYSTEM = _read_prompt("section_system.md")
SECTION_PROMPTS = json.loads(_read_prompt("section_instructions.json"))
CRITIQUE_SYSTEM = _read_prompt("critique_system.md")
CRITIQUE_USER_TMPL = _read_prompt("critique_user.md")
COHERENCE_SYSTEM = _read_prompt("coherence_system.md")
COHERENCE_USER_TMPL = _read_prompt("coherence_user.md")
RETRY_SYSTEM = _read_prompt("retry_system.md")

# ── Non-sanitize helpers ─────────────────────────────────────────────
# Sanitize passes (markdown→LaTeX, CJK strip, package fallbacks, prose
# escape, etc.) now live in hermes_sci.sanitize and are invoked via
# `_sanitize_latex` imported at the top of this file.

_CITE_CALL = re.compile(r"\\cite[tp]?\*?\{([^}]*)\}")
_EMPTY_CITE = re.compile(r"\\cite[tp]?\*?\{[\s,]*\}")
_BIB_KEY = re.compile(r"@\w+\{\s*([A-Za-z0-9_:\-]+)\s*,")

# Meta-commentary patterns the LLM sometimes leaks into critique/retry output.
_META_PROSE = re.compile(
    r"(?im)^\s*(looking at|let me|i don[' ]t see|hmm|wait|i notice|actually,?|"
    r"here'?s the|here is the|this is|i'?ll|i will|it seems|oh[,!]? |ok[,!]? |"
    r"let'?s|alright|on second thought)"
)


def _looks_like_latex(s: str) -> bool:
    """Heuristic: does the output look like LaTeX body, not English commentary?"""
    s_stripped = s.lstrip()
    if not s_stripped:
        return False
    if _META_PROSE.match(s_stripped):
        return False
    head = s_stripped[:400]
    return "\\" in head


_TABLE_REF = re.compile(r"\\ref\{(tab:[A-Za-z0-9_:\-]+)\}")
_TABLE_LABEL = re.compile(r"\\label\{(tab:[A-Za-z0-9_:\-]+)\}")


def _ensure_table_labels(body: str, known_ids: set[str]) -> str:
    """If body references \\ref{tab:X} where X ∈ known_ids but \\label{tab:X} is
    missing, insert the label into the first \\begin{table}. Runs once per id.
    """
    if not known_ids:
        return body
    refs = set(_TABLE_REF.findall(body))
    labs = set(_TABLE_LABEL.findall(body))
    missing = refs - labs
    if not missing:
        return body

    for tid in missing:
        if tid.removeprefix("tab:") not in known_ids:
            continue
        def _insert(match: re.Match, inserted=[False]):
            if inserted[0]:
                return match.group(0)
            inserted[0] = True
            return match.group(0) + f"\\label{{{tid}}}\n"
        body = re.sub(
            r"(\\begin\{table\*?\}[^\n]*(?:\n\s*\\caption\{[^}]*\})?)",
            _insert, body, count=1, flags=re.DOTALL,
        )
    return body


def _bib_keys(bib_path: pathlib.Path) -> set[str]:
    if not bib_path.exists():
        return set()
    return set(_BIB_KEY.findall(bib_path.read_text(encoding="utf-8")))


def _filter_citations(latex: str, allowed: set[str]) -> tuple[str, list[str]]:
    """Drop \\cite{X} where X is not in `allowed`. Returns (clean, removed_keys).

    Also scrubs any residual empty \\cite{} or \\cite{,} that would break
    pdflatex compilation ("I can't find file `,'"). Empty keys and literal
    placeholder "KEY"/"CITE_KEY" are always rejected.
    """
    removed: list[str] = []
    blacklist = {"", "key", "cite_key", "citekey", "xxx", "?"}

    def repl(match: re.Match) -> str:
        keys = [k.strip() for k in match.group(1).split(",")]
        good, bad = [], []
        for k in keys:
            if k.lower() in blacklist:
                bad.append(k)
            elif k in allowed:
                good.append(k)
            else:
                bad.append(k)
        removed.extend(bad)
        if not good:
            return ""  # strip the whole \cite{...}
        return match.group(0).replace(match.group(1), ", ".join(good))

    cleaned = _CITE_CALL.sub(repl, latex)
    # Safety net: any residual empty \cite{} or \cite{,}
    cleaned = _EMPTY_CITE.sub("", cleaned)
    return cleaned, removed


def _extract_json_object(text: str) -> Optional[dict]:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    chunk = m.group(1) if m else None
    if not chunk:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        chunk = m.group(1) if m else None
    if not chunk:
        return None
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        try:
            return json.loads(re.sub(r"[\x00-\x1F\x7F]", "", chunk))
        except json.JSONDecodeError:
            return None


# ── Data ─────────────────────────────────────────────────────────────


@dataclasses.dataclass
class Paper:
    title: str
    abstract: str
    sections: dict[str, str]


def _context(idea: dict, results_blob, hw_hint: str, bib_keys: set[str],
             section_key: Optional[str] = None) -> str:
    """Assemble the prompt context.

    `results_blob` may be:
      - None (no results; Phase 1 placeholder-style output)
      - str (free-form markdown; Phase 1 compat)
      - Results dataclass (Phase 3 structured — rendered with exact numbers)

    `section_key` filters tables: each table with an `owning_section` is only
    fully rendered when writing that section; otherwise it's listed as a
    reference-only label. Prevents duplicate `\\begin{table}` blocks across
    sections (the LLM happily re-emits tables it sees in its prompt).
    """
    parts = [
        "IDEA METADATA",
        "-------------",
        f"Title: {idea.get('Title', '')}",
        f"Hypothesis: {idea.get('Short Hypothesis', '')}",
        f"Abstract: {idea.get('Abstract', '')}",
        f"Experiments: {idea.get('Experiments', '')}",
        f"Risks & Limitations: {idea.get('Risk Factors and Limitations', '')}",
        "",
        f"HARDWARE CONTEXT: {hw_hint}",
        "",
        f"ALLOWED BIB KEYS (use only these): {sorted(bib_keys) or '(none)'}",
    ]
    if isinstance(results_blob, Results):
        parts += ["", results_blob.to_prompt_context(section_key=section_key),
                  "",
                  "STRICT: Only cite numbers that appear in the metrics or "
                  "tables above. Do NOT invent percentages, BLEU scores, "
                  "latencies, or other figures. If a number isn't in the "
                  "results, describe the setup qualitatively without it."]
    elif isinstance(results_blob, str) and results_blob.strip():
        parts += ["", "EXPERIMENT LOG", "--------------", results_blob]
    return "\n".join(parts)


# ── Section generation ──────────────────────────────────────────────


async def _gen_section(
    cfg: BackendConfig, key: str, context_fn: Callable[[str], str],
    model: Optional[str], critique: bool,
) -> tuple[str, str]:
    """Generate (optionally self-critiqued) LaTeX for one section."""
    instr = SECTION_PROMPTS[key]
    context = context_fn(key)
    user = (
        f"{context}\n\nTASK: {instr}\n\n"
        f"Return ONLY the LaTeX body (no \\section header)."
    )
    # Method / Experiments often need more room for equations + specs.
    max_out = 4000 if key in ("method", "experiments") else 2500
    draft = await acomplete(cfg, system=SECTION_SYSTEM, user=user, model=model,
                            temperature=0.3, max_tokens=max_out)
    draft = _sanitize_latex(draft.strip())
    if not critique:
        return key, draft

    rev_user = CRITIQUE_USER_TMPL.format(section=key, draft=draft)
    revised = await acomplete(cfg, system=CRITIQUE_SYSTEM, user=rev_user, model=model,
                              temperature=0.2, max_tokens=max_out)
    revised = _sanitize_latex(revised.strip())

    # Reject degenerate critique output and fall back to draft.
    if len(revised) < max(200, len(draft) * 0.4):
        log.warning("critique for %s too short (%d vs %d chars); keeping draft",
                    key, len(revised), len(draft))
        return key, draft
    if not _looks_like_latex(revised):
        log.warning("critique for %s looked like prose/commentary; keeping draft",
                    key)
        return key, draft
    return key, revised


async def _gen_all_sections(
    cfg: BackendConfig, context_fn: Callable[[str], str], model: Optional[str],
    sections: list[str], critique: bool, parallel: bool,
    concurrency: Optional[int] = None,
    progress: ProgressCallback = _noop_progress,
) -> dict[str, str]:
    total = len(sections)
    done = {"n": 0}  # mutable counter shared by closures

    def _emit_done(key: str, ok: bool) -> None:
        done["n"] += 1
        kind = "item" if ok else "warning"
        msg = key if ok else f"{key} (failed)"
        emit(progress, Progress(kind=kind, stage="section",
                                current=done["n"], total=total, message=msg))

    if parallel:
        limit = concurrency or recommended_concurrency(cfg)
        log.info("async gather concurrency=%d (sections=%d)", limit, total)
        sem = asyncio.Semaphore(limit)

        async def gated(key: str) -> tuple[str, str]:
            async with sem:
                try:
                    r = await _gen_section(cfg, key, context_fn, model, critique)
                    _emit_done(key, ok=True)
                    return r
                except Exception:
                    _emit_done(key, ok=False)
                    raise

        tasks = [gated(k) for k in sections]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        results = []
        for k in sections:
            try:
                r = await _gen_section(cfg, k, context_fn, model, critique)
                results.append(r)
                _emit_done(k, ok=True)
            except Exception as e:  # noqa: BLE001
                results.append(e)
                _emit_done(k, ok=False)
    out: dict[str, str] = {}
    for r in results:
        if isinstance(r, Exception):
            log.error("section failed: %s", r)
            continue
        k, body = r
        out[k] = body
    return out


# ── Coherence pass ──────────────────────────────────────────────────


def _coherence_pass(
    cfg: BackendConfig, sections: dict[str, str], model: Optional[str],
) -> dict[str, str]:
    concat = "\n\n".join(
        f"=== {k} ===\n{v}" for k, v in sections.items()
    )
    # Cap prompt size — transitions don't need full context of mega-long sections.
    if len(concat) > 40000:
        log.info("truncating coherence context %d → 40000", len(concat))
        concat = concat[:40000]
    user = COHERENCE_USER_TMPL.format(concat=concat)
    text, _ = complete(cfg, system=COHERENCE_SYSTEM, user=user, model=model,
                       temperature=0.2, max_tokens=3000)
    obj = _extract_json_object(text)
    if not isinstance(obj, dict):
        log.warning("coherence pass produced no parseable JSON; skipping")
        return sections
    updated = dict(sections)
    for k, v in obj.items():
        if k in updated and isinstance(v, str) and v.strip():
            updated[k] = _sanitize_latex(v.strip())
    log.info("coherence pass updated %d/%d sections", len(obj), len(sections))
    return updated


# ── Public API ──────────────────────────────────────────────────────


def write_paper(
    cfg: BackendConfig,
    *,
    idea: dict,
    results: Optional[str | Results] = None,
    model: Optional[str] = None,
    sections: Optional[list[str]] = None,
    hw: Optional[HardwareProfile] = None,
    critique: bool = True,
    coherence: bool = False,      # experimental
    parallel: bool = True,
    concurrency: Optional[int] = None,
    progress: ProgressCallback = _noop_progress,
) -> Paper:
    sections = sections or list(SECTION_PROMPTS.keys())
    hw = hw or detect_hardware()
    allowed = _bib_keys(TEMPLATE_DIR / "references.bib")
    hw_hint = hint_for_prompt(hw)

    def context_fn(section_key: str) -> str:
        return _context(idea, results, hw_hint, allowed, section_key=section_key)

    log.info("generating %d sections (parallel=%s critique=%s) on hardware=%s peak=%s",
             len(sections), parallel, critique, hw.tier, is_minimax_peak())
    raw_sections = asyncio.run(
        _gen_all_sections(cfg, context_fn, model, sections, critique, parallel,
                          concurrency=concurrency, progress=progress)
    )

    if coherence and len(raw_sections) >= 3:
        emit(progress, Progress(kind="stage_start", stage="coherence",
                                message=f"{len(raw_sections)} sections"))
        t_coh = time.time()
        raw_sections = _coherence_pass(cfg, raw_sections, model)
        emit(progress, Progress(kind="stage_end", stage="coherence",
                                meta={"duration_s": time.time() - t_coh}))

    # Cross-section duplicate-table removal. Runs BEFORE citation / label
    # fixing so the label-recovery pass doesn't attempt to re-label a block
    # we're about to delete.
    table_ownership: dict[str, str] = {}
    known_table_ids: set[str] = set()
    if isinstance(results, Results):
        known_table_ids = {t.id for t in results.tables}
        for t in results.tables:
            if t.owning_section:
                table_ownership[f"tab:{t.id}"] = t.owning_section
    raw_sections, dedup_events = _dedup_tables(
        raw_sections, table_ownership=table_ownership,
    )
    for e in dedup_events:
        emit(progress, Progress(kind="warning", stage="section",
                                message=f"dedup {e.get('reason')}: "
                                        f"{e.get('label') or '(no label)'} "
                                        f"in {e.get('found_in')}",
                                meta=e))

    # Strip citations whose keys aren't in the bib; ensure table labels.
    cleaned: dict[str, str] = {}
    for k, v in raw_sections.items():
        v2, removed = _filter_citations(v, allowed)
        if removed:
            log.warning("dropped %d unknown cite keys in %s: %s",
                        len(removed), k, sorted(set(removed)))
        if known_table_ids:
            before = set(_TABLE_LABEL.findall(v2))
            v2 = _ensure_table_labels(v2, known_table_ids)
            after = set(_TABLE_LABEL.findall(v2))
            if after - before:
                log.info("inserted table labels in %s: %s", k, sorted(after - before))
        cleaned[k] = v2

    # Title + abstract come from idea metadata (ideation step), not from
    # the per-section LLM pass, so they bypass the SANITIZE_PIPELINE. Run
    # them through explicitly — otherwise a prose `_` in the title or a
    # truncated inline equation in the abstract crashes pdflatex before
    # any \section{} even starts.
    title = _sanitize_latex(str(idea.get("Title") or "Untitled Research"))
    abstract = _sanitize_latex(str(idea.get("Abstract") or ""))

    return Paper(
        title=title,
        abstract=abstract,
        sections=cleaned,
    )


def render_tex(paper: Paper, template: str = "icml2024.tex.j2") -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)),
                      autoescape=False, trim_blocks=True, lstrip_blocks=True)
    tmpl = env.get_template(template)
    return tmpl.render(
        title=paper.title,
        abstract=paper.abstract,
        introduction=paper.sections.get("introduction", "Forthcoming."),
        related_work=paper.sections.get("related_work", "Forthcoming."),
        method=paper.sections.get("method", "Forthcoming."),
        experiments=paper.sections.get("experiments", "Forthcoming."),
        results=paper.sections.get("results", ""),
        discussion=paper.sections.get("discussion", "Forthcoming."),
        conclusion=paper.sections.get("conclusion", "Forthcoming."),
        bibliography=True,
    )


# ── pdflatex with log-driven retry ──────────────────────────────────


_LATEX_ERR = re.compile(r"^! (.+)$", re.MULTILINE)
_BAD_LINE = re.compile(r"^l\.(\d+)", re.MULTILINE)


def _parse_latex_log(log_path: pathlib.Path) -> list[str]:
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    errors = _LATEX_ERR.findall(text)
    lines = _BAD_LINE.findall(text)
    return [f"{e} (near line {l})" for e, l in zip(errors, lines)] or errors


def _run_latex(out_dir: pathlib.Path, pdflatex: str) -> tuple[int, list[str]]:
    r = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "paper.tex"],
        cwd=out_dir, capture_output=True, text=True, timeout=180,
    )
    errs = _parse_latex_log(out_dir / "paper.log")
    return r.returncode, errs


def _retry_failing_sections(
    cfg: BackendConfig, paper: Paper, errors: list[str], model: Optional[str],
    retry_cfg: Optional[BackendConfig] = None,
    retry_model: Optional[str] = None,
) -> Paper:
    """Ask the LLM to fix LaTeX of each section given the error list.
    Runs sections in parallel via asyncio.gather — one gather cap saves ~7x
    vs sequential when all sections need fixing.

    If `retry_cfg` is supplied, uses that backend/model for the fix pass
    instead of the main one. Intended for swapping to a stronger model on
    retry (e.g. `claude-opus-4-5` via hybrid) when the main backend is
    cheap/fast MiniMax. Caller is responsible for probing reachability
    before handing us a hybrid cfg — we don't swallow connection errors."""
    fix_cfg = retry_cfg or cfg
    fix_model = retry_model or model
    err_blob = "\n".join(f"- {e}" for e in errors[:12])

    async def _fix_one(key: str, body: str) -> tuple[str, str]:
        user = (
            f"SECTION: {key}\n\nLATEX ERRORS:\n{err_blob}\n\n"
            f"CURRENT BODY:\n---BODY START---\n{body}\n---BODY END---\n\n"
            "Return ONLY the corrected LaTeX body. If this section appears "
            "clean, return it unchanged verbatim."
        )
        try:
            text = await acomplete(fix_cfg, system=RETRY_SYSTEM, user=user,
                                   model=fix_model,
                                   temperature=0.1, max_tokens=2500)
            fixed = _sanitize_latex(text.strip())
            if not fixed or not _looks_like_latex(fixed):
                log.warning("retry output for %s not LaTeX-like; keeping original",
                            key)
                return key, body
            return key, fixed
        except Exception as e:  # noqa: BLE001
            log.warning("retry LLM failed for %s: %s", key, e)
            return key, body

    limit = recommended_concurrency(fix_cfg)

    async def _fix_all():
        sem = asyncio.Semaphore(limit)
        async def gated(k, v):
            async with sem:
                return await _fix_one(k, v)
        tasks = [gated(k, v) for k, v in paper.sections.items()]
        return await asyncio.gather(*tasks, return_exceptions=False)

    log.info("retry gather concurrency=%d", limit)
    results = asyncio.run(_fix_all())
    fixed = dict(paper.sections)
    for k, v in results:
        fixed[k] = v
    return Paper(title=paper.title, abstract=paper.abstract, sections=fixed)


def compile_pdf(
    tex_source: str, out_dir: pathlib.Path,
    bib_path: Optional[pathlib.Path] = None,
    progress: ProgressCallback = _noop_progress,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper.tex").write_text(tex_source, encoding="utf-8")
    bib_src = bib_path or (TEMPLATE_DIR / "references.bib")
    if bib_src.exists():
        shutil.copy(bib_src, out_dir / "references.bib")

    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex not found in PATH; install MacTeX / TeX Live")

    emit(progress, Progress(kind="item", stage="compile",
                            current=1, total=3, message="pdflatex pass 1"))
    rc, errs = _run_latex(out_dir, pdflatex)
    if shutil.which("bibtex") and (out_dir / "paper.aux").exists():
        emit(progress, Progress(kind="item", stage="compile",
                                message="bibtex"))
        subprocess.run(["bibtex", "paper"], cwd=out_dir,
                       capture_output=True, text=True, timeout=60)
    emit(progress, Progress(kind="item", stage="compile",
                            current=2, total=3, message="pdflatex pass 2"))
    _run_latex(out_dir, pdflatex)
    emit(progress, Progress(kind="item", stage="compile",
                            current=3, total=3, message="pdflatex pass 3"))
    rc, errs = _run_latex(out_dir, pdflatex)

    pdf = out_dir / "paper.pdf"
    if pdf.exists():
        return pdf
    if errs:
        log.error("pdflatex errors:\n  %s", "\n  ".join(errs[:10]))
    raise RuntimeError(f"pdflatex finished but {pdf} not produced; check paper.log")


def writeup(
    cfg: BackendConfig,
    *,
    idea: dict,
    out_dir: pathlib.Path,
    results: Optional[str | Results] = None,
    model: Optional[str] = None,
    skip_compile: bool = False,
    critique: bool = True,
    coherence: bool = False,      # experimental; current impl can over-rewrite sections
    parallel: bool = True,
    concurrency: Optional[int] = None,
    audit: bool = True,
    annotate_unverified_claims: bool = False,
    progress: ProgressCallback = _noop_progress,
    retry_cfg: Optional[BackendConfig] = None,
    retry_model: Optional[str] = None,
) -> dict:
    """End-to-end: idea → paper.tex → paper.pdf with Phase-2 quality passes.

    Phase 3: when `results` is a Results dataclass (or JSON loaded via
    results.load), run verify.audit() over the produced sections and attach
    a verification_report to the output. Set annotate_unverified_claims=True
    to mark unverified numbers in red in the PDF.
    """
    emit(progress, Progress(kind="stage_start", stage="writeup",
                            message=idea.get("Title", "")[:80]))
    t_w = time.time()
    paper = write_paper(
        cfg, idea=idea, results=results, model=model,
        critique=critique, coherence=coherence, parallel=parallel,
        concurrency=concurrency, progress=progress,
    )
    emit(progress, Progress(kind="stage_end", stage="writeup",
                            message=f"{len(paper.sections)} sections",
                            meta={"duration_s": time.time() - t_w}))

    # Phase 3 audit
    verification = None
    if audit:
        emit(progress, Progress(kind="stage_start", stage="verify"))
        t_v = time.time()
        r_for_audit = results if isinstance(results, Results) else None
        report = verify_audit(paper.sections, results=r_for_audit)
        verification = report.to_dict()
        emit(progress, Progress(
            kind="stage_end", stage="verify",
            message=f"{len(report.verified)}/{report.total_claims} claims verified",
            meta={"duration_s": time.time() - t_v,
                  "verification_rate": report.verification_rate,
                  "unverified_count": len(report.unverified)}))
        log.info("verification: %d/%d claims verified (rate=%.2f)",
                 len(report.verified), report.total_claims,
                 report.verification_rate)
        if annotate_unverified_claims and report.unverified:
            annotated = dict(paper.sections)
            by_section: dict[str, list] = {}
            for c in report.unverified:
                by_section.setdefault(c.section, []).append(c)
            for sec_name, claims in by_section.items():
                if sec_name in annotated:
                    annotated[sec_name] = annotate_unverified(annotated[sec_name], claims)
            paper = Paper(title=paper.title, abstract=paper.abstract,
                          sections=annotated)
    tex = render_tex(paper)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper.tex").write_text(tex, encoding="utf-8")
    (out_dir / "paper_meta.json").write_text(
        json.dumps({"title": paper.title, "abstract": paper.abstract,
                    "sections": list(paper.sections.keys())}, indent=2),
        encoding="utf-8",
    )
    if verification is not None:
        (out_dir / "verification_report.json").write_text(
            json.dumps(verification, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    result: dict = {"tex": str(out_dir / "paper.tex"), "pdf": None}
    if verification is not None:
        result["verification"] = {
            "report": str(out_dir / "verification_report.json"),
            "verification_rate": verification["verification_rate"],
            "total_claims": verification["total_claims"],
            "unverified_count": verification["unverified_count"],
        }
    if skip_compile:
        return result

    emit(progress, Progress(kind="stage_start", stage="compile"))
    t_c = time.time()
    try:
        pdf = compile_pdf(tex, out_dir, progress=progress)
        result["pdf"] = str(pdf)
        emit(progress, Progress(kind="stage_end", stage="compile",
                                message=str(pdf),
                                meta={"duration_s": time.time() - t_c,
                                      "pdf_path": str(pdf)}))
        return result
    except RuntimeError as first_err:
        log.warning("first compile failed: %s — attempting log-driven retry", first_err)
        emit(progress, Progress(kind="retry", stage="compile",
                                message=str(first_err)[:120]))
        errs = _parse_latex_log(out_dir / "paper.log")
        if not errs:
            result["error"] = str(first_err)
            emit(progress, Progress(kind="stage_end", stage="compile",
                                    message="failed (no log errors parsable)",
                                    meta={"duration_s": time.time() - t_c,
                                          "error": str(first_err)}))
            return result
        fixed = _retry_failing_sections(cfg, paper, errs, model,
                                        retry_cfg=retry_cfg,
                                        retry_model=retry_model)
        tex2 = render_tex(fixed)
        try:
            pdf = compile_pdf(tex2, out_dir, progress=progress)
            result["pdf"] = str(pdf)
            result["retry"] = {"errors": errs[:8]}
            emit(progress, Progress(kind="stage_end", stage="compile",
                                    message=f"recovered after retry ({len(errs)} errors)",
                                    meta={"duration_s": time.time() - t_c,
                                          "pdf_path": str(pdf)}))
        except RuntimeError as second_err:
            result["error"] = f"retry also failed: {second_err}"
            emit(progress, Progress(kind="stage_end", stage="compile",
                                    message="retry also failed",
                                    meta={"duration_s": time.time() - t_c,
                                          "error": str(second_err)}))
        return result
