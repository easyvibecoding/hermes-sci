"""Command-line interface.

    hermes-sci ideate   --topic "..." [--num-ideas N]
    hermes-sci writeup  --idea IDX --ideas-json FILE [--results-md FILE]
    hermes-sci review   --paper PATH
    hermes-sci pipeline --topic "..." [--num-ideas N] [--skip-experiment]
"""
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import Any, Optional

from .config import Backend, apply_env, probe_claude_proxy, resolve_backend
from .ideation import ideate as ideate_fn, save_ideas
from .orchestrator import run_pipeline
from .progress import _resolve_builtin as _resolve_progress
from .results import (
    ResultsSchemaError,
    load as results_load,
    validate as results_validate,
)
from .review import review as review_fn, save_review
from .writeup import writeup as writeup_fn


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("minimax", "hybrid"), default="minimax",
                   help="Hermes backend route (default: minimax)")
    p.add_argument("--model", default=None,
                   help="Override resolved model (default: Hermes config)")
    p.add_argument("--claude-proxy", default="http://127.0.0.1:9099",
                   help="URL for the Anthropic-compat claude -p shim (hybrid only)")
    p.add_argument("--progress", choices=("human", "jsonl", "off"), default="human",
                   help="stage-level progress sink (default: human)")
    p.add_argument("-v", "--verbose", action="store_true")


def _add_retry_common(p: argparse.ArgumentParser) -> None:
    """Retry-backend knobs for writeup/pipeline only (not ideate/review)."""
    p.add_argument("--retry-backend", choices=("same", "hybrid"), default="same",
                   help="backend for log-driven LaTeX retry after pdflatex "
                        "fails. 'same' (default) reuses --backend; 'hybrid' "
                        "routes the retry through claude -p for a stronger "
                        "fix pass. If the claude proxy is unreachable we "
                        "silently fall back to 'same' — users without "
                        "delegation keep working.")
    p.add_argument("--retry-model", default=None,
                   help="model to use on retry (default: --model, or "
                        "'claude-opus-4-5' when --retry-backend=hybrid)")


def _build_retry_cfg(args, primary_cfg, progress_cb):
    """Build a retry BackendConfig if opted in + reachable; else return None.

    Emits a progress warning when hybrid was requested but the proxy probe
    fails, so the user sees WHY the retry is downgrading."""
    if getattr(args, "retry_backend", "same") != "hybrid":
        return None, None
    if not probe_claude_proxy(args.claude_proxy):
        from .progress import Progress, emit
        emit(progress_cb, Progress(
            kind="warning", stage="compile",
            message=f"--retry-backend=hybrid requested but claude proxy at "
                    f"{args.claude_proxy} is unreachable — retry will use "
                    f"the primary backend instead",
        ))
        return None, None
    retry_cfg = resolve_backend(
        backend="hybrid", model_override=None,
        claude_proxy_url=args.claude_proxy,
    )
    retry_model = args.retry_model or "claude-opus-4-5"
    return retry_cfg, retry_model


def cmd_ideate(args) -> int:
    cfg = resolve_backend(backend=args.backend, model_override=args.model,
                          claude_proxy_url=args.claude_proxy)
    apply_env(cfg)
    progress = _resolve_progress(args.progress)
    ideas = ideate_fn(
        cfg, mode=args.mode,
        topic=args.topic,
        workshop_md_path=pathlib.Path(args.workshop) if args.workshop else None,
        num_ideas=args.num_ideas,
        reflect=not args.no_reflect,
        model=args.model,
        progress=progress,
    )
    out = pathlib.Path(args.output)
    save_ideas(ideas, out)
    print(f"✅ wrote {len(ideas)} ideas → {out}")
    return 0 if ideas else 1


def cmd_writeup(args) -> int:
    cfg = resolve_backend(backend=args.backend, model_override=args.model,
                          claude_proxy_url=args.claude_proxy)
    apply_env(cfg)
    ideas_data = json.loads(pathlib.Path(args.ideas_json).read_text(encoding="utf-8"))
    idea_list = ideas_data.get("ideas", ideas_data if isinstance(ideas_data, list) else [])
    if args.idx >= len(idea_list):
        print(f"❌ idx={args.idx} out of range (have {len(idea_list)} ideas)", file=sys.stderr)
        return 2
    idea = idea_list[args.idx]
    results_arg: Any = None
    if args.results_json:
        results_arg = results_load(args.results_json)
    elif args.results_md:
        results_arg = pathlib.Path(args.results_md).read_text(encoding="utf-8")
    out = pathlib.Path(args.output)
    progress = _resolve_progress(args.progress)
    retry_cfg, retry_model = _build_retry_cfg(args, cfg, progress)
    r = writeup_fn(
        cfg, idea=idea, out_dir=out, results=results_arg,
        model=args.model, skip_compile=args.skip_compile,
        critique=not args.no_critique,
        coherence=args.coherence,
        parallel=not args.no_parallel,
        concurrency=args.concurrency,
        annotate_unverified_claims=args.annotate_unverified,
        progress=progress,
        retry_cfg=retry_cfg,
        retry_model=retry_model,
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0 if r.get("pdf") or args.skip_compile else 1


def cmd_validate_results(args) -> int:
    """Schema-check a results.json. Exits 0 if valid, 1 otherwise."""
    path = pathlib.Path(args.path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        print(f"❌ cannot read {path}: {e}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"❌ {path}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}",
              file=sys.stderr)
        return 1
    try:
        results_validate(data)
    except ResultsSchemaError as e:
        print(f"❌ {path}: {e}", file=sys.stderr)
        return 1
    print(f"✅ {path} matches results.json schema "
          f"(metrics={len(data.get('metrics', []))}, "
          f"tables={len(data.get('tables', []))})")
    return 0


def cmd_review(args) -> int:
    cfg = resolve_backend(backend=args.backend, model_override=args.model,
                          claude_proxy_url=args.claude_proxy)
    apply_env(cfg)
    progress = _resolve_progress(args.progress)
    r = review_fn(cfg, paper=pathlib.Path(args.paper),
                  ensemble=args.ensemble, model=args.model,
                  progress=progress)
    out = pathlib.Path(args.output or (pathlib.Path(args.paper).with_suffix(".review.json")))
    save_review(r, out)
    print(f"✅ review → {out}  overall={r.get('Overall')} decision={r.get('Decision')}")
    return 0


def cmd_pipeline(args) -> int:
    cfg = resolve_backend(backend=args.backend, model_override=args.model,
                          claude_proxy_url=args.claude_proxy)
    apply_env(cfg)
    results_arg: Any = None
    if args.results_json:
        results_arg = results_load(args.results_json)
    elif args.results_md:
        results_arg = pathlib.Path(args.results_md).read_text(encoding="utf-8")
    progress = _resolve_progress(args.progress)
    retry_cfg, retry_model = _build_retry_cfg(args, cfg, progress)
    r = run_pipeline(
        cfg, topic=args.topic, out_dir=pathlib.Path(args.output),
        num_ideas=args.num_ideas, results=results_arg,
        skip_experiment=args.skip_experiment,
        skip_review=args.skip_review,
        skip_compile=args.skip_compile,
        model=args.model,
        critique=not args.no_critique,
        coherence=args.coherence,
        parallel=not args.no_parallel,
        concurrency=args.concurrency,
        annotate_unverified_claims=args.annotate_unverified,
        progress=progress,
        retry_cfg=retry_cfg,
        retry_model=retry_model,
    )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    return 0 if not r.get("error") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes-sci")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ideate", help="generate research ideas")
    _add_common(pi)
    pi.add_argument("--mode", choices=("open", "workshop"), default="open")
    pi.add_argument("--topic", help="topic string for open mode")
    pi.add_argument("--workshop", help="workshop CFP markdown path (workshop mode)")
    pi.add_argument("--num-ideas", type=int, default=5)
    pi.add_argument("--no-reflect", action="store_true")
    pi.add_argument("-o", "--output", required=True, help="ideas.json output path")
    pi.set_defaults(func=cmd_ideate)

    pw = sub.add_parser("writeup", help="generate paper from an idea")
    _add_common(pw)
    pw.add_argument("--ideas-json", required=True)
    pw.add_argument("--idx", type=int, default=0, help="idea index (default: 0)")
    pw.add_argument("--results-md", help="markdown file with experiment results (legacy)")
    pw.add_argument("--results-json", help="structured results JSON (preferred; "
                                            "enables numerical claim audit)")
    pw.add_argument("--annotate-unverified", action="store_true",
                    help="mark unverified numbers in red in the output PDF")
    pw.add_argument("--skip-compile", action="store_true")
    pw.add_argument("--no-critique", action="store_true",
                    help="disable per-section self-critique pass")
    pw.add_argument("--coherence", dest="coherence", action="store_true",
                    default=False,
                    help="enable experimental global coherence pass (may over-rewrite)")
    pw.add_argument("--no-parallel", action="store_true",
                    help="disable async parallel section generation")
    pw.add_argument("--concurrency", type=int, default=None,
                    help="override concurrency limit (default: auto — 1 during "
                         "MiniMax peak 15:00-17:30 Asia/Shanghai, 7 off-peak)")
    pw.add_argument("-o", "--output", required=True, help="output directory")
    _add_retry_common(pw)
    pw.set_defaults(func=cmd_writeup)

    pv = sub.add_parser("validate-results",
                        help="schema-check a results.json without running the pipeline")
    pv.add_argument("path", help="path to results.json")
    pv.set_defaults(func=cmd_validate_results)

    pr = sub.add_parser("review", help="peer-review a paper PDF")
    _add_common(pr)
    pr.add_argument("--paper", required=True)
    pr.add_argument("--ensemble", type=int, default=3)
    pr.add_argument("-o", "--output", help="review.json output (default: <paper>.review.json)")
    pr.set_defaults(func=cmd_review)

    pp = sub.add_parser("pipeline", help="end-to-end ideate → writeup → review")
    _add_common(pp)
    pp.add_argument("--topic", required=True)
    pp.add_argument("--num-ideas", type=int, default=3)
    pp.add_argument("--results-md", help="optional markdown of experiment results (legacy)")
    pp.add_argument("--results-json", help="structured results JSON (preferred)")
    pp.add_argument("--annotate-unverified", action="store_true")
    pp.add_argument("--skip-experiment", action="store_true", default=True,
                    help="(phase-1 default) skip the experiment stage")
    pp.add_argument("--skip-review", action="store_true")
    pp.add_argument("--skip-compile", action="store_true")
    pp.add_argument("--no-critique", action="store_true")
    pp.add_argument("--coherence", action="store_true", default=False,
                    help="enable experimental global coherence pass (opt-in; may over-rewrite)")
    pp.add_argument("--no-parallel", action="store_true")
    pp.add_argument("--concurrency", type=int, default=None)
    pp.add_argument("-o", "--output", required=True, help="output directory")
    _add_retry_common(pp)
    pp.set_defaults(func=cmd_pipeline)
    return p


def main() -> int:
    args = build_parser().parse_args()
    _setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
